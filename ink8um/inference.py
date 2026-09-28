"""Sliding-window ink prediction over a rendered surface volume (a stack of layer images).

Reproduces the recipe the released predictions were made with:

* layers are read once into an ``H x W x D`` uint8 stack, intensities clipped to [0, 200]
  (uint16 layers are first reduced to 8 bit, ``>> 8``, as the training loader does);
* optionally the depth axis is reversed (``stack[..., ::-1]``) -- never the XY axes;
* coverage mask = ``max over depth > 0`` followed by a 7x7 elliptical closing;
* only 64x64 tiles lying entirely inside the coverage mask are predicted, stride 21;
* tile input = uint8 * float32(1/255), fp16 autocast on CUDA, sigmoid;
* stitching = sum(prob * gaussian(64, sigma=1 std)/max) / number of overlapping tiles.

The result is a float32 probability map at the native pixel resolution of the layers.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LAYER_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def gkern(kernlen: int = 64, nsig: float = 1.0) -> np.ndarray:
    """2D Gaussian kernel built from the normal CDF (identical to the training-time kernel)."""
    x = np.linspace(-nsig, nsig, kernlen + 1)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    kern1d = np.diff(cdf)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def list_layer_files(layers_dir: Path) -> list[Path]:
    """Numerically sorted layer images (``00.tif``, ``01.tif``, ...) in a directory."""
    layers_dir = Path(layers_dir)
    files = [p for p in layers_dir.iterdir()
             if p.suffix.lower() in LAYER_EXTENSIONS and re.fullmatch(r"\d+", p.stem)]
    if not files:
        raise FileNotFoundError(f"No numbered layer images (00.tif, 01.tif, ...) in {layers_dir}")
    return sorted(files, key=lambda p: int(p.stem))


def _read_layer(path: Path) -> np.ndarray:
    image = None
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            image = tifffile.imread(path)
        except ValueError:  # compressed TIFF without the optional imagecodecs package
            image = None
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"Could not read {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{path}: unsupported layer dtype {image.dtype} (expected uint8 or uint16)")
    return image


def read_stack(layers_dir: Path, *, layer_start: Optional[int] = None, depth: int = 24,
               clip_max: int = 200) -> np.ndarray:
    """Read ``depth`` consecutive layers into an ``H x W x depth`` uint8 array.

    With ``layer_start=None`` the directory must contain exactly ``depth`` layers, or the
    centred ``depth`` layers are used.
    """
    files = list_layer_files(layers_dir)
    if layer_start is None:
        layer_start = max(0, (len(files) - depth) // 2)
    selected = files[layer_start:layer_start + depth]
    if len(selected) != depth:
        raise ValueError(f"Need {depth} layers starting at index {layer_start}, found {len(files)} in {layers_dir}")
    layers = []
    for path in selected:
        image = _read_layer(path)
        if layers and image.shape != layers[0].shape:
            raise ValueError(f"Layer shape mismatch: {path} {image.shape} vs {layers[0].shape}")
        layers.append(np.clip(image, 0, clip_max).astype(np.uint8, copy=False))
    return np.stack(layers, axis=2)


def coverage_mask(stack: np.ndarray) -> np.ndarray:
    coverage = (stack.max(axis=2) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(coverage, cv2.MORPH_CLOSE, kernel)


def tile_positions(mask: np.ndarray, tile_size: int = 64, stride: int = 21) -> list[tuple[int, int]]:
    """Top-left (x, y) of every tile that lies fully inside the mask."""
    inside = mask > 0
    height, width = inside.shape
    # Integral image: a tile is valid iff the number of inside pixels equals tile_size**2.
    integral = np.pad(inside.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    points = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            y1, x1 = y + tile_size, x + tile_size
            total = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            if total == tile_size * tile_size:
                points.append((x, y))
    return points


class _TileDataset(Dataset):
    def __init__(self, stack: np.ndarray, points: Sequence[tuple[int, int]], tile_size: int):
        self.stack, self.points, self.tile_size = stack, points, tile_size

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        x, y = self.points[index]
        tile = self.stack[y:y + self.tile_size, x:x + self.tile_size]
        chw = np.ascontiguousarray(tile.transpose(2, 0, 1))
        return torch.from_numpy(chw).float().mul_(np.float32(1.0 / 255.0)), np.asarray([x, y], dtype=np.int64)


@torch.inference_mode()
def predict_stack(model: torch.nn.Module, stack: np.ndarray, *, reverse: Optional[bool] = None,
                  stride: Optional[int] = None, batch_size: int = 16, num_workers: int = 4,
                  device: Optional[str] = None,
                  progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Predict an ``H x W`` ink probability map from an ``H x W x D`` uint8 stack."""
    tile_size = int(getattr(model, "tile_size", 64))
    stride = int(stride or getattr(model, "stride", tile_size // 3))
    if reverse is None:
        reverse = bool(getattr(model, "reverse_layers", False))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_depth = getattr(model, "in_depth", stack.shape[2])
    if stack.shape[2] != expected_depth:
        raise ValueError(f"Model expects {expected_depth} layers, stack has {stack.shape[2]}")
    if reverse:
        stack = np.ascontiguousarray(stack[..., ::-1])

    height, width = stack.shape[:2]
    points = tile_positions(coverage_mask(stack), tile_size, stride)
    numerator = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    if not points:
        return numerator
    blend = gkern(tile_size, 1)
    blend = (blend / blend.max()).astype(np.float32)

    model = model.to(device).eval()
    loader = DataLoader(_TileDataset(stack, points, tile_size), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"))
    done = 0
    for tiles, xy in loader:
        tiles = tiles.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            logits = model(tiles)
        probs = torch.sigmoid(logits).float()
        if probs.shape[-2:] != (tile_size, tile_size):
            probs = F.interpolate(probs, size=(tile_size, tile_size), mode="bilinear")
        probs = probs[:, 0].cpu().numpy()
        for p, (x, y) in zip(probs, xy.numpy()):
            numerator[y:y + tile_size, x:x + tile_size] += p * blend
            count[y:y + tile_size, x:x + tile_size] += 1.0
        done += len(probs)
        if progress is not None:
            progress(done, len(points))
    prediction = np.divide(numerator, count, out=np.zeros_like(numerator), where=count != 0)
    return np.clip(np.nan_to_num(prediction), 0, 1)


def predict_surface(model: torch.nn.Module, layers_dir: Path, *, layer_start: Optional[int] = None,
                    **kwargs) -> np.ndarray:
    """Read a layer directory and predict it. ``kwargs`` go to :func:`predict_stack`."""
    stack = read_stack(layers_dir, layer_start=layer_start, depth=int(getattr(model, "in_depth", 24)),
                       clip_max=int(getattr(model, "clip_max", 200)))
    return predict_stack(model, stack, **kwargs)


def save_prediction(prediction: np.ndarray, output: Path) -> None:
    """Save as 8-bit PNG/JPG (``prob * 255``), 16-bit TIFF, or float32 ``.npy`` by extension."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".npy":
        np.save(output, prediction.astype(np.float32))
    elif suffix in (".tif", ".tiff"):
        tifffile.imwrite(output, (prediction * 65535).round().astype(np.uint16), compression="zlib")
    elif suffix in (".png", ".jpg", ".jpeg"):
        if not cv2.imwrite(str(output), (prediction * 255).astype(np.uint8)):
            raise OSError(f"Could not write {output}")
    else:
        raise ValueError(f"Unsupported output format: {output}")
