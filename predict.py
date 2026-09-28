#!/usr/bin/env python
"""Predict ink on a rendered surface volume.

Examples
--------
    # weights next to this script (after `hf download <repo> --local-dir <dir>`)
    python predict.py --layers /path/to/segment/layers --output prediction.png

    # any released model, fetched from the Hugging Face Hub
    python predict.py --model YoussefMoNader/ink-8um-v8in --layers /path/to/layers --output prediction.png

``--layers`` is a directory of numbered layer images (``00.tif`` ... ``23.tif``; more layers
are fine, the central 24 or ``--layer-start`` are used). Outputs: ``.png`` (8-bit),
``.tif`` (16-bit) or ``.npy`` (float32); several ``--output`` values may be given.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = str(HERE) if (HERE / "model.safetensors").is_file() else "YoussefMoNader/ink-8um-v8in"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=Path, required=True, help="directory with 00.tif, 01.tif, ...")
    parser.add_argument("--output", type=Path, nargs="+", required=True, help=".png, .tif and/or .npy")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"local directory or Hub repo id (default: {DEFAULT_MODEL})")
    parser.add_argument("--layer-start", type=int, default=None, help="index of the first of the 24 layers (default: centred)")
    reverse = parser.add_mutually_exclusive_group()
    reverse.add_argument("--reverse", dest="reverse", action="store_true", default=None,
                         help="reverse the depth order of the layers")
    reverse.add_argument("--no-reverse", dest="reverse", action="store_false",
                         help="keep the depth order of the layers")
    parser.add_argument("--stride", type=int, default=None, help="tile stride in pixels (default from config: 21)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    args = parser.parse_args()

    from ink8um import InkDetector, predict_surface, save_prediction

    model = InkDetector.from_pretrained(args.model)
    reverse = model.reverse_layers if args.reverse is None else args.reverse
    print(f"model={args.model} reverse_layers={reverse} stride={args.stride or model.stride}", flush=True)

    started = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 10 or done == total:
            last[0] = time.time()
            print(f"  {done}/{total} tiles ({time.time() - started:.0f}s)", flush=True)

    prediction = predict_surface(model, args.layers, layer_start=args.layer_start, reverse=reverse,
                                 stride=args.stride, batch_size=args.batch_size, num_workers=args.workers,
                                 device=args.device, progress=progress)
    for output in args.output:
        save_prediction(prediction, output)
        print(f"wrote {output}", flush=True)
    print(f"done in {time.time() - started:.0f}s; shape={prediction.shape} mean={prediction.mean():.4f} "
          f"max={prediction.max():.3f}")


if __name__ == "__main__":
    main()
