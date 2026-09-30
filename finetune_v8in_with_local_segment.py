#!/usr/bin/env python3
"""Fine-tune v8-in on PHerc1447 w058+w060 plus the local PHerc0211 segment.

This keeps the author's released PHerc1447 fine-tuning recipe intact while using the
base v8-in training library that ships in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAINING_ROOT = REPO / "training"
TRAINING_ENTRY = TRAINING_ROOT / "lib" / "train_resnet3d.py"
BASE_WEIGHTS = REPO / "model.safetensors"
SURFACES_ROOT = REPO / "data" / "training_surfaces"
DEFAULT_SEGMENT_ID = "20260928000003"
DEFAULT_TRAIN = ["w058", "w060", DEFAULT_SEGMENT_ID]

PH1447_RECIPE = {
    "dataset_name": "8um_dataset",
    "description": "v8-in finetune on PHerc1447 w058+w060 plus one local PHerc0211 segment.",
    "wandb": {
        "enabled": False,
        "project": "ink-8um",
        "entity": None,
        "group": "pherc1447-finetune",
        "tags": ["ph1447", "v8in-finetune", "custom-segment"],
    },
    "segments": {
        "w058": {"base_path": "ph1447_refined", "layer_range": [0, 24], "reverse_layers": True},
        "w060": {"base_path": "ph1447_refined", "layer_range": [0, 24], "reverse_layers": True},
        DEFAULT_SEGMENT_ID: {"base_path": "PHerc0211_custom", "layer_range": [0, 24], "reverse_layers": False},
    },
    "training": {
        "objective": "erm",
        "sampler": "shuffle",
        "loss_mode": "batch",
        "save_every_epoch": True,
        "train_segments": list(DEFAULT_TRAIN),
        "val_segments": list(DEFAULT_TRAIN),
        "val_label_suffix": "",
        "val_mask_suffix": "",
        "filter_empty_tile": False,
        "data_backend": "tiff",
        "dataset_root": "<set by launcher>",
        "stitch_all_val": True,
        "stitch_downsample": 4,
        "stitch_train": False,
        "stitching_schedule": {
            "train_every_n_epochs": 1,
            "eval_every_n_epochs": 1,
            "eval_every_n_epochs_plus_one": False,
        },
        "train_label_suffix": "",
        "train_mask_suffix": "",
        "stitch_log_only_segments": [],
    },
    "group_dro": {"group_key": "base_path"},
    "training_hyperparameters": {
        "model": {
            "model_name": "Unet",
            "backbone": "resnet3d",
            "model_impl": "resnet3d_hybrid",
            "in_chans": 24,
            "encoder_depth": 5,
            "target_size": 1,
            "norm": "batch",
            "group_norm_groups": 32,
            "input_batchnorm": True,
            "input_upsample_to": 256,
            "input_upsample_depth_to": 96,
        },
        "training": {
            "size": 64,
            "tile_size": 64,
            "stride": 48,
            "train_batch_size": 4,
            "valid_batch_size": 16,
            "use_amp": True,
            "epochs": 10,
            "scheduler": "cosine",
            "lr": 1e-5,
            "min_lr": 1e-6,
            "weight_decay": 1e-6,
            "num_workers": 8,
            "layer_read_workers": 12,
            "seed": 130697,
            "eval_stitch_metrics": True,
            "eval_topological_metrics_every_n_epochs": 5,
            "eval_save_stitch_debug_images": False,
            "eval_wandb_media_downsample": 2,
            "accumulate_grad_batches": 8,
            "onecycle_pct_start": 0.05,
            "max_grad_norm": 1.0,
        },
        "augmentation": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.5,
            "shift_scale_rotate": {"rotate_limit": 360, "shift_limit": 0.15, "scale_limit": 0.1, "p": 0.75},
            "blur": {"types": ["GaussianBlur", "MotionBlur"], "p": 0.4},
            "coarse_dropout": {"p": 0.5, "max_holes": 2, "max_width_ratio": 0.2, "max_height_ratio": 0.2},
            "fourth_augment": {"p": 0.6, "min_crop_ratio": 0.9, "max_crop_ratio": 1.0, "cutout_max_count": 2, "cutout_p": 0.6},
            "brightness_contrast": {"p": 0.0},
        },
    },
}


def _safe_link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == target.resolve():
                return
        except FileNotFoundError:
            pass
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target, target_is_directory=target.is_dir())


def _require_surface(root: Path, segment_id: str) -> None:
    required = [
        root / segment_id / "layers" / "00.tif",
        root / segment_id / "labels" / "inklabels.png",
        root / segment_id / "labels" / "mask.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing files for {segment_id}: {missing}. Stage PHerc1447 data under {root}/w058 and {root}/w060 first."
        )


def _stage_data(data_root: Path, surfaces_root: Path, segment_ids: list[str]) -> None:
    for segment_id in segment_ids:
        _require_surface(surfaces_root, segment_id)
        source = surfaces_root / segment_id
        target = data_root / segment_id
        target.mkdir(parents=True, exist_ok=True)
        _safe_link(source / "layers", target / "layers")
        _safe_link(source / "labels" / "inklabels.png", target / f"{segment_id}_inklabels.png")
        _safe_link(source / "labels" / "mask.png", target / f"{segment_id}_mask.png")


def _write_config(output_dir: Path, init_path: Path, train_ids: list[str], val_ids: list[str], segment_id: str, enable_wandb: bool) -> Path:
    config = json.loads(json.dumps(PH1447_RECIPE))
    config["segments"] = {
        "w058": config["segments"]["w058"],
        "w060": config["segments"]["w060"],
        segment_id: {"base_path": "PHerc0211_custom", "layer_range": [0, 24], "reverse_layers": False},
    }
    config["training"]["dataset_root"] = str(output_dir / "data")
    config["training"]["train_segments"] = list(train_ids)
    config["training"]["val_segments"] = list(val_ids)
    config["training"]["init_ckpt_path"] = str(init_path.resolve())
    config["wandb"]["enabled"] = bool(enable_wandb)
    config["wandb"]["tags"] = list(dict.fromkeys(config["wandb"]["tags"] + [segment_id]))
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surfaces-root", type=Path, default=SURFACES_ROOT)
    parser.add_argument("--init", type=Path, default=BASE_WEIGHTS)
    parser.add_argument("--segment-id", default=DEFAULT_SEGMENT_ID)
    parser.add_argument("--train", nargs="+", default=list(DEFAULT_TRAIN))
    parser.add_argument("--val", nargs="+", default=None)
    parser.add_argument("--run-name", default="ph1447_plus_20260928000003")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.init.is_file():
        raise FileNotFoundError(f"missing pretrained checkpoint: {args.init}")
    if not TRAINING_ENTRY.is_file():
        raise FileNotFoundError(f"missing training entrypoint: {TRAINING_ENTRY}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = list(args.train)
    val_ids = list(args.val or train_ids)
    _stage_data(output_dir / "data", args.surfaces_root.resolve(), sorted(set(train_ids + val_ids)))
    config_path = _write_config(output_dir, args.init, train_ids, val_ids, args.segment_id, args.wandb)

    print(f"prepared data under {output_dir / 'data'}")
    print(f"config: {config_path}")
    print(f"checkpoints: {output_dir / 'runs'}")
    if args.prepare_only:
        return

    os.environ.update(
        INPUT_NORM_TYPE="instance",
        DECODER_TYPE="2d",
        NEG_WEIGHT="25",
        VAL_STRIDE="64",
        SCHED_WORLD_SIZE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    if not args.wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")

    argv = [
        sys.executable,
        str(TRAINING_ENTRY),
        "--metadata_json",
        str(config_path),
        "--outputs_path",
        str(output_dir),
        "--devices",
        "1",
        "--accelerator",
        "gpu",
        "--precision",
        "16-mixed",
        "--run_name",
        args.run_name,
    ]
    print("launching:", " ".join(argv), flush=True)
    os.chdir(TRAINING_ENTRY.parent)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()