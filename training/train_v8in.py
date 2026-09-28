#!/usr/bin/env python
"""Train the v8-in base ink model from the v8 patch pack.

    hf download YoussefMoNader/ink-8um-v8-patchpack --repo-type dataset --local-dir v8_patchpack
    python training/train_v8in.py --patch-pack v8_patchpack --output-dir runs/v8in --devices 4

Recipe (configs/v8in.json): ResNet3D-50 initialised from Kinetics-700 + Moments-in-Time weights,
2D decoder, input InstanceNorm3d, 64x64x24 tiles upsampled in-model to 256x256x96, AdamW +
cosine schedule (peak lr 2e-5), 10 epochs, fp16 mixed precision, effective batch 128
(micro-batch 4 per GPU; gradient accumulation = 32 / devices). The released checkpoint is epoch 6
(the 7th epoch) of this schedule, trained on 4 GPUs.

Validation stitches predictions over the three validation segments of the pack
(0009B_ag132115, Frag3, Frag4) using the full-resolution labels in ``val_gt/``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EFFECTIVE_BATCH = 128
MICRO_BATCH = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch-pack", type=Path, required=True, help="local copy of the v8 patch pack dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=1, help="number of GPUs (DDP when > 1)")
    parser.add_argument("--backbone", type=Path, default=HERE / "r3d50_KM_200ep.safetensors",
                        help="Kinetics/MiT-pretrained ResNet3D-50 initialiser")
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "v8in.json")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases (uses your own login)")
    parser.add_argument("--smoke-test", type=int, default=None, metavar="N",
                        help="1 epoch of N training batches + full validation, to check the setup")
    args = parser.parse_args()

    pack = args.patch_pack.resolve()
    for name in ("segments.json", "train_images.npy", "train_labels.npy", "val_images.npy", "val_labels.npy"):
        if not (pack / name).is_file():
            raise FileNotFoundError(f"{pack / name} is missing; download the full patch pack first")
    if not args.backbone.is_file():
        raise FileNotFoundError(f"Backbone initialiser not found: {args.backbone}")
    per_step = MICRO_BATCH * args.devices
    if EFFECTIVE_BATCH % per_step:
        raise ValueError(f"--devices must divide {EFFECTIVE_BATCH // MICRO_BATCH}")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())

    # Stitched validation metrics read <root>/<id>/<id>_inklabels.png and <id>_mask.png.
    val_root = out / "val_gt"
    for segment in config["training"]["val_segments"]:
        folder = val_root / segment
        folder.mkdir(parents=True, exist_ok=True)
        for kind in ("inklabels", "mask"):
            link = folder / f"{segment}_{kind}.png"
            if not link.exists():
                link.symlink_to(pack / "val_gt" / f"{segment}_{kind}.png")

    config["training"]["dataset_root"] = str(val_root)
    config["training_hyperparameters"]["training"]["accumulate_grad_batches"] = EFFECTIVE_BATCH // per_step
    config["wandb"]["enabled"] = bool(args.wandb)
    if args.smoke_test:
        config["training_hyperparameters"]["training"]["epochs"] = 1
    resolved = out / "config.json"
    resolved.write_text(json.dumps(config, indent=2) + "\n")

    os.environ.update(
        INPUT_NORM_TYPE="instance",
        DECODER_TYPE="2d",
        PATCH_PACK_ROOT=str(pack),
        SCHED_WORLD_SIZE=str(args.devices),
        BACKBONE_PRETRAINED_PATH=str(args.backbone.resolve()),
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    entry = HERE / "lib" / "train_resnet3d.py"
    argv = [sys.executable, str(entry), "--metadata_json", str(resolved), "--outputs_path", str(out),
            "--devices", str(args.devices), "--accelerator", "gpu", "--precision", "16-mixed",
            "--run_name", "v8in"]
    if args.smoke_test:
        argv += ["--limit_train_batches", str(args.smoke_test)]
    print("launching:", " ".join(argv), flush=True)
    os.chdir(entry.parent)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
