#!/usr/bin/env python
"""Convert a training checkpoint (.ckpt) into a model folder usable by predict.py / from_pretrained.

    python training/export_checkpoint.py runs/.../checkpoints/epochepoch=9.ckpt my-model [--reverse-layers]
    python predict.py --model my-model --layers /path/to/layers --output prediction.png
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ink8um import InkDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reverse-layers", action="store_true",
                        help="store reverse_layers=true (the model was trained on depth-reversed stacks)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k: v for k, v in state.items() if k.startswith(("backbone.", "decoder.", "normalization."))}
    model = InkDetector(reverse_layers=args.reverse_layers)
    model.load_state_dict(state, strict=True)
    model.save_pretrained(args.output_dir)
    print(f"wrote {args.output_dir}/config.json and model.safetensors ({len(state)} tensors)")


if __name__ == "__main__":
    main()
