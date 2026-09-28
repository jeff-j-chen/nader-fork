---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
- vesuvius-challenge
- herculaneum
- papyrology
- computed-tomography
- ink-detection
- resnet3d
datasets:
- YoussefMoNader/ink-8um-v8-patchpack
---

# v8-in: 8 µm ink-detection base model (ResNet3D-50 + 2D decoder)

v8-in detects ink on the rendered surface volumes of Herculaneum papyri scanned at ~8 µm. It is a
ResNet3D-50 encoder with a 2D U-Net decoder, initialised from video-pretrained (Kinetics) weights and
trained on the [v8 patch pack](https://huggingface.co/datasets/YoussefMoNader/ink-8um-v8-patchpack): 508k
labelled 64×64×24 patches from 16 segments across Scroll 1, Scroll 5, PHerc. 1667, PHerc. 0139,
PHerc. 0814, PHerc. 0500P2 and a fragment. "-in" refers to the input normalisation:
`InstanceNorm3d`, so each tile is normalised on its own statistics, which makes the model less
sensitive to scan-to-scan contrast differences than the earlier input-BatchNorm variant.

It is the starting point of the PHerc. 1447 finetune
[`ink-8um-v8in-pherc1447-loo-w062`](https://huggingface.co/YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062).

## Quick start

```bash
pip install -r requirements.txt            # torch, numpy, tifffile, opencv, huggingface_hub, safetensors
hf download YoussefMoNader/ink-8um-v8in --local-dir ink-8um-v8in --exclude "training/*"
python ink-8um-v8in/predict.py --layers /path/to/segment/layers --output prediction.png
```

`--layers` is a folder of numbered layer images (`00.tif`, `01.tif`, …) rendered along the
surface normal. With more than 24 layers the central 24 are used (or pick them with
`--layer-start`). The output can be `.png` (8-bit), `.tif` (16-bit) or `.npy` (float32
probabilities), and several outputs can be written at once.

The depth order matters: the model expects the layers ordered as in its training segments. If
your render stacks the normal the other way (the PHerc. 1447 surfaces in
[`ink-8um-pherc1447-surfaces`](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)
do), add `--reverse`. When unsure, run both and keep the one with coherent strokes.

From Python:

```python
import sys; sys.path.insert(0, "ink-8um-v8in")
from ink8um import InkDetector, predict_surface, save_prediction

model = InkDetector.from_pretrained("YoussefMoNader/ink-8um-v8in").cuda().eval()
prob = predict_surface(model, "/path/to/layers", reverse=False)   # (H, W) float32 in [0, 1]
save_prediction(prob, "prediction.png")

# or on raw tiles: (B, 24, 64, 64) float in [0, 200/255] -> (B, 1, 64, 64) logits
```

## Model

| | |
|---|---|
| Input | 24 layers × 64 × 64 px tile, uint8 clipped to [0, 200], × 1/255 |
| Input norm | `InstanceNorm3d(1, affine=True)` |
| In-model resampling | trilinear to 96 × 256 × 256 (depth × H × W) |
| Encoder | ResNet3D-50 (Bottleneck [3, 4, 6, 3]), 1 input channel |
| Depth collapse | max over depth at each of the 4 encoder levels |
| Decoder | 2D U-Net-style (conv3×3–BN–ReLU per level), 1×1 logit head |
| Output | 64 × 64 ink logits (tile resolution) |
| Parameters | 83.4 M |
| Weights | `model.safetensors` (fp32), the epoch-6 training checkpoint without optimiser state |

Full-surface inference (`ink8um/inference.py`, as used for all released predictions): coverage
mask = `max over depth > 0` closed with a 7×7 ellipse; 64 px tiles at stride 21 kept only if fully
inside the mask; fp16 autocast; sigmoid; tiles weighted by a 64×64 Gaussian (±1σ across the tile, peak 1)
and the weighted sum divided by the per-pixel tile count. On the 4668 × 8864 px
w062 surface (69,794 tiles) this takes ~15-20 min on one recent GPU at batch 16 (~11 GB).

## Training

| | |
|---|---|
| Data | v8 patch pack: 508,006 train / 43,249 validation patches (64 px windows on a 32 px grid inside labelled masks; empty-label tiles filtered for training) |
| Validation | 0009B_ag132115, Frag3, Frag4 (stitched full-segment metrics) |
| Initialisation | ResNet3D-50 pretrained on Kinetics-700 + Moments in Time ([3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch), `r3d50_KM_200ep`); conv1 summed over RGB → 1 channel |
| Loss | 0.5 · Dice + 0.5 · BCE with label smoothing 0.25 |
| Optimiser | AdamW (β = 0.9, 0.999), weight decay 1e-6 (not on biases/norms), grad-norm clip 1.0 |
| Schedule | 10 epochs planned, per-step cosine from 2e-5 to 1e-6, no warm-up |
| Batch | effective 128 (4 per GPU × 4 GPUs × 8 accumulation), fp16 mixed precision |
| Augmentation | flips, shift/scale/rotate (±360°), Gaussian/motion blur, coarse dropout; depth jitter (90–100 % of layers re-pasted at a random depth offset, up to 2 layers zeroed) |
| Released | epoch 6 (the 7th epoch, global step 27,783) |

Reproduce with the code in `training/`:

```bash
hf download YoussefMoNader/ink-8um-v8in --local-dir ink-8um-v8in
hf download YoussefMoNader/ink-8um-v8-patchpack --repo-type dataset --local-dir v8_patchpack   # ~56 GB
pip install -r ink-8um-v8in/training/requirements.txt
python ink-8um-v8in/training/train_v8in.py --patch-pack v8_patchpack --output-dir runs/v8in --devices 4
```

`--devices 1` keeps the effective batch at 128 through 32-step accumulation. `--smoke-test N`
runs one epoch of N batches plus full validation to check the setup. Checkpoints for every epoch
are written to `runs/v8in/runs/<run>/checkpoints/`. Weights & Biases logging is off unless
`--wandb` is given. The full training library is in `training/lib/` (entry point
`train_resnet3d.py`, recipe in `training/configs/v8in.json`).

## Files

```
config.json, model.safetensors     model config + weights (InkDetector.from_pretrained)
ink8um/                            standalone model + inference code (no training dependencies)
predict.py                         command-line inference
training/train_v8in.py             base-model training entry point
training/configs/v8in.json         training recipe
training/r3d50_KM_200ep.safetensors  Kinetics/MiT ResNet3D-50 initialiser (MIT, Hara et al.)
training/lib/                      training library (PyTorch Lightning)
```

The standalone `InkDetector` matches the training implementation exactly: identical parameter
names, and bit-identical logits on real tiles in both fp32 and fp16 autocast.

## Related

- Finetuned model: [YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062](https://huggingface.co/YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062)
- Training data: [YoussefMoNader/ink-8um-v8-patchpack](https://huggingface.co/datasets/YoussefMoNader/ink-8um-v8-patchpack)
- PHerc. 1447 surfaces and predictions, including this model's predictions on w058, w060 and w062 (`predictions/v8in.png`, run with `--reverse`): [YoussefMoNader/ink-8um-pherc1447-surfaces](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)
- Vesuvius Challenge data: <https://scrollprize.org/data> · code base: <https://github.com/ScrollPrize/villa>

## License

Code and weights: **MIT**. The backbone initialiser is from
[3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) (MIT). The training data
are derived from Vesuvius Challenge scans, distributed under **CC BY-NC 4.0**.
