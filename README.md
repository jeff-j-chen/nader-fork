# nader-fork

Minimal setup for fine-tuning `v8-in` on:

- `w058` and `w060` from the public PHerc. 1447 surfaces dataset
- the local PHerc0211 segment already present in this repo at `data/training_surfaces/20260928000003`

Already present in this repo:

- `model.safetensors` and `config.json` for the pretrained `v8-in` checkpoint
- `data/training_surfaces/20260928000003/layers/00.tif` through `23.tif`
- `data/training_surfaces/20260928000003/labels/inklabels.png`
- `data/training_surfaces/20260928000003/labels/mask.png`
- `finetune_v8in_with_local_segment.py`
- `test_inference_copy.ipynb`

Install:

```bash
git lfs pull --include="model.safetensors"
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

Fetch only the public PHerc. 1447 training surfaces you need:

```bash
mkdir -p _tmp data/training_surfaces
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces _tmp/pherc1447
cd _tmp/pherc1447
git lfs pull --include="w058/layers/*,w058/labels/*,w060/layers/*,w060/labels/*"
cd ../..
mkdir -p data/training_surfaces/w058 data/training_surfaces/w060
cp -a _tmp/pherc1447/w058/. data/training_surfaces/w058/
cp -a _tmp/pherc1447/w060/. data/training_surfaces/w060/
rm -rf _tmp/pherc1447
```

Prepare only:

```bash
python3 finetune_v8in_with_local_segment.py --output-dir runs/pherc1447_plus_20260928000003 --prepare-only
```

Fine-tune:

```bash
python3 finetune_v8in_with_local_segment.py --output-dir runs/pherc1447_plus_20260928000003
```

Enable Weights & Biases metrics:

```bash
python3 finetune_v8in_with_local_segment.py --output-dir runs/pherc1447_plus_20260928000003 --wandb
```
