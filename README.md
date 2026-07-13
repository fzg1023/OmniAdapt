# OmniAdapt This is the initial code. We will provide more refined code and more detailed instructions before September 1, 2026.

Official code package for **OmniAdapt: Full-Granularity Adaptive Fusion for Robust RGB-T Tracking**.

## Included

- `lib/`: model, training, testing, configuration, and utility source code
- `experiments/omniadapt/`: full-model and ablation YAML configurations
- `tracking/`: training and local-environment setup entry points
- `RGBT_workspace/`: standalone RGB-T evaluation scripts
- `docs/`: architecture, training, ablation, and paper-use notes

## Intentionally excluded

No model weights, datasets, logs, prior prediction files, cached bytecode, manuscript drafts, or visualization outputs are included.

## Installation

```bash
conda create -n omniadapt python=3.9 -y
conda activate omniadapt
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Model locations

Place model files only after obtaining the release package:

```text
pretrain/DropTrack.pth.tar
output/checkpoints/train/omniadapt/omniadapt_afg_full/OmniAdapt_ep0003.pth.tar
```

The first is required for training initialization. The second is the expected location of the final OmniAdapt checkpoint for evaluation.

## Data paths

Set `DATA_ROOT` and/or `LASHER_ROOT` for framework evaluation, or pass `--data_root` to the standalone evaluator. Datasets are not redistributed.

## Training

```bash
python tracking/train.py --script omniadapt --config omniadapt --save_dir ./output --mode multiple --nproc_per_node 1
```

## Evaluation

```bash
python RGBT_workspace/test.py \
  --checkpoint output/checkpoints/train/omniadapt/omniadapt_afg_full/OmniAdapt_ep0003.pth.tar \
  --dataset lasher --data_root /path/to/lasher --workers 4 --yaml_name omniadapt --epoch 3
```
