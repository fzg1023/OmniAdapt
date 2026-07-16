#!/usr/bin/env python3
"""Data-free checkpoint and forward-pass smoke test for OmniAdapt."""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.config.omniadapt.config import cfg, update_config_from_file
from lib.models.omniadapt import build_omniadapt


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Strictly load an OmniAdapt checkpoint and run one synthetic RGB-T forward pass.')
    parser.add_argument('--checkpoint', required=True, help='checkpoint (.pth.tar) path')
    parser.add_argument('--config', default='std_full', help='YAML name without .yaml')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    config_path = os.path.join(
        PROJECT_ROOT, 'experiments', 'omniadapt', f'{args.config}.yaml')
    if not os.path.isfile(checkpoint_path):
        parser.error(f'checkpoint 文件不存在: {checkpoint_path}')
    if not os.path.isfile(config_path):
        parser.error(f'配置文件不存在: {config_path}')

    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('指定了 CUDA，但当前 PyTorch/CUDA 不可用')
    device = torch.device(device_name)

    update_config_from_file(config_path)
    started = time.perf_counter()
    model = build_omniadapt(cfg, training=False)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', mmap=True,
            weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'net' not in checkpoint:
        raise KeyError("checkpoint 中缺少 'net' 权重")

    try:
        model.load_state_dict(checkpoint['net'], strict=True, assign=True)
    except TypeError:
        model.load_state_dict(checkpoint['net'], strict=True)
    epoch = checkpoint.get('epoch', 'unknown')
    del checkpoint
    gc.collect()

    model = model.to(device).eval()
    template = [
        torch.randn(1, 6, cfg.TEST.TEMPLATE_SIZE, cfg.TEST.TEMPLATE_SIZE,
                    device=device)
        for _ in range(cfg.DATA.TEMPLATE.NUMBER)
    ]
    search = [
        torch.randn(1, 6, cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE,
                    device=device)
    ]
    with torch.inference_mode():
        output = model(template=template, search=search)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    required = {'pred_boxes', 'score_map', 'size_map', 'offset_map'}
    if len(output) != 1 or not required.issubset(output[0]):
        raise RuntimeError(f'输出结构不完整: {output}')
    tensors = {k: v for k, v in output[0].items() if torch.is_tensor(v)}
    non_finite = [k for k, value in tensors.items()
                  if not bool(torch.isfinite(value).all())]
    if non_finite:
        raise RuntimeError(f'输出包含 NaN/Inf: {non_finite}')

    shapes = {k: tuple(value.shape) for k, value in tensors.items()}
    print(f'[PASS] config={args.config} checkpoint_epoch={epoch} device={device}')
    print(f'[PASS] strict parameter load; finite outputs: {shapes}')
    print(f'[PASS] elapsed={time.perf_counter() - started:.3f}s')


if __name__ == '__main__':
    main()
