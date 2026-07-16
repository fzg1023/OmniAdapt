#!/usr/bin/env python3
"""
OmniAdapt — LasHeR / RGBT234 / RGBT210 / GTOT Multi-process Parallel Testing
============================================================================
Features:
  • N independent spawn subprocesses, each loads its own model copy
  • Sequences sorted by frame count then round-robin assigned for load balance
  • Subprocesses return results via multiprocessing.Queue
  • Main process prints real-time progress table with metrics
  • Saves per_seq_metrics.csv (per-sequence details)
  • Appends to eval_history.csv (multi-epoch summary)

Usage:
  python tracking/test.py \
      --checkpoint /path/to/OmniAdapt_ep0012.pth.tar \
      --dataset lasher \
      --workers 1
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from typing import Dict, List

import cv2
import numpy as np

# ── 项目根目录 ────────────────────────────────────────────────────────────────
_PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PRJ_ROOT not in sys.path:
    sys.path.insert(0, _PRJ_ROOT)
os.environ.setdefault('OMNIADAPT_ROOT', _PRJ_ROOT)

# ══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ══════════════════════════════════════════════════════════════════════════════

DATASET_CFG = {
    'lasher': {
        'root':   '/home/fzg/data/lasher/testingset',
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'init.txt',
        'gt_all': 'visible.txt',
    },
    'lasher_val': {
        'root':   '/home/fzg/data/lasher/trainingset',
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'init.txt',
        'gt_all': 'visible.txt',
    },
    'rgbt234': {
        'root':   '/home/fzg/data/RGBT234',
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'groundTruth.txt',
        'gt_all': 'groundTruth.txt',
    },
    'rgbt210': {
        'root':   '/home/fzg/data/RGBT210',
        'rgb':    'visible',
        'tir':    'infrared',
        'gt':     'init.txt',
        'gt_all': 'init.txt',
    },
    'gtot': {
        'root':   '/home/fzg/data/GTOT',
        'rgb':    'v',
        'tir':    'i',
        'gt':     'groundTruth_v.txt',
        'gt_all': 'groundTruth_v.txt',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# IO 工具
# ══════════════════════════════════════════════════════════════════════════════

def _list_frames(seq_dir: str, modal: str) -> List[str]:
    d = os.path.join(seq_dir, modal)
    if not os.path.isdir(d):
        return []
    files = sorted(
        f for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return [os.path.join(d, f) for f in files]


def _imread(path: str, flags: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _read_frame(path: str, is_tir: bool = False) -> np.ndarray:
    """读取一帧，TIR 支持 uint16，统一返回 3 通道 RGB uint8。"""
    if is_tir:
        img = _imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f'cv2.imread failed: {path}')
        if img.dtype == np.uint16:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-6) * 255
                   ).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = _imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'cv2.imread failed: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _read_gt(path: str, dataset: str = '') -> List[List[float]]:
    """读取 GT 文件，支持逗号/空格分隔。
    LasHeR/RGBT234/RGBT210: (x, y, w, h)
    GTOT: (x1, y1, x2, y2) → 自动转换为 (x, y, w, h)
    """
    is_gtot = (dataset == 'gtot')
    if not os.path.isfile(path):
        return []
    bboxes = []
    with open(path, 'rb') as fb:
        raw = fb.read()
    if b'\x00' in raw:
        return []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        line = line.strip().replace(',', ' ').replace('\t', ' ')
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vals = [float(v) for v in parts[:4]]
            # GTOT 格式 (x1,y1,x2,y2) → (x,y,w,h)，仅对 GTOT 数据集转换
            if is_gtot:
                vals = [vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]]
            bboxes.append(vals)
        except ValueError:
            continue
    return bboxes


def _count_frames(seq_dir: str, modal: str = 'visible') -> int:
    return len(_list_frames(seq_dir, modal))


def _get_seq_dirs(dataset: str) -> List[str]:
    dc = DATASET_CFG.get(dataset)
    if dc is None:
        raise ValueError(f'未知数据集: {dataset}，支持: {list(DATASET_CFG)}')
    root = dc['root']
    if not os.path.isdir(root):
        raise FileNotFoundError(f'数据集目录不存在: {root}')
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


# ══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════════════════════

_SUCCESS_THRESHOLDS   = np.linspace(0, 1, 21)
_METRICS = ['AO', 'SS', 'SR50', 'SR75', 'PS', 'NPS']


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.


def _compute_metrics(preds, gts) -> dict:
    """返回单序列指标。"""
    valid = [(p, g) for p, g in zip(preds, gts)
             if len(p) >= 4 and len(g) >= 4 and g[2] > 0 and g[3] > 0]
    if not valid:
        return dict(AO=-1., SS=-1., SR50=-1., SR75=-1.,
                    PS=-1., NPS=-1., n_valid=0)

    ious, dists, nd = [], [], []
    for p, g in valid:
        ious.append(_iou(p, g))
        cx_p = p[0] + p[2] / 2;  cy_p = p[1] + p[3] / 2
        cx_g = g[0] + g[2] / 2;  cy_g = g[1] + g[3] / 2
        d = float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))
        dists.append(d)
        nd.append(d / float(np.sqrt(g[2] * g[3])) if g[2] * g[3] > 0 else 0.)

    ia = np.array(ious,  dtype=np.float64)
    da = np.array(dists, dtype=np.float64)
    na = np.array(nd,    dtype=np.float64)

    sr_curve = np.array([(ia >= t).mean() for t in _SUCCESS_THRESHOLDS])
    ss_auc   = float(np.trapz(sr_curve, _SUCCESS_THRESHOLDS))

    return dict(
        AO   = float(ia.mean()),
        SS   = ss_auc,
        SR50 = float((ia >= 0.50).mean()),
        SR75 = float((ia >= 0.75).mean()),
        PS   = float((da <= 20.).mean()),
        NPS  = float((na <= 0.5).mean()),
        n_valid = len(valid),
    )


def _save_pred(preds: list, seq_name: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{seq_name}.txt'), 'w') as f:
        for b in preds:
            f.write(','.join(f'{v:.4f}' for v in b) + '\n')


def _append_csv_row(csv_path: str, fields: List[str], row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in fields})


# ══════════════════════════════════════════════════════════════════════════════
# 子进程 worker
# ══════════════════════════════════════════════════════════════════════════════

def _worker(worker_id: int,
            seq_dirs: List[str],
            result_dir: str,
            checkpoint: str,
            yaml_name: str,
            epoch: int,
            dataset: str,
            tcsr_enable: bool,
            tcsr_alpha: float,
            tcsr_sigma: float,
            mem_enable: bool,
            torch_threads: int,
            out_q: mp.Queue):
    """
    spawn 子进程入口。
    独立加载 OmniAdapt 模型，串行处理分配到的序列子集。
    """
    try:
        import torch
        import lib.test.parameter.omniadapt as rgbt_params
        from lib.test.tracker.omniadapt import OmniAdapt

        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(torch_threads)
        
        params = rgbt_params.parameters(yaml_name, epoch)
        # override checkpoint path
        params.checkpoint = checkpoint
        # ── TCSR 参数 override ───────────────────────────────────────
        if tcsr_enable:
            if not hasattr(params.cfg.TEST, 'TCSR'):
                from easydict import EasyDict as edict
                params.cfg.TEST.TCSR = edict()
            params.cfg.TEST.TCSR.ENABLE = True
            params.cfg.TEST.TCSR.ALPHA = tcsr_alpha
            params.cfg.TEST.TCSR.SIGMA = tcsr_sigma
        # ── Memory 参数 override ──────────────────────────────────────
        if mem_enable:
            if not hasattr(params.cfg.TEST, 'MEMORY'):
                from easydict import EasyDict as edict
                params.cfg.TEST.MEMORY = edict()
            params.cfg.TEST.MEMORY.ENABLE = True
        # ───────────────────────────────────────────────────────────────
        tracker = OmniAdapt(params)
        out_q.put(('ready', worker_id))
    except Exception:
        out_q.put(('init_error', worker_id, traceback.format_exc()))
        return

    dc = DATASET_CFG[dataset]

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        t0 = time.perf_counter()
        try:
            rgb_paths = _list_frames(seq_dir, dc['rgb'])
            tir_paths = _list_frames(seq_dir, dc['tir'])
            gt_all = _read_gt(os.path.join(seq_dir, dc.get('gt_all', dc['gt'])), dataset)
            if not gt_all:
                gt_all = _read_gt(os.path.join(seq_dir, dc['gt']), dataset)

            n = min(len(rgb_paths), len(tir_paths), len(gt_all))
            if n < 1:
                raise ValueError('no valid frames/gt')

            # 第 0 帧：初始化 — RGB+TIR 拼接为 HxWx6
            rgb0 = _read_frame(rgb_paths[0], is_tir=False)
            tir0 = _read_frame(tir_paths[0], is_tir=True)
            image0 = np.concatenate([rgb0, tir0], axis=2)  # HxWx6
            tracker.initialize(image0, {'init_bbox': gt_all[0]})
            preds = [list(gt_all[0])]

            t_track = time.perf_counter()
            for fi in range(1, n):
                rgb_f = _read_frame(rgb_paths[fi], is_tir=False)
                tir_f = _read_frame(tir_paths[fi], is_tir=True)
                image_f = np.concatenate([rgb_f, tir_f], axis=2)  # HxWx6
                out = tracker.track(image_f)
                preds.append(list(out['target_bbox']))

            elapsed = time.perf_counter() - t0
            fps     = (n - 1) / max(time.perf_counter() - t_track, 1e-6)

            _save_pred(preds, seq_name, result_dir)
            m = _compute_metrics(preds, gt_all[:n])
            m.update(seq_name=seq_name, fps=float(fps), elapsed=float(elapsed))
            out_q.put(('result', worker_id, m))

        except Exception:
            elapsed = time.perf_counter() - t0
            out_q.put(('seq_error', worker_id, seq_name,
                       traceback.format_exc(), float(elapsed)))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser('OmniAdapt 多进程测试')
    p.add_argument('--yaml_name', default='std_full',
                   help='yaml config name (experiments/omniadapt/<name>.yaml)')
    p.add_argument('--checkpoint', required=True,
                   help='模型权重文件路径')
    p.add_argument('--dataset',    default='lasher',
                   choices=list(DATASET_CFG))
    p.add_argument('--data_root',  default='',
                   help='override dataset root; for lasher, a parent containing testingset is also accepted')
    p.add_argument('--save_dir',
                   default=os.path.join(_PRJ_ROOT, 'output', 'results'))
    p.add_argument('--workers',    type=int, default=1,
                   help='并行 worker 进程数（每个独立加载模型）')
    p.add_argument('--torch_threads', type=int, default=1,
                   help='CPU/OpenCV/PyTorch thread count per worker')
    p.add_argument('--epoch',      type=int, default=12,
                   help='权重 epoch 号（用于结果目录命名）')
    p.add_argument('--tag',        default='',
                   help='额外标签，附加到结果目录名（如 _TSSP, _noTSSP）')
    p.add_argument('--sequence',   default='',
                   help='只跑单条序列（调试用）')
    p.add_argument('--max_sequences', type=int, default=0,
                   help='run at most N sorted sequences; 0 means all')
    p.add_argument('--tcsr',       action='store_true', default=False,
                   help='启用 TCSR 轨迹引导 Score 精修')
    p.add_argument('--tcsr_alpha', type=float, default=0.10,
                   help='TCSR 运动先验强度 [0.05, 0.30]')
    p.add_argument('--tcsr_sigma', type=float, default=2.0,
                   help='TCSR 高斯先验宽度 [1.0, 4.0]')
    p.add_argument('--memory',     action='store_true', default=False,
                   help='启用模板记忆库 (Plan C)')
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        p.error(f'checkpoint 文件不存在: {args.checkpoint}')
    yaml_file = os.path.join(_PRJ_ROOT, 'experiments', 'omniadapt',
                             f'{args.yaml_name}.yaml')
    if not os.path.isfile(yaml_file):
        p.error(f'配置文件不存在: {yaml_file}')
    if args.workers < 1:
        p.error('--workers 必须大于或等于 1')

    args.torch_threads = max(1, int(args.torch_threads))
    os.environ['OMP_NUM_THREADS'] = str(args.torch_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.torch_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.torch_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.torch_threads)
    cv2.setNumThreads(args.torch_threads)

    if args.data_root:
        data_root = os.path.abspath(args.data_root)
        if args.dataset == 'lasher' and os.path.isdir(os.path.join(data_root, 'testingset')):
            data_root = os.path.join(data_root, 'testingset')
        elif args.dataset == 'lasher_val' and os.path.isdir(os.path.join(data_root, 'trainingset')):
            data_root = os.path.join(data_root, 'trainingset')
        DATASET_CFG[args.dataset]['root'] = data_root

    # ── 结果目录: save_dir/<dataset>/<yaml_name>_<epoch><tag>/ ─────────────────
    save_name = f'{args.yaml_name}_{args.epoch}{args.tag}'
    if args.tcsr:
        save_name += f'_tcsr_a{args.tcsr_alpha:.2f}_s{args.tcsr_sigma:.1f}'
    result_dir = os.path.join(args.save_dir, args.dataset, save_name)
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 80)
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  YAML       : {args.yaml_name}')
    print(f'  Epoch      : {args.epoch}')
    print(f'  Dataset    : {args.dataset}')
    print(f'  Data root  : {DATASET_CFG[args.dataset]["root"]}')
    print(f'  Result dir : {result_dir}')
    print(f'  Workers    : {args.workers}')
    print(f'  Threads    : {args.torch_threads}')
    if args.tcsr:
        print(f'  TCSR       : ENABLED  α={args.tcsr_alpha}  σ={args.tcsr_sigma}')
    print('=' * 80)

    # ── 收集并过滤序列 ────────────────────────────────────────────────────────
    seq_dirs = _get_seq_dirs(args.dataset)
    if args.sequence:
        seq_dirs = [d for d in seq_dirs
                    if os.path.basename(d) == args.sequence]
    # lasher_val: 从 train 目录中按 lasher_val.txt 过滤
    if args.dataset == 'lasher_val':
        val_list_file = os.path.join(_PRJ_ROOT, 'lib', 'train', 'data_specs', 'lasher_val.txt')
        if os.path.isfile(val_list_file):
            with open(val_list_file) as f:
                val_names = set(line.strip() for line in f if line.strip())
            seq_dirs = [d for d in seq_dirs if os.path.basename(d) in val_names]
            print(f'[INFO] lasher_val: 过滤后 {len(seq_dirs)} 条序列')
    n_seqs = len(seq_dirs)
    if n_seqs == 0:
        print('[ERROR] 没有找到任何序列，请检查数据集路径和 --sequence 参数。')
        sys.exit(1)

    # ── 按帧数升序后 round-robin 分配 ─────────────────────────────────────────
    dc = DATASET_CFG[args.dataset]
    sorted_dirs = sorted(seq_dirs,
                         key=lambda d: _count_frames(d, dc['rgb']))
    if args.max_sequences > 0:
        sorted_dirs = sorted_dirs[:args.max_sequences]
        n_seqs = len(sorted_dirs)
    n_workers   = min(args.workers, n_seqs)
    chunks: List[List[str]] = [[] for _ in range(n_workers)]
    for i, d in enumerate(sorted_dirs):
        chunks[i % n_workers].append(d)
    chunks    = [c for c in chunks if c]
    n_workers = len(chunks)

    print(f'[INFO] 共 {n_seqs} 条序列，启动 {n_workers} 个 worker')
    print(f'       每 worker 约 {max(len(c) for c in chunks)} 条序列')

    # ── 表头 ─────────────────────────────────────────────────────────────────
    HDR = (f"\n{'#':<7} {'序列名':<32} "
           f"{'AO':>6} {'SS':>6} {'SR50':>6} {'SR75':>6} "
           f"{'PS':>6} {'NPS':>6} {'FPS':>6} {'耗时s':>7}")
    SEP = '-' * (len(HDR) - 1)
    print(HDR)
    print(SEP)

    # ── 启动子进程 ────────────────────────────────────────────────────────────
    out_q: mp.Queue = mp.Queue()
    procs = []
    for wid, chunk in enumerate(chunks):
        proc = mp.Process(
            target=_worker,
            args=(wid, chunk, result_dir,
                  args.checkpoint, args.yaml_name, args.epoch,
                  args.dataset,
                  args.tcsr, args.tcsr_alpha, args.tcsr_sigma,
                  args.memory,
                  args.torch_threads,
                  out_q),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    # ── 等待所有 worker 完成模型加载 ─────────────────────────────────────────
    print(f'\n[INFO] 等待 {n_workers} 个 worker 加载模型...', flush=True)
    ready = 0
    init_errors = []
    while ready < n_workers:
        try:
            msg = out_q.get(timeout=5)
        except queue.Empty:
            failed = [p for p in procs
                      if p.exitcode is not None and p.exitcode != 0]
            if failed:
                init_errors.append(
                    'worker 进程异常退出: ' +
                    ', '.join(f'pid={p.pid}, exitcode={p.exitcode}' for p in failed))
                break
            continue
        if msg[0] == 'ready':
            ready += 1
            print(f'[INFO]   worker {msg[1]} 就绪 ({ready}/{n_workers})', flush=True)
        elif msg[0] == 'init_error':
            print(f'[ERROR]  worker {msg[1]} 加载失败:\n{msg[2]}', flush=True)
            init_errors.append(msg[2])
            ready += 1
    if init_errors:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
        sys.exit('[ERROR] 模型加载失败，测试已终止。')
    print(f'[INFO] 所有 worker 就绪，开始追踪...\n', flush=True)

    # ── 收集结果 ──────────────────────────────────────────────────────────────
    all_recs: Dict[str, dict] = {}
    done  = 0
    t_all = time.perf_counter()
    per_seq_fields = ['seq_name'] + _METRICS + ['n_valid', 'fps', 'elapsed']
    partial_csv_path = os.path.join(result_dir, 'per_seq_metrics_partial.csv')
    if os.path.isfile(partial_csv_path):
        os.remove(partial_csv_path)

    while done < n_seqs:
        msg = out_q.get()
        if msg[0] == 'result':
            _, wid, m = msg
            done += 1
            all_recs[m['seq_name']] = m
            _append_csv_row(partial_csv_path, per_seq_fields, m)
            if m['AO'] >= 0:
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<32s} "
                    f"{m['AO']:6.3f} {m['SS']:6.3f} "
                    f"{m['SR50']:6.3f} {m['SR75']:6.3f} "
                    f"{m['PS']:6.3f} {m['NPS']:6.3f} "
                    f"{m['fps']:6.1f} {m['elapsed']:7.1f}",
                    flush=True)
            else:
                print(
                    f"[{done:03d}/{n_seqs:03d}] {m['seq_name']:<32s} "
                    f"{'---':>6} {'---':>6} {'---':>6} {'---':>6} "
                    f"{'---':>6} {'---':>6} "
                    f"{m.get('fps', 0.):6.1f} {m.get('elapsed', 0.):7.1f}",
                    flush=True)

        elif msg[0] == 'seq_error':
            _, wid, seq_name, tb, elapsed = msg
            done += 1
            err_line = tb.strip().splitlines()[-1][:80]
            print(
                f"[{done:03d}/{n_seqs:03d}] {seq_name:<32s} "
                f"[ERROR] {err_line}  ({elapsed:.1f}s) w={wid}",
                flush=True)
            # Print full traceback for debugging
            print(f"  Full traceback:\n{tb}", flush=True)
            all_recs[seq_name] = dict(
                seq_name=seq_name, AO=-1., SS=-1., SR50=-1., SR75=-1.,
                PS=-1., NPS=-1., n_valid=0, fps=0., elapsed=elapsed)
            _append_csv_row(partial_csv_path, per_seq_fields, all_recs[seq_name])

    for proc in procs:
        proc.join(timeout=30)

    t_total = time.perf_counter() - t_all
    print(SEP)
    print(f'\n[INFO] 追踪完成，总耗时 {t_total / 60:.1f} 分钟', flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 汇总统计
    # ══════════════════════════════════════════════════════════════════════════
    recs  = list(all_recs.values())
    valid = [r for r in recs if r['AO'] >= 0]

    if valid:
        total_frames = sum(r['n_valid'] for r in valid)
        seq_means: Dict[str, float] = {}
        frm_means: Dict[str, float] = {}
        for k in _METRICS:
            seq_means[k] = float(np.mean([r[k] for r in valid]))
            frm_means[k] = (
                float(sum(r[k] * r['n_valid'] for r in valid) / total_frames)
                if total_frames > 0 else -1.)
        mfps = float(np.mean([r['fps'] for r in valid if r['fps'] > 0]))
    else:
        seq_means = {k: -1. for k in _METRICS}
        frm_means = {k: -1. for k in _METRICS}
        mfps = 0.
        total_frames = 0

    W = 9
    print(f'\n{"=" * 78}')
    print(f"[汇总] {len(valid)}/{len(recs)} 条有效序列  "
          f"总帧数={total_frames}  平均FPS={mfps:.1f}")
    print(f"{'':12}" + ''.join(f"{k:>{W}}" for k in _METRICS))
    print(f"{'序列均值(%)':<12}" +
          ''.join(f"{seq_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f"{'帧加权(%)' :<12}" +
          ''.join(f"{frm_means[k]*100:>{W}.2f}" for k in _METRICS))
    print(f'{"=" * 78}')
    print(f"[RESULT] " +
          " ".join(f"{k}={seq_means[k]:.4f}" for k in _METRICS), flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # per_seq_metrics.csv
    # ══════════════════════════════════════════════════════════════════════════
    csv_path = os.path.join(result_dir, 'per_seq_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_seq_fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, '') for k in per_seq_fields})
    print(f'[INFO] per_seq_metrics.csv  → {csv_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # eval_history.csv（追加）
    # ══════════════════════════════════════════════════════════════════════════
    history_dir = os.path.join(args.save_dir, args.dataset)
    history_csv = os.path.join(history_dir, 'eval_history.csv')
    os.makedirs(history_dir, exist_ok=True)

    hist_fields = (
        ['ckpt_tag', 'checkpoint', 'dataset', 'n_valid', 'n_total', 'mean_fps'] +
        [f'seq_{k}' for k in _METRICS] +
        [f'frm_{k}' for k in _METRICS]
    )
    hist_row: dict = {
        'ckpt_tag':   save_name,
        'checkpoint': os.path.basename(args.checkpoint),
        'dataset':    args.dataset,
        'n_valid':    len(valid),
        'n_total':    len(recs),
        'mean_fps':   f'{mfps:.2f}',
    }
    for k in _METRICS:
        hist_row[f'seq_{k}'] = f'{seq_means[k]:.4f}'
        hist_row[f'frm_{k}'] = f'{frm_means[k]:.4f}'

    write_header = not os.path.isfile(history_csv)
    with open(history_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        if write_header:
            w.writeheader()
        w.writerow(hist_row)
    print(f'[INFO] eval_history.csv     → {history_csv}')

    # ══════════════════════════════════════════════════════════════════════════
    # summary.json
    # ══════════════════════════════════════════════════════════════════════════
    import json
    summary = dict(
        checkpoint=args.checkpoint, dataset=args.dataset,
        yaml_name=args.yaml_name, epoch=args.epoch,
        n_sequences=len(recs), n_valid=len(valid), mean_fps=mfps,
        seq_means={k: seq_means[k] for k in _METRICS},
        frm_means={k: frm_means[k] for k in _METRICS},
        total_time_min=f'{t_total / 60:.1f}',
    )
    summary_path = os.path.join(result_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[INFO] summary.json         → {summary_path}')


if __name__ == '__main__':
    main()
