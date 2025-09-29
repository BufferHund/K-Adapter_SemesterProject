#!/usr/bin/env python3
"""
Experiment 4 (Inference Efficiency) — Parse ABLATION_STUDY_GUIDE.md and plot:
  - Latency vs Batch Size (Base, +Fac, +Fac+Lin)
  - Throughput vs Batch Size
  - Peak GPU Memory vs Batch Size

Outputs:
  - figures/exp4_inference_efficiency.csv
  - figures/exp4_infer_latency.png
  - figures/exp4_infer_throughput.png
  - figures/exp4_infer_memory.png
"""
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GUIDE_PATH = os.path.join(REPO_ROOT, 'ABLATION_STUDY_GUIDE.md')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_table(lines: List[str], start_idx: int) -> Tuple[List[int], List[float], List[float], List[float], int]:
    """
    Parse a markdown table starting at start_idx (header line). Returns
    (batch_sizes, latency_ms, throughput, peak_mem_mb, next_index).
    Table is expected in the format:
      | Batch Size | Average Latency (ms) | Throughput (samples/sec) | Peak GPU Memory (MB) |
      | ... divider ...
      | 1 | 26.676 | 37.49 | 1392.22 |
    """
    idx = start_idx
    # Skip header and divider (2 lines)
    idx += 2
    batches: List[int] = []
    lat: List[float] = []
    thr: List[float] = []
    mem: List[float] = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith('|'):
            break
        parts = [p.strip() for p in line.strip('|').split('|')]
        if len(parts) < 4:
            break
        try:
            bsz = int(parts[0])
            latency = float(parts[1])
            throughput = float(parts[2])
            memory = float(parts[3])
        except ValueError:
            break
        batches.append(bsz)
        lat.append(latency)
        thr.append(throughput)
        mem.append(memory)
        idx += 1
    return batches, lat, thr, mem, idx


def parse_guide(path: str) -> Dict[str, Dict[str, List[float]]]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data: Dict[str, Dict[str, List[float]]] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('**1. 基线: 原始 RoBERTa-large**'):
            # Next table header should start within a few lines
            while i < len(lines) and 'Batch Size' not in lines[i]:
                i += 1
            b, l, t, m, i = parse_table(lines, i)
            data['Base'] = {'batch': b, 'lat': l, 'thr': t, 'mem': m}
            continue
        if line.strip().startswith('**2. 实验组1: RoBERTa + Factual Adapter**'):
            while i < len(lines) and 'Batch Size' not in lines[i]:
                i += 1
            b, l, t, m, i = parse_table(lines, i)
            data['+Fac'] = {'batch': b, 'lat': l, 'thr': t, 'mem': m}
            continue
        if line.strip().startswith('**3. 实验组2: RoBERTa + Factual & Linguistic Adapter**'):
            while i < len(lines) and 'Batch Size' not in lines[i]:
                i += 1
            b, l, t, m, i = parse_table(lines, i)
            data['+Fac+Lin'] = {'batch': b, 'lat': l, 'thr': t, 'mem': m}
            continue
        i += 1
    return data


def save_csv(data: Dict[str, Dict[str, List[float]]], csv_path: str) -> None:
    lines = ['config,batch_size,latency_ms,throughput_sps,peak_mem_mb']
    for cfg in ['Base', '+Fac', '+Fac+Lin']:
        if cfg not in data:
            continue
        b, l, t, m = data[cfg]['batch'], data[cfg]['lat'], data[cfg]['thr'], data[cfg]['mem']
        for i in range(len(b)):
            lines.append(f"{cfg},{b[i]},{l[i]},{t[i]},{m[i]}")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def plot_lines(data: Dict[str, Dict[str, List[float]]], key: str, ylabel: str, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6,4))
    for label in ['Base', '+Fac', '+Fac+Lin']:
        if label not in data:
            continue
        x = data[label]['batch']
        y = data[label][key]
        ax.plot(x, y, marker='o', label=label)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)
    data = parse_guide(GUIDE_PATH)
    if not data:
        print('No inference data found in guide.')
        return
    csv_path = os.path.join(FIG_DIR, 'exp4_inference_efficiency.csv')
    save_csv(data, csv_path)
    plot_lines(data, 'lat', 'Latency (ms)', 'Experiment 4: Inference Latency vs Batch Size', os.path.join(FIG_DIR, 'exp4_infer_latency.png'))
    plot_lines(data, 'thr', 'Throughput (samples/sec)', 'Experiment 4: Throughput vs Batch Size', os.path.join(FIG_DIR, 'exp4_infer_throughput.png'))
    plot_lines(data, 'mem', 'Peak GPU Memory (MB)', 'Experiment 4: Peak Memory vs Batch Size', os.path.join(FIG_DIR, 'exp4_infer_memory.png'))
    print('Saved CSV and figures to', FIG_DIR)


if __name__ == '__main__':
    main()

