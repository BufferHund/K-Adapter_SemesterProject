#!/usr/bin/env python3
"""
Plot Experiment 2 (Adapter Position ablation) results for OpenEntity.

This script scans outputs for fine-tuned OpenEntity runs comparing positions:
  - Baseline (sparse): adapter_list = 0,11,22 (from size=64 run)
  - Early:   0,1,2
  - Middle:  10,11,12
  - Late:    21,22,23

It generates:
  1) F1 vs. position (Micro/Macro, test)
  2) Precision/Recall vs. position (Micro/Macro, test)
  3) CSV with dev/test metrics

Output:
  - figures/exp2_adapter_position.png
  - figures/exp2_adapter_position_precision_recall.png
  - figures/exp2_adapter_position.csv
"""
import ast
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs_light")
FIG_DIR = os.path.join(REPO_ROOT, "figures")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_metrics_from_tuple(tup: Tuple[Any, Any, Any]) -> Tuple[float, float, float, float, float, float]:
    micro = tup[1]
    macro = tup[2]
    return float(micro[0]), float(micro[1]), float(micro[2]), float(macro[0]), float(macro[1]), float(macro[2])


def parse_last_dev_test_metrics(result_file: str) -> Optional[Tuple[Tuple[float, float, float, float, float, float], Tuple[float, float, float, float, float, float]]]:
    last_line_with_dev = None
    last_line_with_test = None
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            if "dev:" in line and "'dev': (" in line:
                last_line_with_dev = line.strip()
            if "test:" in line and "'test': (" in line:
                last_line_with_test = line.strip()
    if not last_line_with_test:
        return None
    try:
        test_payload = last_line_with_test.split("test:", 1)[1]
        test_dict = ast.literal_eval(test_payload)
        test_tuple = test_dict.get('test')
        if not test_tuple or not isinstance(test_tuple, tuple) or len(test_tuple) < 3:
            return None
        test_metrics = _extract_metrics_from_tuple(test_tuple)
        dev_metrics = None
        if last_line_with_dev:
            dev_payload = last_line_with_dev.split("dev:", 1)[1]
            dev_dict = ast.literal_eval(dev_payload)
            dev_tuple = dev_dict.get('dev')
            if dev_tuple and isinstance(dev_tuple, tuple) and len(dev_tuple) >= 3:
                dev_metrics = _extract_metrics_from_tuple(dev_tuple)
        return (dev_metrics or test_metrics), test_metrics
    except Exception:
        return None


def find_latest_result_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
    return files_sorted[-1]


def collect_position_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping category -> { src, dev, test }
    Categories: Baseline, Early, Middle, Late
    """
    out: Dict[str, Dict[str, Any]] = {}
    # Baseline from size=64 runs (0,11,22)
    baseline_file = find_latest_result_file(os.path.join(OUTPUTS_ROOT, "openentity_finetune_size64", "**", "*_result.txt"))
    if baseline_file:
        parsed = parse_last_dev_test_metrics(baseline_file)
        if parsed is not None:
            dev_t, test_t = parsed
            out['Baseline'] = {
                'src': baseline_file,
                'dev': {
                    'micro_p': dev_t[0], 'micro_r': dev_t[1], 'micro_f1': dev_t[2],
                    'macro_p': dev_t[3], 'macro_r': dev_t[4], 'macro_f1': dev_t[5],
                },
                'test': {
                    'micro_p': test_t[0], 'micro_r': test_t[1], 'micro_f1': test_t[2],
                    'macro_p': test_t[3], 'macro_r': test_t[4], 'macro_f1': test_t[5],
                }
            }
    # Position-specific (size=64)
    base_pos_dir = os.path.join(OUTPUTS_ROOT, "openentity_finetune_position_size64")
    mapping = {
        'Early':  "finetune_pos_0_1_2",
        'Middle': "finetune_pos_10_11_12",
        'Late':   "finetune_pos_21_22_23",
    }
    for label, sub in mapping.items():
        pattern = os.path.join(base_pos_dir, sub, "**", "*_result.txt")
        f = find_latest_result_file(pattern)
        if not f:
            continue
        parsed = parse_last_dev_test_metrics(f)
        if parsed is None:
            continue
        dev_t, test_t = parsed
        out[label] = {
            'src': f,
            'dev': {
                'micro_p': dev_t[0], 'micro_r': dev_t[1], 'micro_f1': dev_t[2],
                'macro_p': dev_t[3], 'macro_r': dev_t[4], 'macro_f1': dev_t[5],
            },
            'test': {
                'micro_p': test_t[0], 'micro_r': test_t[1], 'micro_f1': test_t[2],
                'macro_p': test_t[3], 'macro_r': test_t[4], 'macro_f1': test_t[5],
            }
        }
    return out


def save_csv(metrics: Dict[str, Dict[str, Any]], csv_path: str) -> None:
    header = [
        "position","dev_micro_p","dev_micro_r","dev_micro_f1","dev_macro_p","dev_macro_r","dev_macro_f1",
        "test_micro_p","test_micro_r","test_micro_f1","test_macro_p","test_macro_r","test_macro_f1","source_file"
    ]
    lines = [",".join(header)]
    for pos in ["Baseline","Early","Middle","Late"]:
        if pos not in metrics:
            continue
        m = metrics[pos]
        dev = m['dev']
        test = m['test']
        row = [pos]
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{dev[key]:.6f}" if dev else "")
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{test[key]:.6f}" if test else "")
        row.append(m['src'] or "")
        lines.append(",".join(row))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_f1(metrics: Dict[str, Dict[str, Any]], fig_path: str) -> None:
    labels = [p for p in ["Baseline","Early","Middle","Late"] if p in metrics]
    micro = [metrics[p]['test']['micro_f1'] for p in labels]
    macro = [metrics[p]['test']['macro_f1'] for p in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([i - width/2 for i in x], micro, width=width, label='Micro F1')
    ax.bar([i + width/2 for i in x], macro, width=width, label='Macro F1')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('F1')
    ax.set_title('Experiment 2: Adapter Position (Test F1)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_pr(metrics: Dict[str, Dict[str, Any]], fig_path: str) -> None:
    labels = [p for p in ["Baseline","Early","Middle","Late"] if p in metrics]
    x = list(range(len(labels)))
    micro_p = [metrics[p]['test']['micro_p'] for p in labels]
    micro_r = [metrics[p]['test']['micro_r'] for p in labels]
    macro_p = [metrics[p]['test']['macro_p'] for p in labels]
    macro_r = [metrics[p]['test']['macro_r'] for p in labels]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, micro_p, marker='o', label='Micro Precision')
    ax.plot(x, micro_r, marker='o', label='Micro Recall')
    ax.plot(x, macro_p, marker='s', label='Macro Precision')
    ax.plot(x, macro_r, marker='s', label='Macro Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Experiment 2: Precision/Recall vs Position (Test)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)
    metrics = collect_position_metrics()
    csv_path = os.path.join(FIG_DIR, "exp2_adapter_position.csv")
    fig_f1_path = os.path.join(FIG_DIR, "exp2_adapter_position.png")
    fig_pr_path = os.path.join(FIG_DIR, "exp2_adapter_position_precision_recall.png")
    save_csv(metrics, csv_path)

    if any(k in metrics for k in ("Baseline","Early","Middle","Late")):
        if any(('test' in metrics[k] and metrics[k]['test']) for k in metrics):
            plot_f1(metrics, fig_f1_path)
            print(f"Saved figure: {fig_f1_path}")
            plot_pr(metrics, fig_pr_path)
            print(f"Saved figure: {fig_pr_path}")
        else:
            print("No test metrics found for position study. CSV saved:", csv_path)
    else:
        print("No position results found. CSV saved:", csv_path)

    # Print summary
    for k in ["Baseline","Early","Middle","Late"]:
        if k in metrics and metrics[k]['test']:
            t = metrics[k]['test']
            print(f"{k}: micro_f1={t['micro_f1']:.4f}, macro_f1={t['macro_f1']:.4f}  [{metrics[k]['src']}]")


if __name__ == "__main__":
    main()

