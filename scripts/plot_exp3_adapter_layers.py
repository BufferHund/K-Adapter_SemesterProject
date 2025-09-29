#!/usr/bin/env python3
"""
Plot Experiment 3 (Adapter internal transformer layers) results for OpenEntity.

Collects metrics for adapter_transformer_layers in {1, 2, 4} with size=64, list=0,11,22:
  - layers=1,4: outputs_light/openentity_finetune_layers_size64/finetune_layers_{L}/**/*_result.txt
  - layers=2 (baseline): outputs_light/openentity_finetune_size64/**/_result.txt

Generates:
  - figures/exp3_adapter_layers.png (Test Micro/Macro F1 vs Layers)
  - figures/exp3_adapter_layers_precision_recall.png (Test Micro/Macro P/R vs Layers)
  - figures/exp3_adapter_layers.csv (dev/test micro/macro P/R/F1 + source)
"""
import ast
import glob
import os
from typing import Any, Dict, Optional, Tuple

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


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
    return files_sorted[-1]


def collect_layers_metrics() -> Dict[int, Dict[str, Optional[dict]]]:
    out: Dict[int, Dict[str, Optional[dict]]] = {}
    # layers=1
    f1 = find_latest(os.path.join(OUTPUTS_ROOT, "openentity_finetune_layers_size64", "finetune_layers_1", "**", "*_result.txt"))
    if f1:
        parsed = parse_last_dev_test_metrics(f1)
        if parsed:
            dev_t, test_t = parsed
            out[1] = {
                'src': f1,
                'dev': {
                    'micro_p': dev_t[0], 'micro_r': dev_t[1], 'micro_f1': dev_t[2],
                    'macro_p': dev_t[3], 'macro_r': dev_t[4], 'macro_f1': dev_t[5],
                },
                'test': {
                    'micro_p': test_t[0], 'micro_r': test_t[1], 'micro_f1': test_t[2],
                    'macro_p': test_t[3], 'macro_r': test_t[4], 'macro_f1': test_t[5],
                }
            }
    # layers=2 baseline
    f2 = find_latest(os.path.join(OUTPUTS_ROOT, "openentity_finetune_size64", "**", "*_result.txt"))
    if f2:
        parsed = parse_last_dev_test_metrics(f2)
        if parsed:
            dev_t, test_t = parsed
            out[2] = {
                'src': f2,
                'dev': {
                    'micro_p': dev_t[0], 'micro_r': dev_t[1], 'micro_f1': dev_t[2],
                    'macro_p': dev_t[3], 'macro_r': dev_t[4], 'macro_f1': dev_t[5],
                },
                'test': {
                    'micro_p': test_t[0], 'micro_r': test_t[1], 'micro_f1': test_t[2],
                    'macro_p': test_t[3], 'macro_r': test_t[4], 'macro_f1': test_t[5],
                }
            }
    # layers=4
    f4 = find_latest(os.path.join(OUTPUTS_ROOT, "openentity_finetune_layers_size64", "finetune_layers_4", "**", "*_result.txt"))
    if f4:
        parsed = parse_last_dev_test_metrics(f4)
        if parsed:
            dev_t, test_t = parsed
            out[4] = {
                'src': f4,
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


def save_csv(metrics: Dict[int, Dict[str, Optional[dict]]], csv_path: str) -> None:
    header = [
        "layers","dev_micro_p","dev_micro_r","dev_micro_f1","dev_macro_p","dev_macro_r","dev_macro_f1",
        "test_micro_p","test_micro_r","test_micro_f1","test_macro_p","test_macro_r","test_macro_f1","source_file"
    ]
    lines = [",".join(header)]
    for L in [1,2,4]:
        if L not in metrics:
            continue
        m = metrics[L]
        dev = m['dev']
        test = m['test']
        row = [str(L)]
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{dev[key]:.6f}" if dev else "")
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{test[key]:.6f}" if test else "")
        row.append(m['src'] or "")
        lines.append(",".join(row))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_f1(metrics: Dict[int, Dict[str, Optional[dict]]], fig_path: str) -> None:
    layers = [L for L in [1,2,4] if L in metrics]
    micro = [metrics[L]['test']['micro_f1'] for L in layers]
    macro = [metrics[L]['test']['macro_f1'] for L in layers]

    x = range(len(layers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([i - width/2 for i in x], micro, width=width, label='Micro F1')
    ax.bar([i + width/2 for i in x], macro, width=width, label='Macro F1')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('F1')
    ax.set_xlabel('Adapter Internal Layers')
    ax.set_title('Experiment 3: Adapter Layers (Test F1)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_pr(metrics: Dict[int, Dict[str, Optional[dict]]], fig_path: str) -> None:
    layers = [L for L in [1,2,4] if L in metrics]
    x = list(range(len(layers)))
    micro_p = [metrics[L]['test']['micro_p'] for L in layers]
    micro_r = [metrics[L]['test']['micro_r'] for L in layers]
    macro_p = [metrics[L]['test']['macro_p'] for L in layers]
    macro_r = [metrics[L]['test']['macro_r'] for L in layers]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, micro_p, marker='o', label='Micro Precision')
    ax.plot(x, micro_r, marker='o', label='Micro Recall')
    ax.plot(x, macro_p, marker='s', label='Macro Precision')
    ax.plot(x, macro_r, marker='s', label='Macro Recall')
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_xlabel('Adapter Internal Layers')
    ax.set_title('Experiment 3: Precision/Recall vs Layers (Test)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)
    metrics = collect_layers_metrics()
    csv_path = os.path.join(FIG_DIR, "exp3_adapter_layers.csv")
    fig_f1_path = os.path.join(FIG_DIR, "exp3_adapter_layers.png")
    fig_pr_path = os.path.join(FIG_DIR, "exp3_adapter_layers_precision_recall.png")
    save_csv(metrics, csv_path)

    if any(L in metrics for L in (1,2,4)):
        if any(('test' in metrics[L] and metrics[L]['test']) for L in metrics):
            plot_f1(metrics, fig_f1_path)
            print(f"Saved figure: {fig_f1_path}")
            plot_pr(metrics, fig_pr_path)
            print(f"Saved figure: {fig_pr_path}")
        else:
            print("No test metrics found for layers study. CSV saved:", csv_path)
    else:
        print("No layers results found. CSV saved:", csv_path)

    # Print summary
    for L in [1,2,4]:
        if L in metrics and metrics[L]['test']:
            t = metrics[L]['test']
            print(f"layers={L}: micro_f1={t['micro_f1']:.4f}, macro_f1={t['macro_f1']:.4f}  [{metrics[L]['src']}]")


if __name__ == "__main__":
    main()

