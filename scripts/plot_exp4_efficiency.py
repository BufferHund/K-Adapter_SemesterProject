#!/usr/bin/env python3
"""
Experiment 4: Efficiency comparison (Adapter-Tuning vs Full Fine-tuning) on OpenEntity.

This script assembles a compact comparison of parameter efficiency:
  - Collect Adapter-Tuning results across sizes (16,64,256,768) from Exp1 outputs
  - Optionally collect Full Fine-tuning result if available
  - Estimate trainable parameter counts for adapters vs full model
  - Produce CSV + plots:
      1) F1 vs Trainable Params (log-x) scatter/line
      2) Method comparison bar (Micro-F1)

Outputs:
  - figures/exp4_efficiency.csv
  - figures/exp4_f1_vs_params.png
  - figures/exp4_method_bar.png
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


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
    return files_sorted[-1]


def estimate_adapter_params(adapter_size: int, hidden_size: int = 768, adapter_layers: int = 2, adapter_positions: int = 3) -> int:
    r = adapter_size
    # Down/Up projections (weights + biases)
    proj_params = hidden_size * r + r + r * hidden_size + hidden_size
    # Transformer layers approx params at width r
    per_layer = 12 * r * r
    trans_params = adapter_layers * per_layer
    per_position = proj_params + trans_params
    total = adapter_positions * per_position
    # Add a small head (~hidden->9 + bias) negligible vs above; skip
    return int(total)


def collect_adapter_points(sizes: List[int]) -> List[Dict[str, Any]]:
    points = []
    for size in sizes:
        # look for standard size runs
        file_path = find_latest(os.path.join(OUTPUTS_ROOT, f"openentity_finetune_size{size}", "**", "*_result.txt"))
        # fallback for size=16 concat run if needed
        if not file_path and size == 16:
            file_path = find_latest(os.path.join(OUTPUTS_ROOT, f"openentity_finetune_concat_size{size}", "**", "*_result.txt"))
        if not file_path:
            continue
        parsed = parse_last_dev_test_metrics(file_path)
        if not parsed:
            continue
        _, test_t = parsed
        params = estimate_adapter_params(size)
        points.append({
            'method': f'Adapter(size={size})',
            'adapter_size': size,
            'trainable_params_est': params,
            'test_micro_f1': test_t[2],
            'test_macro_f1': test_t[5],
            'src': file_path,
            'type': 'adapter'
        })
    return points


def collect_full_point() -> Optional[Dict[str, Any]]:
    file_path = find_latest(os.path.join(OUTPUTS_ROOT, "openentity_full_finetune", "**", "*_result.txt"))
    if not file_path:
        return None
    parsed = parse_last_dev_test_metrics(file_path)
    if not parsed:
        return None
    _, test_t = parsed
    # Rough parameter count for RoBERTa-large full fine-tuning (~355M)
    full_params = 355_000_000
    return {
        'method': 'Full FT',
        'adapter_size': None,
        'trainable_params_est': full_params,
        'test_micro_f1': test_t[2],
        'test_macro_f1': test_t[5],
        'src': file_path,
        'type': 'full'
    }


def save_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    header = [
        'method','adapter_size','trainable_params_est','test_micro_f1','test_macro_f1','source_file'
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            ",".join([
                r['method'],
                str(r['adapter_size']) if r['adapter_size'] is not None else "",
                str(r['trainable_params_est']),
                f"{r['test_micro_f1']:.6f}",
                f"{r['test_macro_f1']:.6f}",
                r['src']
            ])
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_f1_vs_params(rows: List[Dict[str, Any]], fig_path: str) -> None:
    # Separate adapter vs full for styling
    adapters = [r for r in rows if r['type'] == 'adapter']
    fulls = [r for r in rows if r['type'] == 'full']

    fig, ax = plt.subplots(figsize=(6,4))
    if adapters:
        xs = [r['trainable_params_est'] for r in adapters]
        ys = [r['test_micro_f1'] for r in adapters]
        labels = [str(r['adapter_size']) for r in adapters]
        ax.plot(xs, ys, marker='o', label='Adapters (size labels)')
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords='offset points', xytext=(4,4), fontsize=8)
    if fulls:
        xs = [r['trainable_params_est'] for r in fulls]
        ys = [r['test_micro_f1'] for r in fulls]
        ax.scatter(xs, ys, marker='^', color='red', label='Full FT')

    ax.set_xscale('log')
    ax.set_xlabel('Trainable Parameters (log scale)')
    ax.set_ylabel('Micro F1 (Test)')
    ax.set_title('Experiment 4: F1 vs Trainable Params')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def plot_method_bar(rows: List[Dict[str, Any]], fig_path: str) -> None:
    # Choose a representative adapter setting (e.g., size=16) + Full FT if present
    # Or plot all adapters + full if available
    labels = [r['method'] for r in rows]
    vals = [r['test_micro_f1'] for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x, vals)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Micro F1 (Test)')
    ax.set_title('Experiment 4: Method Comparison (Micro F1)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)
    sizes = [16, 64, 256, 768]
    rows = collect_adapter_points(sizes)
    full = collect_full_point()
    if full:
        rows.append(full)

    if not rows:
        print("No efficiency data found.")
        return

    csv_path = os.path.join(FIG_DIR, 'exp4_efficiency.csv')
    fig_f1_params_path = os.path.join(FIG_DIR, 'exp4_f1_vs_params.png')
    fig_method_bar_path = os.path.join(FIG_DIR, 'exp4_method_bar.png')

    save_csv(rows, csv_path)
    plot_f1_vs_params(rows, fig_f1_params_path)
    print(f"Saved figure: {fig_f1_params_path}")
    plot_method_bar(rows, fig_method_bar_path)
    print(f"Saved figure: {fig_method_bar_path}")

    # Print summary
    for r in rows:
        print(f"{r['method']}: params~{r['trainable_params_est']}, micro_f1={r['test_micro_f1']:.4f}, macro_f1={r['test_macro_f1']:.4f} [{r['src']}]")


if __name__ == '__main__':
    main()

