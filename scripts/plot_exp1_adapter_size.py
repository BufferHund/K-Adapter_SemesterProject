#!/usr/bin/env python3
"""
Plot Experiment 1 (Adapter Size ablation) results for OpenEntity.

This script scans the outputs_light directory for OpenEntity fine-tune runs
with different adapter sizes, extracts the final dev/test metrics from the
"*_result.txt" files, and generates figures:
  1) F1 vs. adapter size (Micro/Macro, test)
  2) Precision/Recall vs. adapter size (Micro/Macro, test)
  3) Micro-F1 vs. estimated adapter parameter count (test)
It also saves a CSV of the collected metrics (dev/test).

Expected directories (any that exist will be used):
  - outputs_light/openentity_finetune_size{16,64,256,768}/...
  - outputs_light/openentity_finetune_concat_size{16}/... (fallback for size=16)

Output:
  - figures/exp1_adapter_size.png
  - figures/exp1_adapter_size.csv
"""
import ast
import glob
import os
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')  # headless backend for servers/CI
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs_light")
FIG_DIR = os.path.join(REPO_ROOT, "figures")


def find_result_files_for_size(size: int) -> List[str]:
    candidates = []
    # Primary pattern
    primary_dir = os.path.join(OUTPUTS_ROOT, f"openentity_finetune_size{size}")
    candidates.extend(glob.glob(os.path.join(primary_dir, "**", "*_result.txt"), recursive=True))
    # Fallback pattern (e.g., concat runs)
    fallback_dir = os.path.join(OUTPUTS_ROOT, f"openentity_finetune_concat_size{size}")
    candidates.extend(glob.glob(os.path.join(fallback_dir, "**", "*_result.txt"), recursive=True))
    return candidates


def _extract_metrics_from_tuple(tup: Tuple[Any, Any, Any]) -> Tuple[float, float, float, float, float, float]:
    """
    From a tuple like (count, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1))
    return (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1)
    """
    micro = tup[1]
    macro = tup[2]
    return float(micro[0]), float(micro[1]), float(micro[2]), float(macro[0]), float(macro[1]), float(macro[2])


def parse_last_dev_test_metrics(result_file: str) -> Optional[Tuple[Tuple[float, float, float, float, float, float], Tuple[float, float, float, float, float, float]]]:
    """
    Parse the last 'dev' and last 'test' tuples from a result file produced by
    examples/run_finetune_openentity_adapter.py.

    Structure in file lines is like:
      test:{'dev': (cnt, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)), 'test': (...)}

    Returns:
      ((micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1),
       (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1)) if found; else None.
    """
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
    # Extract the dict payload after the first "test:" prefix
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


def collect_metrics(sizes: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Returns a mapping per size with keys:
      - 'src': source_file path or None
      - 'dev': dict with micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 (may be None)
      - 'test': dict with micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 (may be None)
    """
    metrics: Dict[int, Dict[str, Any]] = {}
    for size in sizes:
        files = find_result_files_for_size(size)
        src = None
        dev = None
        test = None
        if files:
            # Prefer the most recently modified result file
            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
            for candidate in reversed(files_sorted):
                parsed = parse_last_dev_test_metrics(candidate)
                if parsed is not None:
                    dev_tup, test_tup = parsed
                    dev = {
                        'micro_p': dev_tup[0], 'micro_r': dev_tup[1], 'micro_f1': dev_tup[2],
                        'macro_p': dev_tup[3], 'macro_r': dev_tup[4], 'macro_f1': dev_tup[5],
                    }
                    test = {
                        'micro_p': test_tup[0], 'micro_r': test_tup[1], 'micro_f1': test_tup[2],
                        'macro_p': test_tup[3], 'macro_r': test_tup[4], 'macro_f1': test_tup[5],
                    }
                    src = candidate
                    break
        metrics[size] = {'src': src, 'dev': dev, 'test': test}
    return metrics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(metrics: Dict[int, Dict[str, Any]], csv_path: str) -> None:
    header = [
        "adapter_size",
        # dev
        "dev_micro_p","dev_micro_r","dev_micro_f1","dev_macro_p","dev_macro_r","dev_macro_f1",
        # test
        "test_micro_p","test_micro_r","test_micro_f1","test_macro_p","test_macro_r","test_macro_f1",
        "source_file",
    ]
    lines = [",".join(header)]
    for size in sorted(metrics.keys()):
        row = [str(size)]
        dev = metrics[size]['dev']
        test = metrics[size]['test']
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{dev[key]:.6f}" if dev else "")
        for key in ["micro_p","micro_r","micro_f1","macro_p","macro_r","macro_f1"]:
            row.append(f"{test[key]:.6f}" if test else "")
        row.append(metrics[size]['src'] or "")
        lines.append(",".join(row))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_f1(metrics: Dict[int, Dict[str, Any]], fig_path: str) -> None:
    sizes = sorted(metrics.keys())
    micro_vals = [(metrics[s]['test']['micro_f1'] if metrics[s]['test'] else None) for s in sizes]
    macro_vals = [(metrics[s]['test']['macro_f1'] if metrics[s]['test'] else None) for s in sizes]

    # Filter out None values for plotting; mark missing as gaps
    x = list(range(len(sizes)))
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, micro_vals, marker='o', label='Micro F1')
    ax.plot(x, macro_vals, marker='s', label='Macro F1')

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('Adapter Size')
    ax.set_ylabel('F1')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Experiment 1: Adapter Size Ablation (Test F1)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
def plot_pr(metrics: Dict[int, Dict[str, Any]], fig_path: str) -> None:
    sizes = sorted(metrics.keys())
    x = list(range(len(sizes)))

    micro_p = [(metrics[s]['test']['micro_p'] if metrics[s]['test'] else None) for s in sizes]
    micro_r = [(metrics[s]['test']['micro_r'] if metrics[s]['test'] else None) for s in sizes]
    macro_p = [(metrics[s]['test']['macro_p'] if metrics[s]['test'] else None) for s in sizes]
    macro_r = [(metrics[s]['test']['macro_r'] if metrics[s]['test'] else None) for s in sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, micro_p, marker='o', label='Micro Precision')
    ax.plot(x, micro_r, marker='o', label='Micro Recall')
    ax.plot(x, macro_p, marker='s', label='Macro Precision')
    ax.plot(x, macro_r, marker='s', label='Macro Recall')

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel('Adapter Size')
    ax.set_ylabel('Score')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Experiment 1: Precision/Recall vs Adapter Size (Test)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def estimate_adapter_params(adapter_size: int, hidden_size: int = 768, adapter_layers: int = 2, adapter_positions: int = 3) -> int:
    """
    Rough parameter count estimation for one factual adapter configuration used in this project.
    Components per position:
      - Down/up projection: hidden->r and r->hidden
      - Internal transformer with `adapter_layers` layers at width r (self-attn + FFN)
    Note: This is an approximation for comparison across sizes.
    """
    r = adapter_size
    # Down/Up projections
    proj_params = hidden_size * r + r + r * hidden_size + hidden_size  # weights + biases
    # Transformer layers (approx): per layer ~ 4*r*r (attn) + 8*r*r (ffn) = 12*r*r, biases small
    per_layer = 12 * r * r
    trans_params = adapter_layers * per_layer
    per_position = proj_params + trans_params
    total = adapter_positions * per_position
    return int(total)


def plot_f1_vs_params(metrics: Dict[int, Dict[str, Any]], fig_path: str) -> None:
    sizes = sorted(metrics.keys())
    xs = [estimate_adapter_params(s) for s in sizes]
    ys = [(metrics[s]['test']['micro_f1'] if metrics[s]['test'] else None) for s in sizes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker='o')
    for x, y, s in zip(xs, ys, sizes):
        if y is not None:
            ax.annotate(str(s), (x, y), textcoords="offset points", xytext=(4,4), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Estimated Adapter Parameters (log scale)')
    ax.set_ylabel('Micro F1 (Test)')
    ax.set_title('Experiment 1: Micro F1 vs Adapter Params')
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)
    target_sizes = [16, 64, 256, 768]
    metrics = collect_metrics(target_sizes)

    csv_path = os.path.join(FIG_DIR, "exp1_adapter_size.csv")
    fig_f1_path = os.path.join(FIG_DIR, "exp1_adapter_size.png")
    fig_pr_path = os.path.join(FIG_DIR, "exp1_adapter_size_precision_recall.png")
    fig_params_path = os.path.join(FIG_DIR, "exp1_adapter_size_param_efficiency.png")
    fig_delta_path = os.path.join(FIG_DIR, "exp1_adapter_size_delta.png")

    save_csv(metrics, csv_path)
    # Only plot if at least one size has test data
    if any((metrics[s]['test'] is not None) for s in target_sizes):
        plot_f1(metrics, fig_f1_path)
        print(f"Saved figure: {fig_f1_path}")
        plot_pr(metrics, fig_pr_path)
        print(f"Saved figure: {fig_pr_path}")
        plot_f1_vs_params(metrics, fig_params_path)
        print(f"Saved figure: {fig_params_path}")
        # Delta vs size (baseline = smallest size available)
        avail = [s for s in target_sizes if metrics[s]['test'] is not None]
        if avail:
            base = min(avail)
            base_micro = metrics[base]['test']['micro_f1']
            base_macro = metrics[base]['test']['macro_f1']
            sizes = sorted(metrics.keys())
            x = list(range(len(sizes)))
            micro_delta = [((metrics[s]['test']['micro_f1'] - base_micro) if metrics[s]['test'] else None) for s in sizes]
            macro_delta = [((metrics[s]['test']['macro_f1'] - base_macro) if metrics[s]['test'] else None) for s in sizes]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.axhline(0.0, color='gray', linewidth=1)
            ax.plot(x, micro_delta, marker='o', label='Δ Micro F1')
            ax.plot(x, macro_delta, marker='s', label='Δ Macro F1')
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in sizes])
            ax.set_xlabel('Adapter Size (baseline = %d)' % base)
            ax.set_ylabel('Δ F1 (absolute)')
            ax.set_title('Experiment 1: ΔF1 vs Adapter Size (Test)')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_delta_path, dpi=200)
            plt.close(fig)
            print(f"Saved figure: {fig_delta_path}")
    else:
        print("No test metrics found to plot. CSV saved for inspection:", csv_path)

    # Print a small summary to stdout
    print("Adapter Size Metrics (Test Micro/Macro F1):")
    for size in target_sizes:
        t = metrics[size]['test']
        src = metrics[size]['src']
        if t is None:
            print(f"  size={size}: MISSING (no result file found)")
        else:
            print(f"  size={size}: micro_f1={t['micro_f1']:.4f}, macro_f1={t['macro_f1']:.4f}  [{src}]")


if __name__ == "__main__":
    main()
