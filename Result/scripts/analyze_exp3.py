import csv
from pathlib import Path
import os

# Headless matplotlib and local cache dir
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
OUT_DIR = ROOT / 'figures' / 'exp3'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / 'tables' / 'exp3'
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_index_csv():
    if not CSV_PATH.exists():
        import subprocess, sys
        parser = ROOT / 'scripts' / 'parse_results.py'
        subprocess.check_call([sys.executable, str(parser)])


def load_rows():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # cast numeric
    num_f = ['lr','epoch','dev_p','dev_r','dev_f1','dev_p2','dev_r2','dev_f12','test_p','test_r','test_f1','test_p2','test_r2','test_f12','dev_n','test_n']
    num_i = ['batch','warmup','adapter_size','adapter_layers']
    for r in rows:
        for k in num_f:
            if k in r:
                r[k] = float(r[k]) if r.get(k) not in (None, '',) else None
        for k in num_i:
            if k in r:
                r[k] = int(float(r[k])) if r.get(k) not in (None, '',) else None
    return rows


def find_exp3_rows(rows):
    exp3 = []
    for r in rows:
        exp_dir = r.get('experiment_dir') or ''
        parts = exp_dir.split('/')
        if len(parts) >= 2 and parts[1].startswith('EXP3'):
            if r.get('adapter_size') is not None:
                exp3.append(r)
    # sort by size
    exp3.sort(key=lambda x: x.get('adapter_size') or 0)
    return exp3


def autolabel(ax, rects, fmt='{:.3f}', dy=3):
    for rect in rects:
        h = rect.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, clip_on=False)


def plot_size_line(rows):
    sizes = [r['adapter_size'] for r in rows]
    f1 = [r['test_f1'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, f1, marker='o', color='#1f77b4')
    for s, v in zip(sizes, f1):
        if v is not None:
            ax.annotate(f'{v:.3f}', (s, v), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=8)
    vals = [v for v in f1 if v is not None]
    if vals:
        ax.set_ylim(min(vals) - 0.01, max(vals) * 1.08)
    ax.set_title('EXP3: Adapter Size vs Test F1-A (higher is better)', pad=10)
    ax.set_xlabel('Adapter Size (hidden dim)')
    ax.set_ylabel('Test F1-A')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / 'size_vs_f1_line.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_size_bars(rows):
    sizes = [str(r['adapter_size']) for r in rows]
    f1 = [r['test_f1'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(sizes)))
    rects = ax.bar(x, f1, color='#4c78a8')
    vals = [v for v in f1 if v is not None]
    if vals:
        ax.set_ylim(0, max(vals) * 1.12)
    ax.set_title('EXP3: Test F1-A by Adapter Size', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=0)
    ax.set_xlabel('Adapter Size')
    ax.set_ylabel('Test F1-A')
    autolabel(ax, rects, dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'size_vs_f1_bars.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_delta_vs_smallest(rows):
    if not rows:
        return
    baseline = rows[0]  # smallest size first due to sorting
    base_f1 = baseline.get('test_f1') or 0.0
    labels = [str(r['adapter_size']) for r in rows[1:]]
    deltas = [(r.get('test_f1') or 0.0) - base_f1 for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(labels)))
    rects = ax.bar(x, deltas, color='#59a14f')
    ax.axhline(0, color='gray', lw=1)
    ax.set_title(f'EXP3: ΔF1-A vs Size {baseline["adapter_size"]}', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Adapter Size')
    ax.set_ylabel('ΔF1-A')
    autolabel(ax, rects, fmt='{:+.3f}', dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'delta_vs_smallest.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr_scatter(rows):
    # Scatter of Precision vs Recall by size with iso-F1 curves
    import numpy as np
    sizes = [r['adapter_size'] for r in rows]
    p = [r.get('test_p') for r in rows]
    rcl = [r.get('test_r') for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for sx, px, rx in zip(sizes, p, rcl):
        if px is None or rx is None:
            continue
        ax.scatter(px, rx, s=40, label=str(sx))
        ax.annotate(str(sx), (px, rx), textcoords='offset points', xytext=(4,2), fontsize=8)
    # dynamic limits
    vals_p = [x for x in p if x is not None]
    vals_r = [x for x in rcl if x is not None]
    if vals_p and vals_r:
        ax.set_xlim(min(vals_p)-0.02, max(vals_p)+0.02)
        ax.set_ylim(min(vals_r)-0.02, max(vals_r)+0.02)
    # iso-F1 curves around observed range
    def add_iso(ax, levels=(0.66, 0.68, 0.70)):
        r = np.linspace(0.45, 0.85, 200)
        for f1 in levels:
            p = f1 * r / (2*r - f1)
            mask = (2*r - f1) > 1e-6
            ax.plot(p[mask], r[mask], lw=0.6, ls='--', color='#cccccc')
            if any(mask):
                ax.text(p[mask][-1], r[mask][-1], f'F1={f1:.2f}', fontsize=7, color='#999999')
    add_iso(ax)
    ax.grid(True, alpha=0.25)
    ax.set_title('EXP3: Precision–Recall by Adapter Size (iso-F1)', pad=10)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    fig.tight_layout()
    out = OUT_DIR / 'pr_scatter.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_tables(rows):
    def fmt(v, d=3):
        if v is None:
            return '-'
        try:
            return f'{float(v):.{d}f}'
        except Exception:
            return str(v)

    # Overview table
    headers = ['Adapter Size', 'Test P', 'Test R', 'Test F1-A', 'Dev P', 'Dev R', 'Dev F1-A', 'Test N', 'Dev N']
    lines = []
    for r in rows:
        lines.append([
            str(r['adapter_size']), fmt(r.get('test_p')), fmt(r.get('test_r')), fmt(r.get('test_f1')),
            fmt(r.get('dev_p')), fmt(r.get('dev_r')), fmt(r.get('dev_f1')),
            str(int(r.get('test_n'))) if r.get('test_n') is not None else '-',
            str(int(r.get('dev_n'))) if r.get('dev_n') is not None else '-',
        ])
    out_md = TABLE_DIR / 'overview.md'
    with out_md.open('w') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in lines:
            f.write('| ' + ' | '.join(row) + ' |\n')

    # Delta table vs smallest
    if rows:
        base = rows[0]
        base_size = base['adapter_size']
        headers2 = [f'ΔF1-A vs {base_size}']
        lines2 = []
        for r in rows[1:]:
            dv = (r.get('test_f1') or 0.0) - (base.get('test_f1') or 0.0)
            lines2.append([f'{r["adapter_size"]}: {dv:+.3f}'])
        out_md2 = TABLE_DIR / 'delta_vs_smallest.md'
        with out_md2.open('w') as f:
            f.write('| ' + ' | '.join(headers2) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(headers2)) + ' |\n')
            for row in lines2:
                f.write('| ' + ' | '.join(row) + ' |\n')


def main():
    ensure_index_csv()
    rows = load_rows()
    exp3 = find_exp3_rows(rows)
    if not exp3:
        print('No EXP3 records found')
        return
    plot_size_line(exp3)
    plot_size_bars(exp3)
    plot_delta_vs_smallest(exp3)
    plot_pr_scatter(exp3)
    write_tables(exp3)
    print(f'EXP3 figures saved to {OUT_DIR}')
    print(f'EXP3 tables saved to {TABLE_DIR}')


if __name__ == '__main__':
    main()

