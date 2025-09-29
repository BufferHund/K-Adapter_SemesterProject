import csv
from pathlib import Path
import os

# Ensure Matplotlib works headless and caches inside workspace
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
OUT_DIR = ROOT / 'figures' / 'exp1'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_index_csv():
    if not CSV_PATH.exists():
        # Fallback: run parser to build CSV
        import subprocess, sys
        parser = ROOT / 'scripts' / 'parse_results.py'
        subprocess.check_call([sys.executable, str(parser)])


def load_rows():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # cast numeric fields
    num_fields_f = ['lr', 'epoch', 'dev_p', 'dev_r', 'dev_f1', 'dev_p2', 'dev_r2', 'dev_f12',
                    'test_p', 'test_r', 'test_f1', 'test_p2', 'test_r2', 'test_f12']
    num_fields_i = ['batch', 'warmup', 'adapter_size', 'adapter_layers']
    for r in rows:
        for k in num_fields_f:
            if k in r and r[k]:
                r[k] = float(r[k])
            else:
                r[k] = None
        for k in num_fields_i:
            if k in r and r[k]:
                r[k] = int(float(r[k]))
            else:
                r[k] = None
    return rows


def find_exp1_rows(rows):
    exp1 = []
    for r in rows:
        exp_dir = r.get('experiment_dir') or ''
        parts = exp_dir.split('/')
        # Expect pattern: GroupX / EXP1 ...
        if len(parts) >= 2 and parts[1].startswith('EXP1'):
            exp1.append(r)
    return exp1


def autolabel(ax, rects, fmt='{:.3f}', dy=3):
    for rect in rects:
        h = rect.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, clip_on=False)


def plot_overview(exp1_rows):
    # Keep one record per setup in desired order
    order = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    data = {}
    for o in order:
        data[o] = next((x for x in exp1_rows if x.get('setup') == o), None)
    labels = [o for o in order if data[o]]
    f1a = [data[o]['test_f1'] for o in labels]
    f1b = [data[o]['test_f12'] for o in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(labels)))
    w = 0.35
    r1 = ax.bar([i - w/2 for i in x], f1a, width=w, label='F1-A')
    r2 = ax.bar([i + w/2 for i in x], f1b, width=w, label='F1-B')
    # Ensure enough headroom for labels not to hit the top spine
    y_vals = [v for v in (f1a + f1b) if v is not None]
    if y_vals:
        y_max = max(y_vals)
        ax.set_ylim(0, y_max * 1.12)
    ax.set_title('EXP1: Pretrained Adapter Combinations — Test F1 (A/B higher is better)', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('F1')
    # Place legend outside at bottom to avoid overlapping with title/bars
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    autolabel(ax, r1, dy=6)
    autolabel(ax, r2, dy=6)
    # Reserve bottom space for the legend
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    out = OUT_DIR / 'overview_combination.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_delta_vs_baseline(exp1_rows):
    # Compute deltas against baseline for both F1 metrics
    by_setup = {r.get('setup'): r for r in exp1_rows if r.get('setup')}
    base = by_setup.get('baseline')
    if not base or base.get('test_f1') is None:
        return
    targets = ['lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    labels = [t for t in targets if t in by_setup]
    d_f1a = [(by_setup[t]['test_f1'] or 0.0) - base['test_f1'] for t in labels]
    d_f1b = [(by_setup[t]['test_f12'] or 0.0) - (base.get('test_f12') or 0.0) for t in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(labels)))
    w = 0.35
    r1 = ax.bar([i - w/2 for i in x], d_f1a, width=w, label='ΔF1-A')
    r2 = ax.bar([i + w/2 for i in x], d_f1b, width=w, label='ΔF1-B')
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_title('EXP1: Gains over Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('ΔF1')
    ax.legend()
    autolabel(ax, r1, fmt='{:+.3f}')
    autolabel(ax, r2, fmt='{:+.3f}')
    fig.tight_layout()
    out = OUT_DIR / 'delta_vs_baseline.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr_scatter(exp1_rows):
    # Scatter of Precision vs Recall for both metrics; add iso-F1 curves
    setups = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    data = {s: next((x for x in exp1_rows if x.get('setup') == s), None) for s in setups}
    colors = {
        'baseline': '#7f7f7f',
        'lin_only': '#1f77b4',
        'fac_only': '#2ca02c',
        'fac_lin_add': '#ff7f0e',
        'fac_lin_concat': '#d62728',
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    def add_iso_f1(ax, levels=(0.60, 0.70, 0.75)):
        import numpy as np
        r = np.linspace(0.3, 0.9, 200)
        for f1 in levels:
            # f1 = 2PR/(P+R) => P = f1*R/(2R - f1)
            p = f1 * r / (2*r - f1)
            mask = (2*r - f1) > 1e-6
            ax.plot(p[mask], r[mask], lw=0.6, ls='--', color='#cccccc')
            if any(mask):
                ax.text(p[mask][-1], r[mask][-1], f'F1={f1:.2f}', fontsize=7, color='#999999')

    for setup, rec in data.items():
        if not rec:
            continue
        p, r = rec.get('test_p'), rec.get('test_r')
        p2, r2 = rec.get('test_p2'), rec.get('test_r2')
        if p is not None and r is not None:
            ax1.scatter(p, r, label=setup, color=colors.get(setup, None))
            ax1.annotate(setup, (p, r), textcoords='offset points', xytext=(4, 2), fontsize=8)
        if p2 is not None and r2 is not None:
            ax2.scatter(p2, r2, label=setup, color=colors.get(setup, None))
            ax2.annotate(setup, (p2, r2), textcoords='offset points', xytext=(4, 2), fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlim(0.45, 0.86)
        ax.set_ylim(0.45, 0.86)
        ax.grid(True, alpha=0.2)
        add_iso_f1(ax)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')

    ax1.set_title('EXP1: PR Scatter (Metric A)')
    ax2.set_title('EXP1: PR Scatter (Metric B)')
    fig.tight_layout()
    out = OUT_DIR / 'pr_scatter.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dev_test_slope(exp1_rows):
    # Slopegraph showing dev->test F1 movement for each setup
    setups = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    data = [next((x for x in exp1_rows if x.get('setup') == s), None) for s in setups]
    data = [x for x in data if x]
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = [0, 1]
    for idx, rec in enumerate(data):
        da, ta = rec.get('dev_f1'), rec.get('test_f1')
        db, tb = rec.get('dev_f12'), rec.get('test_f12')
        c = colors[idx % len(colors)]
        if da is not None and ta is not None:
            ax1.plot(x, [da, ta], marker='o', color=c)
            ax1.annotate(rec.get('setup'), (x[1]+0.02, ta), fontsize=8)
        if db is not None and tb is not None:
            ax2.plot(x, [db, tb], marker='o', color=c)
            ax2.annotate(rec.get('setup'), (x[1]+0.02, tb), fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(['Dev', 'Test'])
        ax.set_ylabel('F1')
        ax.grid(True, axis='y', alpha=0.2)

    ax1.set_title('EXP1: Dev → Test (Metric A)')
    ax2.set_title('EXP1: Dev → Test (Metric B)')
    fig.tight_layout()
    out = OUT_DIR / 'dev_test_slope.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_metric_gains_bar(exp1_rows):
    # Δ(P, R, F1) over baseline for both metrics
    by_setup = {r.get('setup'): r for r in exp1_rows if r.get('setup')}
    base = by_setup.get('baseline')
    if not base:
        return
    targets = ['lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    labels = [t for t in targets if t in by_setup]
    metricsA = ['test_p', 'test_r', 'test_f1']
    metricsB = ['test_p2', 'test_r2', 'test_f12']

    def collect_deltas(keys):
        rows = []
        for t in labels:
            deltas = []
            for k in keys:
                v = (by_setup[t].get(k) or 0.0) - (base.get(k) or 0.0)
                deltas.append(v)
            rows.append(deltas)
        return rows

    rowsA = collect_deltas(metricsA)
    rowsB = collect_deltas(metricsB)

    import numpy as np
    x = np.arange(len(labels))
    w = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for i, m in enumerate(metricsA):
        ax1.bar(x + (i-1)*w, [r[i] for r in rowsA], width=w, label=m.replace('test_', '').upper())
    ax1.axhline(0, color='gray', lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title('EXP1: Δ over Baseline by Metric (A)')
    ax1.set_ylabel('Δ')
    ax1.legend()

    for i, m in enumerate(metricsB):
        ax2.bar(x + (i-1)*w, [r[i] for r in rowsB], width=w, label=m.replace('test_', '').upper())
    ax2.axhline(0, color='gray', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_title('EXP1: Δ over Baseline by Metric (B)')
    ax2.legend()

    fig.tight_layout()
    out = OUT_DIR / 'delta_by_metric.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ensure_index_csv()
    rows = load_rows()
    exp1_rows = find_exp1_rows(rows)
    if not exp1_rows:
        print('No EXP1 records found')
        return
    plot_overview(exp1_rows)
    plot_delta_vs_baseline(exp1_rows)
    plot_pr_scatter(exp1_rows)
    plot_dev_test_slope(exp1_rows)
    plot_metric_gains_bar(exp1_rows)
    print(f'EXP1 figures saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
