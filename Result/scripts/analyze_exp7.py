import csv
from pathlib import Path
import os

# Headless + local cache
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
OUT_DIR = ROOT / 'figures' / 'exp7'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / 'tables' / 'exp7'
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
    num_f = ['lr','epoch','dev_p','dev_r','dev_f1','dev_p2','dev_r2','dev_f12','test_p','test_r','test_f1','test_p2','test_r2','test_f12','dev_n','test_n']
    for r in rows:
        for k in num_f:
            r[k] = float(r[k]) if r.get(k) else None
    return rows


def find_exp7_row(rows):
    # Use Group3/Exp6 Domainshift as EXP7 (labeling only)
    cand = [r for r in rows if (r.get('experiment_dir') or '').endswith('Group3/Exp6 Domainshift')]
    if not cand:
        # fallback: any row under Group3
        cand = [r for r in rows if (r.get('experiment_dir') or '').startswith('Group3')]
    return cand[0] if cand else None


def autolabel(ax, rects, fmt='{:.3f}', dy=3):
    for rect in rects:
        h = rect.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, clip_on=False)


def plot_test_prf(row):
    labels = ['P', 'R', 'F1-A']
    vals = [row.get('test_p'), row.get('test_r'), row.get('test_f1')]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(labels))
    rects = ax.bar(x, vals, color=['#4c78a8', '#72b7b2', '#f28e2b'])
    vmax = max(v for v in vals if v is not None)
    ax.set_ylim(0, vmax * 1.12)
    ax.set_title('EXP7: Domain Shift — Test PRF (A)', pad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    autolabel(ax, rects, dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'test_prf_A.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_test_prf_B(row):
    labels = ['P', 'R', 'F1-B']
    vals = [row.get('test_p2'), row.get('test_r2'), row.get('test_f12')]
    if any(v is None for v in vals):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(labels))
    rects = ax.bar(x, vals, color=['#4c78a8', '#72b7b2', '#e15759'])
    vmax = max(v for v in vals if v is not None)
    ax.set_ylim(0, vmax * 1.12)
    ax.set_title('EXP7: Domain Shift — Test PRF (B)', pad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    autolabel(ax, rects, dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'test_prf_B.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dev_test_grouped(row):
    labels = ['P', 'R', 'F1']
    dev = [row.get('dev_p'), row.get('dev_r'), row.get('dev_f1')]
    test = [row.get('test_p'), row.get('test_r'), row.get('test_f1')]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = range(len(labels))
    w = 0.35
    r1 = ax.bar([i - w/2 for i in x], dev, width=w, label='Dev', color='#9ecae1')
    r2 = ax.bar([i + w/2 for i in x], test, width=w, label='Test', color='#3182bd')
    vmax = max([v for v in dev+test if v is not None])
    ax.set_ylim(0, vmax * 1.12)
    ax.set_title('EXP7: Domain Shift — Dev vs Test (PRF-A)', pad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    autolabel(ax, r1, dy=6)
    autolabel(ax, r2, dy=6)
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    out = OUT_DIR / 'dev_test_grouped_A.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr_scatter(row):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    dp, dr = row.get('dev_p'), row.get('dev_r')
    tp, tr = row.get('test_p'), row.get('test_r')
    if dp is not None and dr is not None:
        ax.scatter(dp, dr, s=45, label='Dev', color='#1f77b4')
        ax.annotate('Dev', (dp, dr), textcoords='offset points', xytext=(4, 2), fontsize=8)
    if tp is not None and tr is not None:
        ax.scatter(tp, tr, s=45, label='Test', color='#ff7f0e')
        ax.annotate('Test', (tp, tr), textcoords='offset points', xytext=(4, 2), fontsize=8)

    vals_p = [v for v in [dp, tp] if v is not None]
    vals_r = [v for v in [dr, tr] if v is not None]
    if vals_p and vals_r:
        ax.set_xlim(min(vals_p)-0.02, max(vals_p)+0.02)
        ax.set_ylim(min(vals_r)-0.02, max(vals_r)+0.02)

    def add_iso(ax, levels=(0.30, 0.35, 0.40)):
        r = np.linspace(0.2, 0.9, 200)
        for f1 in levels:
            p = f1 * r / (2*r - f1)
            mask = (2*r - f1) > 1e-6
            ax.plot(p[mask], r[mask], lw=0.6, ls='--', color='#cccccc')
            if any(mask):
                ax.text(p[mask][-1], r[mask][-1], f'F1={f1:.2f}', fontsize=7, color='#999999')
    add_iso(ax)
    ax.grid(True, alpha=0.25)
    ax.set_title('EXP7: Domain Shift — PR scatter with iso-F1 (A)', pad=10)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    fig.tight_layout()
    out = OUT_DIR / 'pr_scatter_A.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_table(row):
    def fmt(v, d=3):
        if v is None:
            return '-'
        try:
            return f'{float(v):.{d}f}'
        except Exception:
            return str(v)

    headers = ['Split', 'P', 'R', 'F1-A', 'P2', 'R2', 'F1-B', 'N']
    lines = [
        ['Dev', fmt(row.get('dev_p')), fmt(row.get('dev_r')), fmt(row.get('dev_f1')), fmt(row.get('dev_p2')), fmt(row.get('dev_r2')), fmt(row.get('dev_f12')), str(int(row.get('dev_n'))) if row.get('dev_n') is not None else '-'],
        ['Test', fmt(row.get('test_p')), fmt(row.get('test_r')), fmt(row.get('test_f1')), fmt(row.get('test_p2')), fmt(row.get('test_r2')), fmt(row.get('test_f12')), str(int(row.get('test_n'))) if row.get('test_n') is not None else '-'],
    ]

    out_md = TABLE_DIR / 'overview.md'
    with out_md.open('w') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in lines:
            f.write('| ' + ' | '.join(row) + ' |\n')


def main():
    ensure_index_csv()
    rows = load_rows()
    r = find_exp7_row(rows)
    if not r:
        print('No EXP7 record found')
        return
    plot_test_prf(r)
    plot_test_prf_B(r)
    plot_dev_test_grouped(r)
    plot_pr_scatter(r)
    write_table(r)
    print(f'EXP7 figures saved to {OUT_DIR}')
    print(f'EXP7 tables saved to {TABLE_DIR}')


if __name__ == '__main__':
    main()

