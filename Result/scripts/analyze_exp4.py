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
OUT_DIR = ROOT / 'figures' / 'exp4'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / 'tables' / 'exp4'
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
    # cast
    num_f = ['lr','epoch','dev_p','dev_r','dev_f1','test_p','test_r','test_f1','dev_p2','dev_r2','dev_f12','test_p2','test_r2','test_f12','dev_n','test_n']
    num_i = ['batch','warmup','adapter_layers','adapter_size']
    for r in rows:
        for k in num_f:
            r[k] = float(r[k]) if r.get(k) else None
        for k in num_i:
            r[k] = int(float(r[k])) if r.get(k) else None
        if r.get('adapter_pos'):
            r['adapter_pos'] = r['adapter_pos'].strip('_')
    return rows


def find_exp4_rows(rows):
    exp4 = []
    for r in rows:
        exp_dir = r.get('experiment_dir') or ''
        parts = exp_dir.split('/')
        if len(parts) >= 2 and parts[1].startswith('EXP4') and r.get('adapter_pos'):
            exp4.append(r)
    # sort by numeric positions if possible
    def pos_key(s):
        try:
            lab = (s.get('adapter_pos') or '').replace(',', '_')
            return tuple(int(x) for x in lab.split('_'))
        except Exception:
            return (s.get('adapter_pos') or '')
    exp4.sort(key=pos_key)
    return exp4


def make_baseline_row(all_rows):
    base_size, _base_f1 = find_exp3_baseline(all_rows)
    if base_size is None:
        return None
    # pick the EXP3 row with that size
    for r in all_rows:
        if (r.get('experiment_dir') or '').split('/')[:2][1].startswith('EXP3') and int(r.get('adapter_size') or 0) == int(base_size):
            # synthesize an EXP4-like row
            return {
                'adapter_pos': '0,11,22',
                'test_p': r.get('test_p'), 'test_r': r.get('test_r'), 'test_f1': r.get('test_f1'),
                'dev_p': r.get('dev_p'), 'dev_r': r.get('dev_r'), 'dev_f1': r.get('dev_f1'),
                'test_n': r.get('test_n'), 'dev_n': r.get('dev_n'),
            }
    return None


def augment_with_baseline(rows, all_rows):
    base_row = make_baseline_row(all_rows)
    if not base_row:
        return rows
    labels = [r.get('adapter_pos') for r in rows]
    if '0,11,22' not in labels and '0_11_22' not in labels:
        # place baseline first for clarity
        return [base_row] + rows
    return rows


def find_exp3_baseline(rows):
    """Use EXP3 adapter size 768 as the baseline if available; otherwise fallback to largest size."""
    exp3 = []
    for r in rows:
        exp_dir = r.get('experiment_dir') or ''
        parts = exp_dir.split('/')
        if len(parts) >= 2 and parts[1].startswith('EXP3') and r.get('adapter_size') is not None:
            exp3.append(r)
    if not exp3:
        return None, None
    # Prefer size 768 explicitly
    for r in exp3:
        if int(r.get('adapter_size') or 0) == 768:
            return r.get('adapter_size'), r.get('test_f1')
    # Fallback to the largest available size
    exp3.sort(key=lambda x: x.get('adapter_size') or 0, reverse=True)
    base = exp3[0]
    return base.get('adapter_size'), base.get('test_f1')


def autolabel(ax, rects, fmt='{:.3f}', dy=3):
    for rect in rects:
        h = rect.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, clip_on=False)


def plot_bars(rows):
    labels = [r['adapter_pos'] for r in rows]
    f1 = [r['test_f1'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(labels)))
    rects = ax.bar(x, f1, color='#4c78a8')
    vals = [v for v in f1 if v is not None]
    if vals:
        ax.set_ylim(0, max(vals) * 1.12 if max(vals) > 0 else 0.1)
    ax.set_title('EXP4: Adapter Positions — Test F1-A by position', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Positions (layer indices)')
    ax.set_ylabel('Test F1-A')
    autolabel(ax, rects, dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'positions_bars.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_delta_vs_baseline(rows, all_rows):
    if not rows:
        return
    base_size, base_f1 = find_exp3_baseline(all_rows)
    if base_f1 is None:
        # fallback to first position as baseline if EXP3 missing
        by_pos = {r['adapter_pos']: r for r in rows}
        baseline_key = '0_1_2' if '0_1_2' in by_pos else rows[0]['adapter_pos']
        base_f1 = by_pos[baseline_key].get('test_f1') or 0.0
        title = f'EXP4: ΔF1-A vs baseline position {baseline_key}'
    else:
        # Use dispersed strategy label instead of size: 0,11,22
        title = 'EXP4: ΔF1-A vs baseline 0,11,22'
    labels = [r['adapter_pos'] for r in rows]
    deltas = [((r.get('test_f1') or 0.0) - (base_f1 or 0.0)) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(labels)))
    rects = ax.bar(x, deltas, color='#59a14f')
    ax.axhline(0, color='gray', lw=1)
    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('Positions (layer indices)')
    ax.set_ylabel('ΔF1-A')
    autolabel(ax, rects, fmt='{:+.3f}', dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'delta_vs_baseline.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr_scatter(rows):
    import numpy as np
    labels = [r['adapter_pos'] for r in rows]
    p = [r.get('test_p') for r in rows]
    rcl = [r.get('test_r') for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, px, rx in zip(labels, p, rcl):
        if px is None or rx is None:
            continue
        ax.scatter(px, rx, s=45, label=lab)
        ax.annotate(lab, (px, rx), textcoords='offset points', xytext=(4, 2), fontsize=8)
    vals_p = [x for x in p if x is not None]
    vals_r = [x for x in rcl if x is not None]
    if vals_p and vals_r:
        ax.set_xlim(min(vals_p)-0.02, max(vals_p)+0.02)
        ax.set_ylim(min(vals_r)-0.02, max(vals_r)+0.02)
    def add_iso(ax, levels=(0.60, 0.65, 0.70)):
        r = np.linspace(0.4, 0.85, 200)
        for f1 in levels:
            p = f1 * r / (2*r - f1)
            mask = (2*r - f1) > 1e-6
            ax.plot(p[mask], r[mask], lw=0.6, ls='--', color='#cccccc')
            if any(mask):
                ax.text(p[mask][-1], r[mask][-1], f'F1={f1:.2f}', fontsize=7, color='#999999')
    add_iso(ax)
    ax.grid(True, alpha=0.25)
    ax.set_title('EXP4: Precision–Recall by position (iso-F1)', pad=12)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    fig.tight_layout()
    out = OUT_DIR / 'pr_scatter.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dumbbell_vs_baseline(rows, all_rows):
    # Horizontal dumbbell: baseline (EXP3 size 768) vs each position's Test F1-A
    base_size, base_f1 = find_exp3_baseline(all_rows)
    if base_f1 is None:
        return
    labels = [r['adapter_pos'] for r in rows]
    vals = [r.get('test_f1') for r in rows]
    # Sort by position labels to keep consistent order
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for yi, v in zip(y, vals):
        if v is None:
            continue
        x0, x1 = base_f1, v
        # line
        ax.plot([x0, x1], [yi, yi], color='#9ecae1', lw=3)
        # points
        ax.scatter([x0], [yi], color='#636363', zorder=3, s=30, label='_baseline')
        ax.scatter([x1], [yi], color='#1f77b4', zorder=3, s=30, label='_value')
        ax.annotate(f"{(x1 - x0):+0.3f}", (x1, yi), textcoords='offset points', xytext=(6, -2), fontsize=8)
    # Legend should label baseline as dispersed strategy "0,11,22"
    ax.axvline(base_f1, color='#636363', lw=1, ls='--', label='Baseline 0,11,22')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test F1-A')
    ax.set_title('EXP4: Positions — Dumbbell vs baseline 0,11,22', pad=12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    out = OUT_DIR / 'dumbbell_vs_exp3_baseline.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_percent_gain_sorted(rows, all_rows):
    # Percent gain over EXP3 baseline
    base_size, base_f1 = find_exp3_baseline(all_rows)
    if base_f1 is None or base_f1 == 0:
        return
    data = []
    for r in rows:
        f1 = r.get('test_f1') or 0.0
        data.append((r['adapter_pos'], 100.0 * (f1 - base_f1) / base_f1))
    data.sort(key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in data]
    gains = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    rects = ax.bar(x, gains, color=['#59a14f' if g >= 0 else '#e15759' for g in gains])
    ax.axhline(0, color='gray', lw=1)
    ax.set_title('EXP4: % Gain in Test F1-A vs 0,11,22', pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('% Gain')
    for r, g in zip(rects, gains):
        ax.annotate(f'{g:+.1f}%', (r.get_x()+r.get_width()/2, r.get_height()),
                    textcoords='offset points', xytext=(0, 6), ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / 'percent_gain_sorted.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dev_test_slope(rows):
    # Slopegraph dev->test for F1-A
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [0, 1]
    for r in rows:
        d = r.get('dev_f1')
        t = r.get('test_f1')
        if d is None or t is None:
            continue
        ax.plot(x, [d, t], marker='o', label=r['adapter_pos'])
        ax.annotate(r['adapter_pos'], (x[1]+0.02, t), fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Dev', 'Test'])
    ax.set_ylabel('F1-A')
    ax.set_title('EXP4: Dev → Test F1-A by position', pad=12)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    out = OUT_DIR / 'dev_test_slope.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)

def write_tables(rows, all_rows):
    def fmt(v, d=3):
        if v is None:
            return '-'
        try:
            return f'{float(v):.{d}f}'
        except Exception:
            return str(v)

    # Overview table
    headers = ['Positions', 'Test P', 'Test R', 'Test F1-A', 'Dev P', 'Dev R', 'Dev F1-A', 'Test N', 'Dev N']
    lines = []
    for r in rows:
        lines.append([
            r['adapter_pos'], fmt(r.get('test_p')), fmt(r.get('test_r')), fmt(r.get('test_f1')),
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

    # Delta table vs EXP3 baseline size
    if rows:
        base_size, base_f1 = find_exp3_baseline(all_rows)
        if base_f1 is not None:
            # Label as dispersed strategy "0,11,22"
            headers2 = ['ΔF1-A vs 0,11,22']
            lines2 = []
            for r in rows:
                dv = (r.get('test_f1') or 0.0) - (base_f1 or 0.0)
                lines2.append([f'{r["adapter_pos"]}: {dv:+.3f}'])
        else:
            # Fallback to first row baseline
            base = rows[0]
            headers2 = [f'ΔF1-A vs {base["adapter_pos"]}']
            lines2 = []
            for r in rows[1:]:
                dv = (r.get('test_f1') or 0.0) - (base.get('test_f1') or 0.0)
                lines2.append([f'{r["adapter_pos"]}: {dv:+.3f}'])
        out_md2 = TABLE_DIR / 'delta_vs_baseline.md'
        with out_md2.open('w') as f:
            f.write('| ' + ' | '.join(headers2) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(headers2)) + ' |\n')
            for row in lines2:
                f.write('| ' + ' | '.join(row) + ' |\n')


def main():
    ensure_index_csv()
    rows = load_rows()
    exp4 = find_exp4_rows(rows)
    if not exp4:
        print('No EXP4 records found')
        return
    exp4b = augment_with_baseline(exp4, rows)
    plot_bars(exp4b)
    plot_delta_vs_baseline(exp4b, rows)
    plot_percent_gain_sorted(exp4b, rows)
    plot_dumbbell_vs_baseline(exp4b, rows)
    plot_dev_test_slope(exp4b)
    plot_pr_scatter(exp4b)
    write_tables(exp4b, rows)
    print(f'EXP4 figures saved to {OUT_DIR}')
    print(f'EXP4 tables saved to {TABLE_DIR}')


if __name__ == '__main__':
    main()
