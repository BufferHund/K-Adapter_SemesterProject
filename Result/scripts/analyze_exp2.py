import csv
from pathlib import Path
import os

# Headless + cache inside workspace
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
OUT_DIR = ROOT / 'figures' / 'exp2'
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    num_f = ['lr','epoch','dev_p','dev_r','dev_f1','dev_p2','dev_r2','dev_f12','test_p','test_r','test_f1','test_p2','test_r2','test_f12']
    num_i = ['batch','warmup','adapter_size','adapter_layers']
    for r in rows:
        for k in num_f:
            r[k] = float(r[k]) if r.get(k) else None
        for k in num_i:
            r[k] = int(float(r[k])) if r.get(k) else None
    return rows


def find_exp2_rows(rows):
    exp2 = []
    for r in rows:
        exp_dir = r.get('experiment_dir') or ''
        parts = exp_dir.split('/')
        if len(parts) >= 2 and parts[1].startswith('EXP2'):
            exp2.append(r)
    return exp2


def autolabel(ax, rects, fmt='{:.3f}', dy=3):
    for rect in rects:
        h = rect.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, dy), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, clip_on=False)


SETUPS = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
COLORS = {
    'baseline': '#7f7f7f',
    'lin_only': '#1f77b4',
    'fac_only': '#2ca02c',
    'fac_lin_add': '#ff7f0e',
    'fac_lin_concat': '#d62728',
}


def overview_generic(rows):
    r = [x for x in rows if x.get('dataset_tag') == 'zeroshot']
    data = {s: next((x for x in r if x.get('setup') == s), None) for s in SETUPS}
    labels = [s for s, v in data.items() if v]
    f1 = [data[s]['test_f1'] for s in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    rects = ax.bar(x, f1, color=[COLORS.get(s) for s in labels])
    vals = [v for v in f1 if v is not None]
    if vals:
        ax.set_ylim(0, max(vals) * 1.12)
    ax.set_title('EXP2: Zeroshot (Generic) — Test F1-A (higher is better)', pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('F1-A')
    autolabel(ax, rects, dy=6)
    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    out = OUT_DIR / 'overview_generic.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def delta_vs_baseline_generic(rows):
    r = [x for x in rows if x.get('dataset_tag') == 'zeroshot']
    by_setup = {x.get('setup'): x for x in r if x.get('setup')}
    base = by_setup.get('baseline')
    if not base:
        return
    targets = [s for s in SETUPS if s in by_setup and s != 'baseline']
    d_f1 = [(by_setup[s]['test_f1'] or 0.0) - (base.get('test_f1') or 0.0) for s in targets]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(targets))
    rects = ax.bar(x, d_f1, color=[COLORS.get(s) for s in targets])
    ax.axhline(0, color='gray', lw=1)
    ax.set_title('EXP2: Zeroshot (Generic) — ΔF1-A vs Baseline', pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(targets, rotation=0)
    ax.set_ylabel('ΔF1-A')
    autolabel(ax, rects, fmt='{:+.3f}', dy=6)
    fig.tight_layout()
    out = OUT_DIR / 'delta_vs_baseline_generic.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figer_grouped_bars(rows):
    r = [x for x in rows if x.get('dataset_tag') == 'figer_zeroshot']
    data = {s: next((x for x in r if x.get('setup') == s), None) for s in SETUPS}
    labels = [s for s, v in data.items() if v]
    f1_macro = [data[s]['test_f1'] for s in labels]
    f1_micro = [data[s]['test_f12'] for s in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(labels)))
    w = 0.35
    rects1 = ax.bar([i - w/2 for i in x], f1_macro, width=w, label='FIGER Macro F1', color='#59a14f')
    rects2 = ax.bar([i + w/2 for i in x], f1_micro, width=w, label='FIGER Micro F1', color='#e15759')
    vals = [v for v in (f1_macro + f1_micro) if v is not None]
    if vals:
        ax.set_ylim(0, max(vals) * 1.12)
    ax.set_title('EXP2: FIGER Zeroshot — Macro vs Micro F1', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('F1')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    autolabel(ax, rects1, dy=6)
    autolabel(ax, rects2, dy=6)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    out = OUT_DIR / 'figer_macro_micro_grouped.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figer_pr_scatter(rows):
    import numpy as np
    r = [x for x in rows if x.get('dataset_tag') == 'figer_zeroshot']
    data = {s: next((x for x in r if x.get('setup') == s), None) for s in SETUPS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

    def add_iso_f1(ax, levels=(0.10, 0.15, 0.20, 0.25, 0.30)):
        r = np.linspace(0.3, 0.9, 200)
        for f1 in levels:
            p = f1 * r / (2*r - f1)
            mask = (2*r - f1) > 1e-6
            ax.plot(p[mask], r[mask], lw=0.6, ls='--', color='#cccccc')
            if any(mask):
                ax.text(p[mask][-1], r[mask][-1], f'F1={f1:.2f}', fontsize=7, color='#999999')

    for s, rec in data.items():
        if not rec:
            continue
        # Macro PR
        p, r1 = rec.get('test_p'), rec.get('test_r')
        if p is not None and r1 is not None:
            ax1.scatter(p, r1, label=s, color=COLORS.get(s, None))
            ax1.annotate(s, (p, r1), textcoords='offset points', xytext=(4, 2), fontsize=8)
        # Micro PR
        p2, r2 = rec.get('test_p2'), rec.get('test_r2')
        if p2 is not None and r2 is not None:
            ax2.scatter(p2, r2, label=s, color=COLORS.get(s, None))
            ax2.annotate(s, (p2, r2), textcoords='offset points', xytext=(4, 2), fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlim(0.01, 0.20)
        ax.set_ylim(0.45, 0.70)
        ax.grid(True, alpha=0.2)
        add_iso_f1(ax)
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')

    ax1.set_title('FIGER Macro — PR Scatter with iso-F1')
    ax2.set_title('FIGER Micro — PR Scatter with iso-F1')
    fig.tight_layout()
    out = OUT_DIR / 'figer_pr_scatter.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figer_delta_vs_baseline(rows):
    r = [x for x in rows if x.get('dataset_tag') == 'figer_zeroshot']
    by_setup = {x.get('setup'): x for x in r if x.get('setup')}
    base = by_setup.get('baseline')
    if not base:
        return
    targets = [s for s in SETUPS if s in by_setup and s != 'baseline']
    import numpy as np
    x = np.arange(len(targets))
    w = 0.35
    d_macro = [(by_setup[s]['test_f1'] or 0.0) - (base.get('test_f1') or 0.0) for s in targets]
    d_micro = [(by_setup[s]['test_f12'] or 0.0) - (base.get('test_f12') or 0.0) for s in targets]

    fig, ax = plt.subplots(figsize=(9, 4))
    r1 = ax.bar(x - w/2, d_macro, width=w, label='Δ Macro F1', color='#59a14f')
    r2 = ax.bar(x + w/2, d_micro, width=w, label='Δ Micro F1', color='#e15759')
    ax.axhline(0, color='gray', lw=1)
    ax.set_title('EXP2: FIGER Zeroshot — ΔF1 vs Baseline', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=0)
    ax.set_ylabel('ΔF1')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    autolabel(ax, r1, fmt='{:+.3f}', dy=6)
    autolabel(ax, r2, fmt='{:+.3f}', dy=6)
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    out = OUT_DIR / 'figer_delta_vs_baseline.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ensure_index_csv()
    rows = load_rows()
    exp2 = find_exp2_rows(rows)
    if not exp2:
        print('No EXP2 records found')
        return
    overview_generic(exp2)
    delta_vs_baseline_generic(exp2)
    figer_grouped_bars(exp2)
    figer_pr_scatter(exp2)
    figer_delta_vs_baseline(exp2)
    print(f'EXP2 figures saved to {OUT_DIR}')


if __name__ == '__main__':
    main()

