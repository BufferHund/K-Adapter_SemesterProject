import os
from pathlib import Path
import csv

# Headless + local cache
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = ROOT / 'figures' / 'exp6'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / 'tables' / 'exp6'
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Data provided in the prompt (latency ms, throughput samples/sec, peak GPU memory MB)
batch_sizes = [1, 8, 16, 32]
base = {
    'latency':    [26.676, 111.611, 223.866, 421.133],
    'throughput': [37.49,   71.68,   71.47,   75.99  ],
    'memory':     [1392.22, 1535.75, 1699.77, 2027.83],
}
fac = {
    'latency':    [36.058, 131.591, 259.807, 491.540],
    'throughput': [27.73,   60.79,   61.58,   65.10  ],
    'memory':     [1584.25, 1731.52, 1899.57, 2237.67],
}
fac_lin = {
    'latency':    [43.807, 152.145, 296.917, 563.723],
    'throughput': [22.83,   52.58,   53.89,   56.77  ],
    'memory':     [1775.50, 1926.15, 2098.33, 2444.43],
}


def save_table():
    headers = ['Model', 'Batch Size', 'Latency(ms)', 'Throughput(samples/sec)', 'Peak GPU Mem (MB)']
    rows = []
    for name, data in [('Base', base), ('+Fac', fac), ('+Fac+Lin', fac_lin)]:
        for bs, lat, thr, mem in zip(batch_sizes, data['latency'], data['throughput'], data['memory']):
            rows.append([name, bs, lat, thr, mem])
    out_md = TABLE_DIR / 'inference_overview.md'
    with out_md.open('w') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for r in rows:
            f.write('| ' + ' | '.join([str(c) for c in r]) + ' |\n')

    out_csv = TABLE_DIR / 'inference_overview.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def line_plot(ykey: str, title: str, ylabel: str, fname: str):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(batch_sizes, base[ykey], marker='o', label='Base', color='#4c78a8')
    ax.plot(batch_sizes, fac[ykey], marker='o', label='+Fac', color='#59a14f')
    ax.plot(batch_sizes, fac_lin[ykey], marker='o', label='+Fac+Lin', color='#e15759')
    ax.set_title(title, pad=10)
    ax.set_xlabel('Batch Size')
    ax.set_xticks(batch_sizes)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)


def summary_bars():
    # Bar comparison at batch size 1 and 32
    import numpy as np
    idx1 = batch_sizes.index(1)
    idx32 = batch_sizes.index(32)
    configs = ['Base', '+Fac', '+Fac+Lin']
    data_sets = {
        'Latency (ms)': [base['latency'][idx1], fac['latency'][idx1], fac_lin['latency'][idx1]],
        'Throughput (samples/sec)': [base['throughput'][idx1], fac['throughput'][idx1], fac_lin['throughput'][idx1]],
        'Peak GPU Mem (MB)': [base['memory'][idx1], fac['memory'][idx1], fac_lin['memory'][idx1]],
    }
    data_sets32 = {
        'Latency (ms)': [base['latency'][idx32], fac['latency'][idx32], fac_lin['latency'][idx32]],
        'Throughput (samples/sec)': [base['throughput'][idx32], fac['throughput'][idx32], fac_lin['throughput'][idx32]],
        'Peak GPU Mem (MB)': [base['memory'][idx32], fac['memory'][idx32], fac_lin['memory'][idx32]],
    }

    def plot_one(data, title_suffix, fname):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        colors = ['#4c78a8', '#59a14f', '#e15759']
        for ax, (metric, vals) in zip(axes, data.items()):
            x = np.arange(len(configs))
            rects = ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=0)
            ax.set_title(metric)
            for r, v in zip(rects, vals):
                ax.annotate(f'{v:.2f}', (r.get_x()+r.get_width()/2, v), textcoords='offset points', xytext=(0, 6), ha='center', va='bottom', fontsize=8)
        fig.suptitle(f'EXP6: Summary @ {title_suffix}', y=1.02)
        fig.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)

    plot_one(data_sets, 'Batch Size 1', 'summary_bs1.png')
    plot_one(data_sets32, 'Batch Size 32', 'summary_bs32.png')


def relative_overheads():
    # Compute relative overhead vs Base in %
    import numpy as np
    pct = lambda a, b: (a - b) / b * 100.0
    rows = []
    for i, bs in enumerate(batch_sizes):
        rows.append({
            'Batch Size': bs,
            'Latency +Fac %': pct(fac['latency'][i], base['latency'][i]),
            'Latency +Fac+Lin %': pct(fac_lin['latency'][i], base['latency'][i]),
            'Throughput +Fac %': pct(fac['throughput'][i], base['throughput'][i]),
            'Throughput +Fac+Lin %': pct(fac_lin['throughput'][i], base['throughput'][i]),
            'Memory +Fac %': pct(fac['memory'][i], base['memory'][i]),
            'Memory +Fac+Lin %': pct(fac_lin['memory'][i], base['memory'][i]),
        })

    # Save table
    out_md = TABLE_DIR / 'relative_overheads.md'
    keys = ['Batch Size','Latency +Fac %','Latency +Fac+Lin %','Throughput +Fac %','Throughput +Fac+Lin %','Memory +Fac %','Memory +Fac+Lin %']
    with out_md.open('w') as f:
        f.write('| ' + ' | '.join(keys) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(keys)) + ' |\n')
        for r in rows:
            f.write('| ' + ' | '.join([f'{r[k]:.1f}' if isinstance(r[k], float) else str(r[k]) for k in keys]) + ' |\n')

    # Plot percent overheads (latency/memory positive is worse; throughput negative is worse)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = range(len(batch_sizes))
    # Latency
    axes[0].plot(batch_sizes, [r['Latency +Fac %'] for r in rows], marker='o', label='+Fac', color='#59a14f')
    axes[0].plot(batch_sizes, [r['Latency +Fac+Lin %'] for r in rows], marker='o', label='+Fac+Lin', color='#e15759')
    axes[0].axhline(0, color='gray', lw=1)
    axes[0].set_title('Latency overhead vs Base (%)')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('%')
    axes[0].legend()
    # Throughput
    axes[1].plot(batch_sizes, [r['Throughput +Fac %'] for r in rows], marker='o', label='+Fac', color='#59a14f')
    axes[1].plot(batch_sizes, [r['Throughput +Fac+Lin %'] for r in rows], marker='o', label='+Fac+Lin', color='#e15759')
    axes[1].axhline(0, color='gray', lw=1)
    axes[1].set_title('Throughput change vs Base (%)')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('%')
    axes[1].legend()
    # Memory
    axes[2].plot(batch_sizes, [r['Memory +Fac %'] for r in rows], marker='o', label='+Fac', color='#59a14f')
    axes[2].plot(batch_sizes, [r['Memory +Fac+Lin %'] for r in rows], marker='o', label='+Fac+Lin', color='#e15759')
    axes[2].axhline(0, color='gray', lw=1)
    axes[2].set_title('Peak GPU memory overhead vs Base (%)')
    axes[2].set_xlabel('Batch Size')
    axes[2].set_ylabel('%')
    axes[2].legend()
    fig.tight_layout()
    out = OUT_DIR / 'relative_overheads.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    save_table()
    line_plot('latency', 'EXP6: Latency vs Batch Size', 'Latency (ms)', 'latency_vs_bs.png')
    line_plot('throughput', 'EXP6: Throughput vs Batch Size', 'Throughput (samples/sec)', 'throughput_vs_bs.png')
    line_plot('memory', 'EXP6: Peak GPU Memory vs Batch Size', 'Peak GPU Memory (MB)', 'memory_vs_bs.png')
    summary_bars()
    relative_overheads()
    print(f'EXP6 figures saved to {OUT_DIR}')
    print(f'EXP6 tables saved to {TABLE_DIR}')


if __name__ == '__main__':
    main()

