import csv
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
# Ensure matplotlib uses a writable config/cache dir inside workspace
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_rows():
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert floatish fields
    for r in rows:
        for k in list(r.keys()):
            if k in {'batch', 'warmup', 'adapter_size', 'adapter_layers'}:
                r[k] = int(r[k]) if r[k] else None
            elif k in {'lr', 'epoch', 'dev_p', 'dev_r', 'dev_f1', 'dev_p2', 'dev_r2', 'dev_f12', 'test_p', 'test_r', 'test_f1', 'test_p2', 'test_r2', 'test_f12', 'dev_acc', 'test_acc'}:
                r[k] = float(r[k]) if r[k] else None
    return rows


def autolabel(ax, rects, fmt='{:.3f}'):
    for rect in rects:
        height = rect.get_height()
        if height is None:
            continue
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def plot_exp1(rows):
    exp_dir = 'Group1/EXP1 verify_finetune'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    order = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    data = {}
    for o in order:
        for x in r:
            if x['setup'] == o:
                data[o] = x
                break
    labels = [o for o in order if o in data]
    f1a = [data[o]['test_f1'] for o in labels]
    f1b = [data[o]['test_f12'] for o in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    width = 0.35
    rects1 = ax.bar([i - width/2 for i in x], f1a, width, label='F1-A')
    rects2 = ax.bar([i + width/2 for i in x], f1b, width, label='F1-B')
    ax.set_title('EXP1: Finetune Verify (Test F1)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('F1')
    ax.legend()
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    out = FIG_DIR / 'exp1_verify_finetune.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_exp2(rows):
    exp_dir = 'Group1/EXP2 zeroshot_test'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    setups = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']

    # Zeroshot generic
    rz = [x for x in r if x['dataset_tag'] == 'zeroshot']
    data_z = {s: next((x for x in rz if x['setup'] == s), None) for s in setups}
    labels_z = [s for s, v in data_z.items() if v]
    f1z = [data_z[s]['test_f1'] for s in labels_z]

    # FIGER zeroshot (macro vs micro)
    rf = [x for x in r if x['dataset_tag'] == 'figer_zeroshot']
    data_f = {s: next((x for x in rf if x['setup'] == s), None) for s in setups}
    labels_f = [s for s, v in data_f.items() if v]
    f1_macro = [data_f[s]['test_f1'] for s in labels_f]
    f1_micro = [data_f[s]['test_f12'] for s in labels_f]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    x = range(len(labels_z))
    rects1 = ax1.bar(x, f1z, color='#4c78a8')
    ax1.set_title('EXP2: Zeroshot (Test F1-A)')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels_z, rotation=15)
    ax1.set_ylabel('F1')
    autolabel(ax1, rects1)

    x2 = range(len(labels_f))
    w = 0.35
    rects2 = ax2.bar([i - w/2 for i in x2], f1_macro, w, label='FIGER Macro F1', color='#59a14f')
    rects3 = ax2.bar([i + w/2 for i in x2], f1_micro, w, label='FIGER Micro F1', color='#e15759')
    ax2.set_title('EXP2: FIGER Zeroshot (Macro vs Micro)')
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(labels_f, rotation=15)
    ax2.legend()
    autolabel(ax2, rects2)
    autolabel(ax2, rects3)

    fig.tight_layout()
    out = FIG_DIR / 'exp2_zeroshot.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_exp3(rows):
    exp_dir = 'Group2/EXP3 Size'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    r = [x for x in r if x['adapter_size'] is not None]
    r.sort(key=lambda x: x['adapter_size'])
    sizes = [x['adapter_size'] for x in r]
    f1 = [x['test_f1'] for x in r]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, f1, marker='o')
    for sx, fy in zip(sizes, f1):
        ax.annotate(f'{fy:.3f}', (sx, fy), textcoords='offset points', xytext=(0,5), ha='center', fontsize=8)
    ax.set_title('EXP3: Adapter Size vs Test F1')
    ax.set_xlabel('Adapter Size')
    ax.set_ylabel('Test F1-A')
    ax.grid(True, alpha=0.2)
    out = FIG_DIR / 'exp3_size.png'
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_exp4(rows):
    exp_dir = 'Group2/EXP4 Position'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    r = [x for x in r if x['adapter_pos'] is not None]
    # Clean trailing underscores introduced by regex
    for x in r:
        if isinstance(x['adapter_pos'], str):
            x['adapter_pos'] = x['adapter_pos'].strip('_')
    r.sort(key=lambda x: x['adapter_pos'])
    labels = [x['adapter_pos'] for x in r]
    f1 = [x['test_f1'] for x in r]
    fig, ax = plt.subplots(figsize=(6, 4))
    rects = ax.bar(range(len(labels)), f1)
    ax.set_title('EXP4: Layer Positions vs Test F1')
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Test F1-A')
    autolabel(ax, rects)
    fig.tight_layout()
    out = FIG_DIR / 'exp4_position.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_exp5(rows):
    exp_dir = 'Group2/EXP5 Layers'
    r = [x for x in rows if x['experiment_dir'] == exp_dir and x['adapter_layers'] is not None]
    r.sort(key=lambda x: x['adapter_layers'])
    labels = [str(x['adapter_layers']) for x in r]
    f1 = [x['test_f1'] for x in r]
    fig, ax = plt.subplots(figsize=(5, 4))
    rects = ax.bar(range(len(labels)), f1)
    ax.set_title('EXP5: Adapter Layers vs Test F1')
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Adapter Layers')
    ax.set_ylabel('Test F1-A')
    autolabel(ax, rects)
    fig.tight_layout()
    out = FIG_DIR / 'exp5_layers.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_exp6(rows):
    exp_dir = 'Group3/Exp6 Domainshift'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    if not r:
        return
    x = r[0]
    fig, ax = plt.subplots(figsize=(4, 3))
    rects = ax.bar([0], [x['test_f1']], color='#4c78a8')
    ax.set_title('EXP6: Domain Shift (with TACRED Adapter)')
    ax.set_xticks([0])
    ax.set_xticklabels(['TACRED adapter'])
    ax.set_ylabel('Test F1-A')
    autolabel(ax, rects)
    fig.tight_layout()
    out = FIG_DIR / 'exp6_domainshift.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    rows = load_rows()
    plot_exp1(rows)
    plot_exp2(rows)
    plot_exp3(rows)
    plot_exp4(rows)
    plot_exp5(rows)
    plot_exp6(rows)
    print(f'Figures saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
