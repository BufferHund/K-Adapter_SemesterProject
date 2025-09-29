import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'artifacts' / 'results_index.csv'
TABLE_DIR = ROOT / 'tables'
TABLE_DIR.mkdir(exist_ok=True)


def load_rows():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def fmt(x, digits=3):
    try:
        if x is None or x == '':
            return '-'
        v = float(x)
        return f'{v:.{digits}f}'
    except Exception:
        return str(x)


def write_table(path: Path, headers, rows):
    with path.open('w') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')


def exp1(rows):
    exp_dir = 'Group1/EXP1 verify_finetune'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    order = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    data = [next((x for x in r if x['setup'] == o), None) for o in order]
    data = [x for x in data if x]
    headers = ['Setup', 'Batch', 'LR', 'Warmup', 'Test F1-A', 'Test F1-B']
    out_rows = []
    for x in data:
        out_rows.append([
            x['setup'], x['batch'], x['lr'], x['warmup'], fmt(x['test_f1']), fmt(x['test_f12'])
        ])
    write_table(TABLE_DIR / 'exp1_verify_finetune.md', headers, [[str(c) for c in row] for row in out_rows])


def exp2(rows):
    exp_dir = 'Group1/EXP2 zeroshot_test'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    setups = ['baseline', 'lin_only', 'fac_only', 'fac_lin_add', 'fac_lin_concat']
    # generic zeroshot
    rz = [x for x in r if x['dataset_tag'] == 'zeroshot']
    data_z = [next((x for x in rz if x['setup'] == s), None) for s in setups]
    data_z = [x for x in data_z if x]
    headers_z = ['Setup', 'Test F1-A']
    rows_z = [[x['setup'], fmt(x['test_f1'])] for x in data_z]
    write_table(TABLE_DIR / 'exp2_zeroshot_generic.md', headers_z, [[str(c) for c in row] for row in rows_z])

    # FIGER zeroshot
    rf = [x for x in r if x['dataset_tag'] == 'figer_zeroshot']
    data_f = [next((x for x in rf if x['setup'] == s), None) for s in setups]
    data_f = [x for x in data_f if x]
    headers_f = ['Setup', 'FIGER Macro F1', 'FIGER Micro F1']
    rows_f = [[x['setup'], fmt(x['test_f1']), fmt(x['test_f12'])] for x in data_f]
    write_table(TABLE_DIR / 'exp2_zeroshot_figer.md', headers_f, [[str(c) for c in row] for row in rows_f])


def exp3(rows):
    exp_dir = 'Group2/EXP3 Size'
    r = [x for x in rows if x['experiment_dir'] == exp_dir and x['adapter_size']]
    r.sort(key=lambda x: int(x['adapter_size']))
    headers = ['Adapter Size', 'Test F1-A']
    out_rows = [[x['adapter_size'], fmt(x['test_f1'])] for x in r]
    write_table(TABLE_DIR / 'exp3_size.md', headers, [[str(c) for c in row] for row in out_rows])


def exp4(rows):
    exp_dir = 'Group2/EXP4 Position'
    r = [x for x in rows if x['experiment_dir'] == exp_dir and x['adapter_pos']]
    # Clean pos
    for x in r:
        x['adapter_pos'] = x['adapter_pos'].strip('_')
    r.sort(key=lambda x: x['adapter_pos'])
    headers = ['Positions', 'Test F1-A']
    out_rows = [[x['adapter_pos'], fmt(x['test_f1'])] for x in r]
    write_table(TABLE_DIR / 'exp4_position.md', headers, [[str(c) for c in row] for row in out_rows])


def exp5(rows):
    exp_dir = 'Group2/EXP5 Layers'
    r = [x for x in rows if x['experiment_dir'] == exp_dir and x['adapter_layers']]
    r.sort(key=lambda x: int(x['adapter_layers']))
    headers = ['Adapter Layers', 'Test F1-A']
    out_rows = [[x['adapter_layers'], fmt(x['test_f1'])] for x in r]
    write_table(TABLE_DIR / 'exp5_layers.md', headers, [[str(c) for c in row] for row in out_rows])


def exp6(rows):
    exp_dir = 'Group3/Exp6 Domainshift'
    r = [x for x in rows if x['experiment_dir'] == exp_dir]
    if not r:
        return
    x = r[0]
    headers = ['Setup', 'Test F1-A']
    out_rows = [[x['setup'], fmt(x['test_f1'])]]
    write_table(TABLE_DIR / 'exp6_domainshift.md', headers, [[str(c) for c in row] for row in out_rows])


def main():
    rows = load_rows()
    exp1(rows)
    exp2(rows)
    exp3(rows)
    exp4(rows)
    exp5(rows)
    exp6(rows)
    print(f'Tables saved to {TABLE_DIR}')


if __name__ == '__main__':
    main()

