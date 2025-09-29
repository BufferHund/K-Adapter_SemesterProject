import re
import os
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = [ROOT / 'Group1', ROOT / 'Group2', ROOT / 'Group3']


def parse_filename(p: Path):
    name = p.name
    rel = p.relative_to(ROOT)
    d = {
        'file': str(p.relative_to(ROOT)),
        'group': rel.parts[0] if len(rel.parts) > 0 else None,
        'experiment_dir': str(p.parent.relative_to(ROOT)),
        'batch': None,
        'lr': None,
        'warmup': None,
        'epoch': None,
        'mode': None,  # eval / exp / finetune
        'dataset_tag': None,  # figer_zeroshot / zeroshot / None
        'setup': None,  # baseline / lin_only / fac_only / fac_lin_add / fac_lin_concat / with-tacred-adapter
        'adapter_size': None,
        'adapter_pos': None,
        'adapter_layers': None,
    }

    m = re.search(r'batch-(\d+)', name)
    if m:
        d['batch'] = int(m.group(1))
    m = re.search(r'lr-([0-9.eE-]+)', name)
    if m:
        d['lr'] = float(m.group(1))
    m = re.search(r'warmup-(\d+)', name)
    if m:
        d['warmup'] = int(m.group(1))
    m = re.search(r'epoch-([0-9.]+)', name)
    if m:
        try:
            d['epoch'] = float(m.group(1))
        except ValueError:
            d['epoch'] = None

    # mode
    for key in ['eval', 'exp', 'finetune']:
        if f'_{key}_' in name or name.startswith(key) or f'-{key}-' in name:
            d['mode'] = key
            break

    # dataset tag
    if 'figer_zeroshot' in name:
        d['dataset_tag'] = 'figer_zeroshot'
    elif 'zeroshot' in name:
        d['dataset_tag'] = 'zeroshot'

    # setup
    if 'baseline' in name:
        d['setup'] = 'baseline'
    elif 'lin_only' in name:
        d['setup'] = 'lin_only'
    elif 'fac_only' in name:
        d['setup'] = 'fac_only'
    elif 'fac_lin_add' in name:
        d['setup'] = 'fac_lin_add'
    elif 'fac_lin_concat' in name:
        d['setup'] = 'fac_lin_concat'
    elif 'with-tacred-adapter' in name:
        d['setup'] = 'with_tacred_adapter'

    # adapter size patterns (handle both 'adapter-size-16' and general 'size-16')
    m = re.search(r'adapter-size-(\d+)', name)
    if m:
        d['adapter_size'] = int(m.group(1))
    else:
        m = re.search(r'[^a-zA-Z]size-(\d+)', name)
        if m:
            d['adapter_size'] = int(m.group(1))
    m = re.search(r'pos-([0-9_]+)', name)
    if m:
        d['adapter_pos'] = m.group(1)
    m = re.search(r'layers-(\d+)', name)
    if m:
        d['adapter_layers'] = int(m.group(1))

    return d


def parse_metrics(txt: str):
    # Parse all lines starting with 'test:'; pick the candidate with best available test F1
    candidates = []
    for line in txt.strip().splitlines():
        if not line.strip().startswith('test:'):
            continue
        raw = line.strip()[len('test:'):]
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            try:
                raw_fixed = raw
                if raw_fixed.rfind('}') != -1:
                    raw_fixed = raw_fixed[: raw_fixed.rfind('}') + 1]
                obj = ast.literal_eval(raw_fixed)
            except Exception:
                continue
        if isinstance(obj, dict):
            candidates.append(obj)

    data = None
    # If we have multiple candidates, prefer the one with larger test F1
    def score_from_obj(obj):
        # tuple-based format
        if 'test' in obj and isinstance(obj['test'], tuple) and len(obj['test']) >= 2:
            t = obj['test']
            s = 0.0
            if isinstance(t[1], tuple) and len(t[1]) == 3 and isinstance(t[1][2], (int, float)):
                s = max(s, t[1][2])
            if len(t) >= 3 and isinstance(t[2], tuple) and len(t[2]) == 3 and isinstance(t[2][2], (int, float)):
                s = max(s, t[2][2])
            return s
        # FIGER-like nested dict
        if 'test' in obj and isinstance(obj['test'], dict):
            t = obj['test']
            s = 0.0
            if isinstance(t.get('ERINE_macro'), tuple) and len(t['ERINE_macro']) == 3:
                s = max(s, float(t['ERINE_macro'][2]))
            if isinstance(t.get('ERINE_micro'), tuple) and len(t['ERINE_micro']) == 3:
                s = max(s, float(t['ERINE_micro'][2]))
            return s
        return -1.0

    if candidates:
        data = max(candidates, key=score_from_obj)
    else:
        # Try full-file dict (e.g., FIGER format without 'test:' prefix)
        try:
            maybe = ast.literal_eval(txt.strip())
            if isinstance(maybe, dict):
                data = maybe
        except Exception:
            data = None

    def unpack(v):
        # v := (n_or_flag, (p, r, f1), (p2, r2, f12)) or sometimes missing second triple
        out = {
            'n': None,
            'p': None, 'r': None, 'f1': None,
            'p2': None, 'r2': None, 'f12': None,
        }
        if isinstance(v, tuple) and len(v) >= 2:
            out['n'] = v[0]
            if isinstance(v[1], tuple) and len(v[1]) == 3:
                out['p'], out['r'], out['f1'] = v[1]
            if len(v) >= 3 and isinstance(v[2], tuple) and len(v[2]) == 3:
                out['p2'], out['r2'], out['f12'] = v[2]
        return out

    res = {}
    # FIGER-like nested dict must be handled first if present
    if isinstance(data, dict) and 'test' in data and isinstance(data['test'], dict) and (
        'ERINE_macro' in data['test'] or 'ERINE_micro' in data['test']
    ):
        t = data['test']
        def unpack_figer(dct):
            out = {
                'n': None,
                'p': None, 'r': None, 'f1': None,
                'p2': None, 'r2': None, 'f12': None,
                'acc': None,
            }
            if 'ERINE_macro' in dct and isinstance(dct['ERINE_macro'], tuple) and len(dct['ERINE_macro']) == 3:
                out['p'], out['r'], out['f1'] = dct['ERINE_macro']
            if 'ERINE_micro' in dct and isinstance(dct['ERINE_micro'], tuple) and len(dct['ERINE_micro']) == 3:
                out['p2'], out['r2'], out['f12'] = dct['ERINE_micro']
            if 'ERINE_accuracy' in dct:
                out['acc'] = dct['ERINE_accuracy']
            return out
        res = {'test': unpack_figer(t)}
        if 'dev' in data and isinstance(data['dev'], dict):
            res['dev'] = unpack_figer(data['dev'])
        return res
    # Generic tuple-based format
    if isinstance(data, dict) and ('dev' in data or 'test' in data):
        for split in ['dev', 'test']:
            if split in data:
                res[split] = unpack(data[split])
        return res
    return {}


def main():
    rows = []
    for base in RESULT_DIRS:
        if not base.exists():
            continue
        for p in base.rglob('*.txt'):
            meta = parse_filename(p)
            txt = p.read_text(errors='ignore')
            m = parse_metrics(txt)
            row = {**meta}
            for split in ['dev', 'test']:
                sd = m.get(split, {})
                row[f'{split}_n'] = sd.get('n')
                row[f'{split}_p'] = sd.get('p')
                row[f'{split}_r'] = sd.get('r')
                row[f'{split}_f1'] = sd.get('f1')
                row[f'{split}_p2'] = sd.get('p2')
                row[f'{split}_r2'] = sd.get('r2')
                row[f'{split}_f12'] = sd.get('f12')
                row[f'{split}_acc'] = sd.get('acc')
            rows.append(row)

    out_dir = ROOT / 'artifacts'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'results_index.csv'
    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f'Wrote {out_csv} with {len(rows)} rows')


if __name__ == '__main__':
    main()
