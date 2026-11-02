"""
[ADDED] Builds the ONE consolidated, machine-readable results report from the structured per-run files
(<run>_results.json written by train.py and calibration_<DS>.json written by calibrate.py). Nothing is parsed out of
terminal logs.  Used by auto_tune.py (and usable on its own):

    python report.py tuning_results/MELD/runs.json           # re-render the report from a saved runs file

Outputs (written next to the runs file):
    final_results.json   everything: per run and per seed, plus mean / std over seeds
    final_results.csv    one row per (method, seed): every scalar metric, per-class P/R/F1, spans, calibrated variants
    final_report.md      the human-readable tables (classification, per-class P/R/F1, confusion matrices,
                         calibration deterministic vs stochastic, per-pair calibration, attention spans)

Convention: percentages in %, NLL and Brier raw; "±" is the sample standard deviation (n-1) over seeds.
"""
import csv
import json
import os
import sys

import numpy as np

SCALARS = ['accuracy', 'weighted_f1', 'macro_f1', 'weighted_precision', 'weighted_recall', 'macro_precision', 'macro_recall', 'ece', 'nll', 'brier']
CAL_KEYS = ['acc', 'weighted_f1', 'macro_f1', 'ece', 'nll', 'brier', 'avg_conf']
DETERMINISTIC, STOCHASTIC = ['Uncalibrated', 'Temp scaling'], ['MC-Dropout', 'MC-Dropout + TS']


def ms(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return float('nan'), float('nan'), 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def f(m, s, d=2):
    return 'n/a' if not np.isfinite(m) else '{:.{d}f} ± {:.{d}f}'.format(m, s, d=d)


def make_record(method, run, calib):
    """run: the train.py results dict; calib: the calibrate.py json dict or None."""
    rec = {'method': method, 'seed': run['seed'], 'flags': ' '.join(run.get('flag_list', [])) or '(baseline)', 'ckpt': run['ckpt'],
           'best_epoch': run['best_epoch'], 'epochs_run': run['epochs_run'], 'protocol': run['protocol'],
           'valid': {k: run['valid'][k] for k in ('weighted_f1', 'macro_f1', 'accuracy', 'ece', 'nll', 'brier')},
           'test': {k: run['test'][k] for k in SCALARS + ['per_class', 'confusion_matrix', 'class_order', 'n']},
           'span': run.get('span')}
    if calib:
        rec['calibration'] = {'T': calib['T'], 'T_mc': calib['T_mc'], 'protocol': calib.get('protocol'),
                              'variants': {m: {k: calib['test_full'][m][{'acc': 'accuracy'}.get(k, k)] for k in CAL_KEYS} for m in DETERMINISTIC + STOCHASTIC},
                              'pairs': calib['test_pairs']}
    return rec


def aggregate(records):
    out = {'n_seeds': len(records), 'seeds': [r['seed'] for r in records]}
    out['test'] = {k: dict(zip(('mean', 'std', 'n'), ms([r['test'][k] for r in records]))) for k in SCALARS}
    out['valid'] = {k: dict(zip(('mean', 'std', 'n'), ms([r['valid'][k] for r in records]))) for k in ('weighted_f1', 'macro_f1', 'accuracy')}
    classes = records[0]['test']['class_order']
    out['per_class'] = {c: {m: dict(zip(('mean', 'std', 'n'), ms([r['test']['per_class'][c][m] for r in records]))) for m in ('precision', 'recall', 'f1')}
                        for c in classes}
    out['confusion_matrix_mean'] = np.mean([np.array(r['test']['confusion_matrix'], dtype=float) for r in records], axis=0).tolist()
    cal = [r for r in records if r.get('calibration')]
    if cal:
        out['calibration'] = {m: {k: dict(zip(('mean', 'std', 'n'), ms([r['calibration']['variants'][m][k] for r in cal]))) for k in CAL_KEYS}
                              for m in DETERMINISTIC + STOCHASTIC}
        out['calibration']['T'] = dict(zip(('mean', 'std', 'n'), ms([r['calibration']['T'] for r in cal])))
        pairs = {}
        for pair in cal[0]['calibration']['pairs']['Uncalibrated']:
            pairs[pair] = {}
            for m in DETERMINISTIC + STOCHASTIC:
                g = lambda path: [r['calibration']['pairs'][m][pair][path[0]][path[1]] for r in cal]
                pairs[pair][m] = {'ece_true_in_pair': dict(zip(('mean', 'std', 'n'), ms(g(('true_in_pair', 'ece'))))),
                                  'n_true_in_pair': float(np.mean(g(('true_in_pair', 'n')))),
                                  'ece_pred_in_pair': dict(zip(('mean', 'std', 'n'), ms(g(('pred_in_pair', 'ece'))))),
                                  'n_pred_in_pair': float(np.mean(g(('pred_in_pair', 'n')))),
                                  'confusion_rate_pct': dict(zip(('mean', 'std', 'n'), ms(g(('confusions', 'rate_of_true_in_pair'))))),
                                  'mean_conf_on_confusions': dict(zip(('mean', 'std', 'n'), ms(g(('confusions', 'mean_conf_on_confusions')))))}
        out['pairs'] = pairs
    sp = [r['span'] for r in records if r.get('span')]
    if sp:
        out['span'] = {'span_max': sp[0]['span_max'], 'max_dialogue_length': sp[0]['max_dialogue_length'], 'note': sp[0]['note'], 'branches': {}}
        for b in sp[0]['branches']:
            out['span']['branches'][b] = {'status': sp[0]['branches'][b]['status'],
                                          'mean_span': dict(zip(('mean', 'std', 'n'), ms([s['branches'][b]['mean'] for s in sp]))),
                                          'per_head_mean': np.mean([s['branches'][b]['per_head'] for s in sp], axis=0).tolist()}
        out['span']['mean_span_active'] = dict(zip(('mean', 'std', 'n'), ms([s['mean_span_active_branches'] for s in sp])))
    return out


def to_markdown(ds, results, meta):
    L = ['# {} - final results (test set evaluated once per run; "±" = sample std over seeds)\n'.format(ds)]
    st = meta.get('status') or {}
    if st and not st.get('complete', True):
        L += ['> **RESULTS INCOMPLETE - this is NOT a finished experiment.** {} required result(s) are missing '
              '({} of {} calibrations produced). Tables below only cover what exists.'.format(len(st.get('missing', [])), st.get('calibrations_produced'), st.get('expected_calibrations')), '>']
        for mm in st.get('missing', [])[:30]:
            L.append('> - {} (seed {}): {}'.format(mm['method'], mm['seed'], mm['what']))
        for fl in st.get('failures', [])[:30]:
            L.append('> - failure in `{}` of `{}` after {} attempt(s): {}'.format(fl['stage'], fl['run'], fl.get('attempts', 1), (fl.get('reason') or '')[:160]))
        L.append('')
    elif st:
        L.append('> Status: **COMPLETE** - all {} training runs and {} calibrations required by the protocol exist.\n'.format(st.get('trained_runs'), st.get('calibrations_produced')))
    for w in st.get('warnings', []):
        L.append('> Warning: ' + w + '\n')
    for line in meta.get('notes', []):
        L.append('> ' + line)
    L.append('\nProtocol: ' + meta.get('protocol', '') + '\n')
    methods = list(results)
    base = methods[0]
    L += ['## 1. Classification (test)\n', '| method | seeds | accuracy | weighted F1 | macro F1 | weighted P | weighted R | macro P | macro R |', '|---|---|---|---|---|---|---|---|---|']
    for m in methods:
        t = results[m]['test']
        row = [f(t[k]['mean'], t[k]['std']) for k in ('accuracy', 'weighted_f1', 'macro_f1', 'weighted_precision', 'weighted_recall', 'macro_precision', 'macro_recall')]
        L.append('| {} | {} | {} |'.format(m, results[m]['n_seeds'], ' | '.join(row)))
    L.append('\nDifference to the baseline (mean): ' + '; '.join(
        '{}: wF1 {:+.2f}, macro-F1 {:+.2f}'.format(m, results[m]['test']['weighted_f1']['mean'] - results[base]['test']['weighted_f1']['mean'],
                                                results[m]['test']['macro_f1']['mean'] - results[base]['test']['macro_f1']['mean']) for m in methods[1:]) + '\n')

    classes = list(results[base]['per_class'])
    for metric, title in (('f1', 'F1'), ('precision', 'precision'), ('recall', 'recall')):
        L += ['## 2{}. Per-class {} (test, %; mean ± std over seeds)\n'.format('abc'[('f1', 'precision', 'recall').index(metric)], title),
              '| method | ' + ' | '.join(classes) + ' |', '|---|' + '---|' * len(classes)]
        for m in methods:
            L.append('| {} | {} |'.format(m, ' | '.join(f(results[m]['per_class'][c][metric]['mean'], results[m]['per_class'][c][metric]['std'], 1) for c in classes)))
        L.append('')
    L.append('## 3. Confusion matrices (test; rows = true class, columns = predicted; mean count over seeds)\n')
    for m in methods:
        cm = np.array(results[m]['confusion_matrix_mean'])
        L += ['**{}**\n'.format(m), '```', '{:<6}'.format('') + ''.join('{:>8}'.format(c) for c in classes)]
        L += ['{:<6}'.format(classes[i]) + ''.join('{:>8.1f}'.format(v) for v in cm[i]) for i in range(len(classes))] + ['```', '']

    no_cal = [m for m in methods if 'calibration' not in results[m]]
    if no_cal:
        L += ['## 4-5. Calibration - NOT AVAILABLE for: ' + ', '.join(no_cal) + '\n',
              'The calibration step produced no (valid) result for these methods, so the calibration comparison is omitted rather than shown partially.\n']
    if not no_cal:
        L += ['## 4. Calibration (test; T fitted on validation; same checkpoints/seeds as section 1; lower is better for ECE / NLL / Brier)\n',
              'Accuracy and F1 are unchanged by temperature scaling; the training modifications and the calibration step are separated '
              'by the rows: compare methods within one calibration column to see the effect of the training change, and the two columns of a '
              'group to see the effect of calibration.\n']
        for title, group in (('4a. Deterministic (dropout off)', DETERMINISTIC), ('4b. Stochastic (MC-Dropout passes averaged)', STOCHASTIC)):
            L += ['### ' + title + '\n', '| method | calibration | ECE % | NLL | Brier | accuracy | weighted F1 |', '|---|---|---|---|---|---|---|']
            for m in methods:
                for v in group:
                    c = results[m]['calibration'][v]
                    L.append('| {} | {} | {} | {} | {} | {} | {} |'.format(m, v, f(c['ece']['mean'], c['ece']['std']), f(c['nll']['mean'], c['nll']['std'], 4),
                                                                          f(c['brier']['mean'], c['brier']['std'], 4), f(c['acc']['mean'], c['acc']['std']),
                                                                          f(c['weighted_f1']['mean'], c['weighted_f1']['std'])))
            L.append('')
        L.append('Fitted temperature T (mean over seeds): ' + ', '.join('{} {:.3f}'.format(m, results[m]['calibration']['T']['mean']) for m in methods) + '\n')
        L += ['## 5. Per-pair calibration (test; TEMPERATURE-SCALED and UNCALIBRATED; populations: *true-in-pair* = true label is one of the two classes, '
              '*pred-in-pair* = predicted label is one of the two; n = utterances; confusion rate = % of true-in-pair utterances predicted as the partner)\n']
        for pair in results[base]['pairs']:
            L += ['### {}\n'.format(pair), '| method | calibration | ECE true-in-pair (n) | ECE pred-in-pair (n) | confusion rate % | mean confidence on the confusions % |', '|---|---|---|---|---|---|']
            for m in methods:
                for v in ('Uncalibrated', 'Temp scaling'):
                    p = results[m]['pairs'][pair][v]
                    L.append('| {} | {} | {} ({:.0f}) | {} ({:.0f}) | {} | {} |'.format(
                        m, v, f(p['ece_true_in_pair']['mean'], p['ece_true_in_pair']['std']), p['n_true_in_pair'],
                        f(p['ece_pred_in_pair']['mean'], p['ece_pred_in_pair']['std']), p['n_pred_in_pair'],
                        f(p['confusion_rate_pct']['mean'], p['confusion_rate_pct']['std']), f(p['mean_conf_on_confusions']['mean'], p['mean_conf_on_confusions']['std'], 1)))
            L.append('')
    span_methods = [m for m in methods if 'span' in results[m]]
    if span_methods:
        L += ['## 6. Learned attention spans (utterances; best-validation checkpoint)\n']
        for m in span_methods:
            s = results[m]['span']
            act = [b for b, v in s['branches'].items() if v['status'].startswith('active')]
            ina = [b for b in s['branches'] if b not in act]
            L += ['**{}** - {} of {} branches have a learnable span (active: {}; inactive by construction: {}).'.format(m, len(act), len(s['branches']), ', '.join(act), ', '.join(ina) or 'none')]
            L += ['**{}** - span_max {:.0f} (= longest dialogue {}), so training starts from exactly full-history attention. Mean over the active branches: {} utterances.\n'.format(
                m, s['span_max'], s['max_dialogue_length'], f(s['mean_span_active']['mean'], s['mean_span_active']['std'], 1)),
                '| branch | status | mean span ± std | mean span per head (8 heads) |', '|---|---|---|---|']
            for b, v in s['branches'].items():
                L.append('| {} | {} | {} | {} |'.format(b, v['status'], f(v['mean_span']['mean'], v['mean_span']['std'], 1), ' '.join('{:.1f}'.format(x) for x in v['per_head_mean'])))
            L.append('')
        L.append('*Only the three history-attending branches have a meaningful span; the self-event branch attends to the target utterance only (distance 0) '
                 'and is reported as inactive. ' + results[span_methods[0]]['span']['note'][0].upper() + results[span_methods[0]]['span']['note'][1:] + '.*\n')
    return '\n'.join(L)


def to_csv(path, records):
    classes = records[0]['test']['class_order']
    rows = []
    for r in records:
        row = {'method': r['method'], 'seed': r['seed'], 'flags': r['flags'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
               **{'valid_' + k: r['valid'][k] for k in ('weighted_f1', 'macro_f1')},
               **{'test_' + k: r['test'][k] for k in SCALARS}}
        for c in classes:
            for m in ('precision', 'recall', 'f1'):
                row['{}_{}'.format(m, c)] = r['test']['per_class'][c][m]
        if r.get('calibration'):
            row['T'] = r['calibration']['T']
            for v, d in r['calibration']['variants'].items():
                for k in ('ece', 'nll', 'brier'):
                    row['{}_{}'.format(k, v.replace(' ', '_').replace('+', 'plus'))] = d[k]
        if r.get('span'):
            row['mean_span_active'] = r['span']['mean_span_active_branches']
            for b, v in r['span']['branches'].items():
                row['span_' + b] = v['mean']
        rows.append(row)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_final_report(ds, runs_by_method, out_dir, meta=None):
    """runs_by_method: {method: [(run_results_dict, calibration_dict_or_None), ...]} (one entry per seed)."""
    meta = meta or {}
    os.makedirs(out_dir, exist_ok=True)
    records = {m: [make_record(m, r, c) for r, c in lst] for m, lst in runs_by_method.items()}
    results = {m: aggregate(recs) for m, recs in records.items()}
    payload = {'dataset': ds, 'meta': meta, 'conventions': 'percentages in %, NLL/Brier raw, std = sample std (n-1) over seeds',
               'aggregate': results, 'runs': [r for recs in records.values() for r in recs]}
    with open(os.path.join(out_dir, 'final_results.json'), 'w') as fh:
        json.dump(payload, fh, indent=2)
    to_csv(os.path.join(out_dir, 'final_results.csv'), [r for recs in records.values() for r in recs])
    md = to_markdown(ds, results, meta)
    with open(os.path.join(out_dir, 'final_report.md'), 'w') as fh:
        fh.write(md + '\n')
    return md


if __name__ == '__main__':
    p = sys.argv[1]
    saved = json.load(open(p))
    print(build_final_report(saved['dataset'], {m: [(r['run'], r.get('calib')) for r in lst] for m, lst in saved['methods'].items()}, os.path.dirname(p), saved.get('meta')))
