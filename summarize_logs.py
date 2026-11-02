"""
[ADDED] Tabulate several runs side by side from their STRUCTURED results files (<run>_results.json written by train.py).
Nothing is parsed from terminal logs any more.

    python summarize_logs.py "MELD/*_results.json"        # glob pattern (default: *_results.json)

Columns: validation weighted-/macro-F1 (use THESE to pick hyper-parameters), the test result of the run (the test set is
evaluated once per run), calibration scores of the uncalibrated model, mean learned span (--lim4 runs) and per-class test F1.
"""
import glob
import json
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else '*_results.json'
rows = []
for fn in sorted(glob.glob(pat)):
    r = json.load(open(fn))
    if r.get('schema') != 2:
        continue
    t, v = r['test'], r['valid']
    sp = r.get('span', {}).get('mean_span_active_branches')
    rows.append((fn.split('/')[-1][-34:], r['seed'], r['best_epoch'], v['weighted_f1'], v['macro_f1'], t['weighted_f1'], t['macro_f1'], t['accuracy'],
                 t['ece'], t['nll'], t['brier'], sp, ' '.join('%s=%.1f' % (c, d['f1']) for c, d in t['per_class'].items())))
print('%-34s %4s %5s %8s %8s %8s %8s %7s %6s %6s %6s %6s  %s' % ('run', 'seed', 'epoch', 'val_wF1', 'val_mF1', 'test_wF1', 'test_mF1', 'acc', 'ECE', 'NLL', 'Brier', 'span', 'per-class test F1'))
for x in rows:
    print('%-34s %4d %5d %8.2f %8.2f %8.2f %8.2f %7.2f %6.2f %6.3f %6.3f %6s  %s' % (*x[:11], '-' if x[11] is None else '%.1f' % x[11], x[12]))
