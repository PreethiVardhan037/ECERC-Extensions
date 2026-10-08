"""
[ADDED] Tabulate several train.py logs side by side (picks the validation-selected epoch's test result).

    python summarize_logs.py "MELD/log_*.txt"        # glob pattern (default: log_*.txt)

Columns: validation weighted-F1 at the chosen epoch (use THIS to pick hyper-parameters), test weighted-F1,
test macro-F1, mean learned span (--lim4 runs only) and the per-class test F1 of the chosen epoch.
"""
import glob
import re
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else 'log_*.txt'
CLASSES = r'(neu|sur|fea|sad|joy|dis|ang|hap|exc|fru)'
print('%-30s %5s %9s %8s %8s %9s  %s' % ('log', 'epoch', 'valid_wF1', 'test_wF1', 'macroF1', 'mean_span', 'per-class test F1'))
for f in sorted(glob.glob(pat)):
    t = open(f).read()
    ev = re.search(r'Eval-metric: F1, Epoch: (\d+), best_eval_fscore: ([\d.]+), Accuracy: ([\d.]+), F1-Score: ([\d.]+)', t)
    if not ev:
        continue
    blk = t.split('Per-class test report')[-1] if 'Per-class test report' in t else t
    cls = re.findall(r'^\s+' + CLASSES + r'\s+[\d.]+\s+[\d.]+\s+([\d.]+)', blk, re.M)
    macro = re.findall(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', blk)
    sp = re.search(r'Learned spans at best-validation epoch.*mean ([\d.]+)', t)
    print('%-30s %5s %9s %8s %8s %9s  %s' % (
        f[-30:], ev.group(1), ev.group(2), ev.group(4),
        '%.2f' % (float(macro[-1]) * 100) if macro else '-', sp.group(1) if sp else '-',
        ' '.join('%s=%.1f' % (k, float(v) * 100) for k, v in cls)))
