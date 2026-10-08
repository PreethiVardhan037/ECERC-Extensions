#!/usr/bin/env python3
"""
[ADDED] One-command tuning + final evaluation of all four limitations.  Run from the repository root:

    python auto_tune.py MELD                 # full grid (about 1-1.5 h on a Kaggle T4, estimate)
    python auto_tune.py IEMOCAP --quick      # small grid
    python auto_tune.py MELD --seeds 0 1 2   # seeds for the final comparison

What it does
  1. Trains the baseline (authors' code, no flags) with the tuning seed.
  2. Grid-searches each fix on its own:  L1 (--cb_beta x --focal_gamma),  L2 (--ls_eps x --ls_neighbor_mass),
     L4 (--span_penalty).  The winner of each grid is chosen by a VALIDATION score only (--select: weighted-F1,
     macro-F1, or their mean = default) -- the test set is never used for any choice and test numbers are not
     even printed or tabulated during the grid search.
  3. Final comparison over several seeds: baseline, L1*, L2*, L4*, and ALL (L1*+L2*+L4*), where * = tuned.
  4. L3: runs calibrate.py (temperature scaling + MC-Dropout) on the tuning-seed checkpoint of every method.
  5. Writes tuning_results/<DATASET>/{summary.md, grid.csv, final_runs.csv, best_config.json, logs/}.

Nothing here guarantees that a fix beats the baseline; the table shows exactly what happened. If no setting
helps on validation, the table says so -- report that honestly.
"""
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLS = r'(neu|sur|fea|sad|joy|dis|ang|hap|exc|fru)'

CFG = {
    'MELD': dict(seed=0, minority=['fea', 'dis'],
                 l1_full=[(b, g) for b in (0.99, 0.995, 0.999, 0.9995) for g in (0.5, 1.0, 2.0)],
                 l1_quick=[(0.995, 1.0), (0.999, 1.0)],
                 l2_full=[(e, m) for e in (0.02, 0.05, 0.1) for m in (0.5, 0.9)], l2_quick=[(0.05, 0.7)],
                 l4_full=[0.002, 0.005, 0.02, 0.1], l4_quick=[0.005, 0.02]),
    'IEMOCAP': dict(seed=2007, minority=['hap'],
                    l1_full=[(b, g) for b in (0.999, 0.9999) for g in (0.0, 1.0, 2.0)],
                    l1_quick=[(0.999, 1.0), (0.999, 2.0)],
                    l2_full=[(e, m) for e in (0.02, 0.05, 0.1) for m in (0.5, 0.9)], l2_quick=[(0.05, 0.7)],
                    l4_full=[0.002, 0.005, 0.02, 0.1], l4_quick=[0.005, 0.02]),
}


def parse_log(text):
    ev = re.search(r'Eval-metric: F1, Epoch: (\d+), best_eval_fscore: ([\d.]+), Accuracy: ([\d.]+), F1-Score: ([\d.]+)', text)
    if not ev:
        return None
    blk = text.split('Per-class test report')[-1]
    cls = {k: float(v) * 100 for k, v in re.findall(r'^\s+' + CLS + r'\s+[\d.]+\s+[\d.]+\s+([\d.]+)', blk, re.M)}
    macro = re.findall(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', blk)
    sp = re.search(r'Learned spans at best-validation epoch.*mean ([\d.]+)', text)
    br = re.search(r'cross_event=([\d.]+), contagion=([\d.]+), cross_emotion=([\d.]+)', text.split('Learned spans at best-validation epoch')[-1]) if sp else None
    vm = re.search(r'Validation scores at best-validation epoch: weighted-F1 [\d.]+ \| macro-F1 ([\d.]+)', text)
    return {'epoch': int(ev.group(1)), 'valid_wf1': float(ev.group(2)), 'test_acc': float(ev.group(3)),
            'test_wf1': float(ev.group(4)), 'macro_f1': float(macro[-1]) * 100 if macro else float('nan'),
            'valid_macro': float(vm.group(1)) if vm else None,
            'per_class': cls, 'mean_span': float(sp.group(1)) if sp else None,
            'branch_spans': [float(x) for x in br.groups()] if br else None}


class Runner:
    def __init__(self, ds, args):
        self.ds, self.args = ds, args
        self.out = os.path.join(HERE, 'tuning_results', ds)
        for sub in ('logs', 'ckpts', 'calib'):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        self.cache = {}

    def train(self, name, flags, seed):
        key = (name, seed)
        if key in self.cache:
            return self.cache[key]
        stem = '{}_s{}'.format(name, seed)
        log = os.path.join(self.out, 'logs', stem + '.txt')
        ck = os.path.join(self.out, 'ckpts', stem + '.pkl')
        res = None
        if self.args.resume and os.path.exists(log) and os.path.exists(ck):
            res = parse_log(open(log).read())
        if res is None:
            cmd = [sys.executable, 'train.py', '--seed', str(seed), '--save_model_path', ck] + flags + shlex.split(self.args.extra)
            print('  >> {}  {}'.format(stem, ' '.join(flags) or '(baseline)'), flush=True)
            p = subprocess.run(cmd, cwd=os.path.join(HERE, self.ds), capture_output=True, text=True)
            open(log, 'w').write(p.stdout + '\n' + p.stderr)
            res = parse_log(p.stdout)
            if res is None:
                raise RuntimeError('training failed for {} -- last output:\n{}'.format(stem, (p.stdout + p.stderr)[-2000:]))
        res.update(name=name, seed=seed, flags=' '.join(flags), ckpt=ck)
        self.cache[key] = res
        print('     validation: wF1 {:.2f} | macro-F1 {}{}'.format(
            res['valid_wf1'], '{:.2f}'.format(res['valid_macro']) if res['valid_macro'] is not None else 'n/a',
            ' | mean span {:.1f}'.format(res['mean_span']) if res['mean_span'] is not None else ''), flush=True)
        return res

    def calibrate(self, res):
        stem = '{}_s{}'.format(res['name'], res['seed'])
        out_dir = os.path.join(self.out, 'calib', stem)
        cmd = [sys.executable, 'calibrate.py', '--load_model_state_dir', res['ckpt'], '--out_dir', out_dir] + shlex.split(self.args.calib_args)
        p = subprocess.run(cmd, cwd=os.path.join(HERE, self.ds), capture_output=True, text=True)
        js = os.path.join(out_dir, 'calibration_{}.json'.format(self.ds))
        if not os.path.exists(js):
            print('     calibration failed for {}:\n{}'.format(stem, (p.stdout + p.stderr)[-800:]))
            return None
        d = json.load(open(js))
        t = d['test']
        return {'T': d['T'], **{k: t[k]['ece'] for k in ('Uncalibrated', 'Temp scaling', 'MC-Dropout', 'MC-Dropout + TS')}}


def mean_std(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float('nan'), float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['MELD', 'IEMOCAP'])
    ap.add_argument('--quick', action='store_true', help='much smaller grids')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help="final-comparison seeds (default: tuning seed and the next two)")
    ap.add_argument('--tune_seed', type=int, default=None, help="seed used for the grid search (default = the authors' seed)")
    ap.add_argument('--extra', type=str, default='', help='extra arguments passed to every train.py, e.g. "--epochs 20"')
    ap.add_argument('--calib_args', type=str, default='', help='extra arguments for calibrate.py, e.g. "--mc_passes 10"')
    ap.add_argument('--select', choices=['wf1', 'macro', 'mean'], default='mean',
                    help="validation score used to choose settings: weighted-F1, macro-F1, or their mean (default; macro-F1 matters "
                         "for the class-imbalance fix, but the MELD validation set has few Fear/Disgust utterances so it is noisy)")
    ap.add_argument('--skip_calibration', action='store_true')
    ap.add_argument('--resume', action='store_true', help='reuse finished runs found in tuning_results/ (e.g. after a Kaggle disconnect)')
    ap.add_argument('--keep_ckpts', action='store_true', help='do not delete checkpoints at the end')
    args = ap.parse_args()

    ds, cfg = args.dataset, CFG[args.dataset]
    tune_seed = cfg['seed'] if args.tune_seed is None else args.tune_seed
    seeds = args.seeds or [tune_seed, tune_seed + 1, tune_seed + 2]
    mode = 'quick' if args.quick else 'full'
    R = Runner(ds, args)
    grid_rows, notes = [], []

    def score(r):
        if args.select == 'wf1' or r['valid_macro'] is None:
            return r['valid_wf1']
        if args.select == 'macro':
            return r['valid_macro']
        return 0.5 * (r['valid_wf1'] + r['valid_macro'])

    print('=== [1/4] baseline (authors\' code), tuning seed {} ==='.format(tune_seed))
    base = R.train('baseline', [], tune_seed)

    def search(label, cands, to_flags, to_name):
        print('=== grid: {} ({} settings, chosen by VALIDATION score "{}") ==='.format(label, len(cands), args.select))
        best = None
        for c in cands:
            r = R.train(to_name(c), to_flags(c), tune_seed)
            grid_rows.append({'limitation': label, 'setting': to_name(c), 'valid_wf1': r['valid_wf1'], 'valid_macro': r['valid_macro'],
                              'score': score(r), 'mean_span': r['mean_span']})     # validation only: no test numbers in the grid
            if best is None or score(r) > score(best[1]):
                best = (c, r)
        print('   -> chosen for {}: {} (validation score {:.2f}; baseline {:.2f})'.format(
            label, to_name(best[0]), score(best[1]), score(base)))
        if score(best[1]) < score(base):
            notes.append('{}: no setting in the grid beat the baseline on the validation score "{}" ({:.2f} vs {:.2f}); the best one is shown anyway.'.format(
                label, args.select, score(best[1]), score(base)))
            print('   !! ' + notes[-1])
        return best

    print('=== [2/4] tuning each fix on its own ===')
    b1 = search('L1', cfg['l1_' + mode], lambda c: ['--lim1', '--cb_beta', str(c[0]), '--focal_gamma', str(c[1])], lambda c: 'L1_b{}_g{}'.format(*c))
    b2 = search('L2', cfg['l2_' + mode], lambda c: ['--lim2', '--ls_eps', str(c[0]), '--ls_neighbor_mass', str(c[1])], lambda c: 'L2_e{}_m{}'.format(*c))
    b4 = search('L4', cfg['l4_' + mode], lambda c: ['--lim4', '--span_penalty', str(c)], lambda c: 'L4_p{}'.format(c))
    F1 = b1[1]['flags'].split()
    F2 = b2[1]['flags'].split()
    F4 = b4[1]['flags'].split()
    method_flags = {'baseline': [], 'L1': F1, 'L2': F2, 'L4': F4, 'ALL (L1+L2+L4)': F1 + F2 + F4}
    best_cfg = {'dataset': ds, 'tune_seed': tune_seed, 'L1_ARGS': ' '.join(F1[1:]), 'L2_ARGS': ' '.join(F2[1:]), 'L4_ARGS': ' '.join(F4[1:])}
    json.dump(best_cfg, open(os.path.join(R.out, 'best_config.json'), 'w'), indent=2)

    print('=== [3/4] final comparison over seeds {} ==='.format(seeds))
    final = {m: [] for m in method_flags}
    reuse = {'baseline': base, 'L1': b1[1], 'L2': b2[1], 'L4': b4[1]}   # same flags + seed as the grid winners -> no need to retrain
    for s in seeds:
        for m, fl in method_flags.items():
            if s == tune_seed and m in reuse:
                final[m].append(reuse[m])
            else:
                final[m].append(R.train('final_' + m.split(' ')[0], fl, s))

    calib = {}
    if not args.skip_calibration:
        print('=== [4/4] L3 calibration (temperature scaling + MC-Dropout) on the seed-{} checkpoints ==='.format(tune_seed))
        for m in method_flags:
            r = [x for x in final[m] if x['seed'] == tune_seed][0]
            calib[m] = R.calibrate(r)

    # ------------------------------ report ------------------------------
    mi = cfg['minority']
    lines = ['# {} - tuned L1/L2/L4 vs baseline ({} grid)\n'.format(ds, mode),
             'Settings were chosen on VALIDATION scores only (score = {}, tuning seed {}). Final numbers: test set, mean ± std over seeds {}.\n'.format(args.select, tune_seed, seeds),
             'Chosen settings: `L1_ARGS="{}"`  `L2_ARGS="{}"`  `L4_ARGS="{}"`\n'.format(best_cfg['L1_ARGS'], best_cfg['L2_ARGS'], best_cfg['L4_ARGS']),
             *(['**Note:** ' + n + '\n' for n in notes]),
             '| method | weighted F1 | macro F1 | accuracy | {} | learned spans (utterances): cross-event / contagion / cross-emotion |'.format(' | '.join(c + ' F1' for c in mi)),
             '|---|---|---|---|{}---|'.format('---|' * len(mi))]
    bm = mean_std([x['test_wf1'] for x in final['baseline']])[0]
    bmac = mean_std([x['macro_f1'] for x in final['baseline']])[0]
    csv_rows = []
    for m, runs in final.items():
        w, wsd = mean_std([x['test_wf1'] for x in runs]); mc, msd = mean_std([x['macro_f1'] for x in runs])
        a, asd = mean_std([x['test_acc'] for x in runs])
        mins = [mean_std([x['per_class'].get(c) for x in runs]) for c in mi]
        brs = [x['branch_spans'] for x in runs if x.get('branch_spans')]
        sp = ' / '.join('{:.1f}'.format(np.mean([b[i] for b in brs])) for i in range(3)) if brs else '-'
        lines.append('| {} | {:.2f} ± {:.2f} ({:+.2f}) | {:.2f} ± {:.2f} ({:+.2f}) | {:.2f} ± {:.2f} | {} | {} |'.format(
            m, w, wsd, w - bm, mc, msd, mc - bmac, a, asd, ' | '.join('{:.1f} ± {:.1f}'.format(*x) for x in mins),
            sp))
        for x in runs:
            csv_rows.append({'method': m, 'seed': x['seed'], 'flags': x['flags'], 'valid_wf1': x['valid_wf1'], 'test_wf1': x['test_wf1'],
                             'macro_f1': x['macro_f1'], 'test_acc': x['test_acc'], **{c + '_f1': x['per_class'].get(c) for c in mi},
                             'mean_span': x['mean_span'], 'branch_spans(cross_event/contagion/cross_emotion)': x.get('branch_spans')})
    lines.append('\n(in brackets: difference to the baseline mean)\n')
    if calib:
        lines += ['## L3 calibration (test ECE %, tuning-seed checkpoints; lower is better)\n',
                  '| method | T | uncalibrated | temperature scaling | MC-Dropout | MC-Dropout + TS |', '|---|---|---|---|---|---|']
        for m, c in calib.items():
            if c:
                lines.append('| {} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(m, c['T'], c['Uncalibrated'], c['Temp scaling'], c['MC-Dropout'], c['MC-Dropout + TS']))
    vm = lambda r: '{:.2f}'.format(r['valid_macro']) if r['valid_macro'] is not None else 'n/a'
    lines += ['\n## Grid search (tuning seed {}; VALIDATION scores only; pick = highest "score" = {})\n'.format(tune_seed, args.select),
              '| limitation | setting | valid wF1 | valid macro-F1 | score | mean span |', '|---|---|---|---|---|---|']
    lines.append('| baseline | - | {:.2f} | {} | {:.2f} | - |'.format(base['valid_wf1'], vm(base), score(base)))
    for g in grid_rows:
        lines.append('| {} | {} | {:.2f} | {} | {:.2f} | {} |'.format(g['limitation'], g['setting'], g['valid_wf1'], vm(g), g['score'],
                                                                   '{:.1f}'.format(g['mean_span']) if g['mean_span'] is not None else '-'))
    open(os.path.join(R.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    for name, rows in (('grid.csv', grid_rows), ('final_runs.csv', csv_rows)):
        with open(os.path.join(R.out, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if not args.keep_ckpts:
        keep = {os.path.basename(x['ckpt']) for m in final for x in final[m] if x['seed'] == tune_seed}
        for f in os.listdir(os.path.join(R.out, 'ckpts')):
            if f not in keep:
                os.remove(os.path.join(R.out, 'ckpts', f))
    print('\n' + '\n'.join(lines))
    print('\nSaved to {}'.format(R.out))


if __name__ == '__main__':
    main()
