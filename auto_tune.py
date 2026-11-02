#!/usr/bin/env python3
"""
[ADDED] One-command tuning + final evaluation of all four limitations.  Run from the repository root:

    python auto_tune.py MELD                      # tune on validation, then the full final comparison
    python auto_tune.py IEMOCAP --quick           # small grids
    python auto_tune.py MELD --tune_seeds 0 1 2   # choose settings by the MEAN validation score over 3 seeds (more stable)
    python auto_tune.py MELD --final_only         # skip tuning: reuse tuning_results/MELD/best_config.json (or --L1_ARGS ...)

Protocol (identical for every method)
  * train.py selects the checkpoint by VALIDATION weighted-F1 and evaluates the test set exactly ONCE, after training.
  * Hyper-parameters (L1: cb_beta/focal_gamma, L2: ls_eps/ls_neighbor_mass, L4: span_penalty) are chosen on VALIDATION
    scores only (--select: weighted-F1, macro-F1 or their mean). Default: ONE fixed, documented tuning seed (the authors'
    seed). --tune_seeds gives several tuning seeds; a setting is then ranked by its mean validation score.
    No test number is printed or tabulated before tuning has finished.
  * Final comparison over --seeds, all with the same settings and the same code path:
        baseline | L1 | L2 | L1+L2 | L4 | ALL (L1+L2+L4)       (* = tuned settings; L1+L2 and ALL reuse the tuned parts)
  * L3: calibrate.py (temperature scaling + MC-Dropout) is run on EVERY final checkpoint of EVERY method and seed.
  * Results come from the structured JSON files of train.py / calibrate.py (never from parsed terminal output) and are
    consolidated by report.py into  tuning_results/<DS>/{final_results.json, final_results.csv, final_report.md, summary.md}.

Nothing here guarantees that a fix beats the baseline; the tables show exactly what happened.
"""
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report import build_final_report  # noqa: E402

CFG = {
    'MELD': dict(seed=0,
                 l1_full=[(b, g) for b in (0.99, 0.995, 0.999, 0.9995) for g in (0.5, 1.0, 2.0)],
                 l1_quick=[(0.995, 1.0), (0.999, 1.0)],
                 l2_full=[(e, m) for e in (0.02, 0.05, 0.1) for m in (0.5, 0.9)], l2_quick=[(0.05, 0.7)],
                 l4_full=[0.002, 0.005, 0.02, 0.1], l4_quick=[0.005, 0.02]),
    'IEMOCAP': dict(seed=2007,
                    l1_full=[(b, g) for b in (0.999, 0.9999) for g in (0.0, 1.0, 2.0)],
                    l1_quick=[(0.999, 1.0), (0.999, 2.0)],
                    l2_full=[(e, m) for e in (0.02, 0.05, 0.1) for m in (0.5, 0.9)], l2_quick=[(0.05, 0.7)],
                    l4_full=[0.002, 0.005, 0.02, 0.1], l4_quick=[0.005, 0.02]),
}
METHOD_ORDER = ['baseline', 'L1', 'L2', 'L1+L2', 'L4', 'ALL (L1+L2+L4)']


def flags_stem(flags):
    return re.sub(r'[^A-Za-z0-9.]+', '_', ' '.join(flags)).strip('_') or 'baseline'


REQUIRED_RUN_KEYS = ('seed', 'ckpt', 'best_epoch', 'valid', 'test', 'protocol', 'flags')
CAL_VARIANTS = ('Uncalibrated', 'Temp scaling', 'MC-Dropout', 'MC-Dropout + TS')


def run_problems(run, needs_span=False):
    """Why a train.py results dict is unusable (empty list = fine)."""
    p = ['missing key ' + k for k in REQUIRED_RUN_KEYS if k not in run]
    if not p:
        for split in ('valid', 'test'):
            for k in ('accuracy', 'weighted_f1', 'macro_f1', 'per_class', 'confusion_matrix', 'ece', 'nll', 'brier'):
                if k not in run[split]:
                    p.append('{} metrics lack {}'.format(split, k))
        if needs_span and 'span' not in run:
            p.append('L4 run has no span results')
    return p


def calibration_problems(cal):
    """Why a calibrate.py json is unusable (empty list = fine)."""
    if not isinstance(cal, dict):
        return ['not a calibration file']
    p = ['missing key ' + k for k in ('T', 'T_mc', 'test_full', 'test_pairs', 'protocol') if k not in cal]
    for sec in ('test_full', 'test_pairs'):
        for v in CAL_VARIANTS:
            if sec in cal and v not in cal[sec]:
                p.append('{} lacks variant {}'.format(sec, v))
    return p


class Runner:
    def __init__(self, ds, args, out_root=None):
        self.ds, self.args = ds, args
        self.failures = []          # every failed / invalid stage, so nothing can fail silently
        self.out = os.path.join(out_root or os.path.join(HERE, 'tuning_results'), ds)
        for sub in ('logs', 'ckpts', 'calib'):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        self.cache, self.cal_cache = {}, {}

    def train(self, label, flags, seed):
        """Train (or reuse) one run; identical flags + seed are trained once whatever they are called."""
        key = (tuple(flags), seed)
        if key not in self.cache:
            stem = '{}_s{}'.format(flags_stem(flags), seed)
            log, ck, js = [os.path.join(self.out, d, stem + e) for d, e in (('logs', '.txt'), ('ckpts', '.pkl'), ('ckpts', '_results.json'))]
            run = None
            if self.args.resume and os.path.exists(js) and os.path.exists(ck):
                try:
                    run = json.load(open(js))
                    if run_problems(run):
                        run = None          # unreadable / incomplete earlier result: train again instead of trusting it
                except ValueError:
                    run = None
            if run is None:
                cmd = [sys.executable, 'train.py', '--seed', str(seed), '--save_model_path', ck, '--results_json', js] + flags + shlex.split(self.args.extra)
                print('  >> {}  {}'.format(stem, ' '.join(flags) or '(baseline)'), flush=True)
                p = subprocess.run(cmd, cwd=os.path.join(HERE, self.ds), capture_output=True, text=True)
                open(log, 'w').write(p.stdout + '\n' + p.stderr)
                if p.returncode != 0 or not os.path.exists(js):
                    raise RuntimeError('training failed for {} -- last output:\n{}'.format(stem, (p.stdout + p.stderr)[-2500:]))
                run = json.load(open(js))
                bad = run_problems(run)
                if bad:
                    raise RuntimeError('training of {} produced an incomplete results file: {}'.format(stem, bad))
            run['flag_list'], run['stem'] = list(flags), stem
            self.cache[key] = run
            v = run['valid']
            print('     validation: wF1 {:.2f} | macro-F1 {:.2f}{}'.format(
                v['weighted_f1'], v['macro_f1'], ' | mean span {:.1f}'.format(run['span']['mean_span_active_branches']) if run.get('span') else ''), flush=True)
        return self.cache[key]

    def calibrate(self, run):
        """
        Calibrate one checkpoint; returns the calibration dict or None. Never silent: every failure is appended to
        self.failures (with the log path and the error tail) and a stale output file can never be mistaken for a fresh one.
        """
        stem = run['stem']
        if stem in self.cal_cache:
            return self.cal_cache[stem]
        out_dir = os.path.join(self.out, 'calib', stem)
        js = os.path.join(out_dir, 'calibration_{}.json'.format(self.ds))
        cal, reason = None, None
        if self.args.resume and os.path.exists(js):
            try:
                cal = json.load(open(js))
                if calibration_problems(cal):
                    cal = None
            except ValueError:
                cal = None
        attempts = 0
        while cal is None and attempts <= self.args.calib_retries:
            attempts += 1
            if os.path.exists(js):
                os.remove(js)                         # stale output from an earlier run must not be read
            cmd = [sys.executable, 'calibrate.py', '--load_model_state_dir', run['ckpt'], '--out_dir', out_dir] + shlex.split(self.args.calib_args)
            p = subprocess.run(cmd, cwd=os.path.join(HERE, self.ds), capture_output=True, text=True)
            log = os.path.join(self.out, 'logs', stem + '_calibration.txt')
            open(log, 'w').write('attempt {}\n'.format(attempts) + p.stdout + '\n' + p.stderr)
            if p.returncode != 0:
                reason = 'calibrate.py exited with code {}: {}'.format(p.returncode, (p.stderr.strip() or p.stdout.strip())[-400:])
            elif not os.path.exists(js):
                reason = 'calibrate.py finished but wrote no {}'.format(os.path.basename(js))
            else:
                try:
                    cal = json.load(open(js))
                    bad = calibration_problems(cal)
                    if bad:
                        reason, cal = 'calibration file incomplete: {}'.format(bad), None
                except ValueError as e:
                    reason, cal = 'calibration file unreadable: {}'.format(e), None
            if cal is None and attempts <= self.args.calib_retries:
                print('     (calibration of {} failed - retrying: {})'.format(stem, reason[:120]))
        if cal is None:
            self.failures.append({'stage': 'calibration', 'run': stem, 'attempts': attempts, 'reason': reason, 'log': 'logs/{}_calibration.txt'.format(stem)})
            print('     !! CALIBRATION FAILED for {} after {} attempt(s): {}'.format(stem, attempts, (reason or '')[:200]))
        self.cal_cache[stem] = cal
        return cal


def mean_of(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['MELD', 'IEMOCAP'])
    ap.add_argument('--quick', action='store_true', help='much smaller grids')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help="final-comparison seeds (default: tuning seed and the next two)")
    ap.add_argument('--tune_seed', type=int, default=None, help="the fixed tuning seed (default = the authors' seed)")
    ap.add_argument('--tune_seeds', type=int, nargs='+', default=None,
                    help='several tuning seeds: a setting is ranked by its MEAN validation score over these seeds (default: just --tune_seed)')
    ap.add_argument('--extra', type=str, default='', help='extra arguments passed to every train.py, e.g. "--epochs 20"')
    ap.add_argument('--calib_args', type=str, default='', help='extra arguments for calibrate.py, e.g. "--mc_passes 10"')
    ap.add_argument('--select', choices=['wf1', 'macro', 'mean'], default='mean',
                    help='validation score used to choose settings: weighted-F1, macro-F1, or their mean (default)')
    ap.add_argument('--final_only', action='store_true', help='skip tuning; use --config / --L1_ARGS / --L2_ARGS / --L4_ARGS')
    ap.add_argument('--config', type=str, default=None, help='best_config.json from an earlier tuning run (default: tuning_results/<DS>/best_config.json)')
    ap.add_argument('--L1_ARGS', type=str, default=None, help='e.g. "--cb_beta 0.999 --focal_gamma 1.0" (overrides --config)')
    ap.add_argument('--L2_ARGS', type=str, default=None, help='e.g. "--ls_eps 0.05 --ls_neighbor_mass 0.7"')
    ap.add_argument('--L4_ARGS', type=str, default=None, help='e.g. "--span_penalty 0.005"')
    ap.add_argument('--skip_calibration', action='store_true', help='explicitly do not calibrate (the run is then labelled as having no calibration)')
    ap.add_argument('--calib_retries', type=int, default=1, help='how many times to retry a failed calibration before recording it as missing')
    ap.add_argument('--ensemble', action='store_true', help='OPTIONAL extra: also ensemble the seeds of baseline and ALL (ensemble_eval.py)')
    ap.add_argument('--resume', action='store_true', help='reuse finished runs found in tuning_results/ (e.g. after a Kaggle disconnect)')
    ap.add_argument('--keep_ckpts', action='store_true', help='do not delete checkpoints at the end')
    # a pass-through value such as `--extra --legacy_rng_compat` (one word starting with '--') would otherwise be taken by argparse for
    # an option of auto_tune itself -> glue the value to its option: `--extra=--legacy_rng_compat`
    argv, i = [], 1
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in ('--extra', '--calib_args', '--L1_ARGS', '--L2_ARGS', '--L4_ARGS') and i + 1 < len(sys.argv):
            argv.append(tok + '=' + sys.argv[i + 1]); i += 2
        else:
            argv.append(tok); i += 1
    args = ap.parse_args(argv)

    ds, cfg = args.dataset, CFG[args.dataset]
    tune_seed = cfg['seed'] if args.tune_seed is None else args.tune_seed
    tune_seeds = args.tune_seeds or [tune_seed]
    seeds = args.seeds or [tune_seed, tune_seed + 1, tune_seed + 2]
    mode = 'quick' if args.quick else 'full'
    R = Runner(ds, args)
    grid_rows, notes = [], []

    def vscore(r):
        w, m = r['valid']['weighted_f1'], r['valid']['macro_f1']
        return w if args.select == 'wf1' else m if args.select == 'macro' else 0.5 * (w + m)

    if not args.final_only:
        print('=== [1/5] baseline (authors\' code), tuning seed(s) {} ==='.format(tune_seeds))
        base_runs = [R.train('baseline', [], s) for s in tune_seeds]
        base_score = float(np.mean([vscore(r) for r in base_runs]))

        def search(label, cands, to_flags, to_name):
            print('=== grid: {} ({} settings x {} tuning seed(s); chosen by mean VALIDATION score "{}") ==='.format(label, len(cands), len(tune_seeds), args.select))
            best = None
            for c in cands:
                runs = [R.train(to_name(c), to_flags(c), s) for s in tune_seeds]
                sc = [vscore(r) for r in runs]
                grid_rows.append({'limitation': label, 'setting': to_name(c), 'tune_seeds': len(runs),
                                  'valid_wf1': float(np.mean([r['valid']['weighted_f1'] for r in runs])),
                                  'valid_macro': float(np.mean([r['valid']['macro_f1'] for r in runs])),
                                  'score': float(np.mean(sc)), 'score_std': float(np.std(sc, ddof=1)) if len(sc) > 1 else 0.0,
                                  'mean_span': mean_of([r['span']['mean_span_active_branches'] for r in runs if r.get('span')])})
                if best is None or grid_rows[-1]['score'] > best[2]:
                    best = (c, to_flags(c), grid_rows[-1]['score'])
            print('   -> chosen for {}: {} (validation score {:.2f}; baseline {:.2f})'.format(label, to_name(best[0]), best[2], base_score))
            if best[2] < base_score:
                notes.append('{}: no setting in the grid beat the baseline on the validation score "{}" ({:.2f} vs {:.2f}); the best one is used anyway.'.format(
                    label, args.select, best[2], base_score))
                print('   !! ' + notes[-1])
            return best

        print('=== [2/5] tuning each fix on its own (validation only) ===')
        F1 = search('L1', cfg['l1_' + mode], lambda c: ['--lim1', '--cb_beta', str(c[0]), '--focal_gamma', str(c[1])], lambda c: 'L1_b{}_g{}'.format(*c))[1]
        F2 = search('L2', cfg['l2_' + mode], lambda c: ['--lim2', '--ls_eps', str(c[0]), '--ls_neighbor_mass', str(c[1])], lambda c: 'L2_e{}_m{}'.format(*c))[1]
        F4 = search('L4', cfg['l4_' + mode], lambda c: ['--lim4', '--span_penalty', str(c)], lambda c: 'L4_p{}'.format(c))[1]
    else:
        base_score = None
        cfgfile = args.config or os.path.join(R.out, 'best_config.json')
        saved = json.load(open(cfgfile)) if os.path.exists(cfgfile) else {}
        pick = lambda cli, key: cli if cli is not None else saved.get(key, '')
        F1 = ['--lim1'] + shlex.split(pick(args.L1_ARGS, 'L1_ARGS'))
        F2 = ['--lim2'] + shlex.split(pick(args.L2_ARGS, 'L2_ARGS'))
        F4 = ['--lim4'] + shlex.split(pick(args.L4_ARGS, 'L4_ARGS'))
        notes.append('--final_only: settings taken from {} / the command line, not tuned in this run.'.format(cfgfile if saved else 'the command line (defaults of train.py where empty)'))
        print('=== final-only: L1 {} | L2 {} | L4 {} ==='.format(F1, F2, F4))
    best_cfg = {'dataset': ds, 'tune_seeds': tune_seeds, 'select': args.select, 'L1_ARGS': ' '.join(F1[1:]), 'L2_ARGS': ' '.join(F2[1:]), 'L4_ARGS': ' '.join(F4[1:])}
    json.dump(best_cfg, open(os.path.join(R.out, 'best_config.json'), 'w'), indent=2)

    method_flags = {'baseline': [], 'L1': F1, 'L2': F2, 'L1+L2': F1 + F2, 'L4': F4, 'ALL (L1+L2+L4)': F1 + F2 + F4}
    print('=== [3/5] final comparison: {} methods x seeds {} (same protocol for all) ==='.format(len(method_flags), seeds))
    final = {m: [] for m in method_flags}
    for s in seeds:
        for m, fl in method_flags.items():
            final[m].append(R.train(m, fl, s))

    calib = {}
    if not args.skip_calibration:
        print('=== [4/5] L3 calibration (temperature scaling + MC-Dropout) on EVERY final checkpoint ({} runs) ==='.format(sum(len(v) for v in final.values())))
        for m in final:
            for r in final[m]:
                calib[(m, r['seed'])] = R.calibrate(r)

    # ---- completeness: the experiment is only 'COMPLETE' when every required result exists and is valid ----
    methods_needing_span = {m for m, fl in method_flags.items() if '--lim4' in fl}
    missing = []
    for m in final:
        for r in final[m]:
            for why in run_problems(r, needs_span=m in methods_needing_span):
                missing.append({'method': m, 'seed': r['seed'], 'what': 'training result invalid: ' + why})
            if not args.skip_calibration and calib.get((m, r['seed'])) is None:
                missing.append({'method': m, 'seed': r['seed'], 'what': 'calibration result missing'})
    n_runs = sum(len(v) for v in final.values())
    status = {'status': 'COMPLETE' if not missing and not R.failures else 'INCOMPLETE', 'complete': not missing and not R.failures,
              'expected_runs': len(method_flags) * len(seeds), 'trained_runs': n_runs,
              'calibration_required': not args.skip_calibration,
              'expected_calibrations': 0 if args.skip_calibration else len(method_flags) * len(seeds),
              'calibrations_produced': sum(1 for v in calib.values() if v is not None),
              'missing': missing, 'failures': R.failures, 'warnings': []}
    if args.skip_calibration:
        status['warnings'].append('calibration was skipped on request (--skip_calibration): no calibration results exist')
    banner = ''
    if not status['complete']:
        banner = ('RESULTS INCOMPLETE - {} required result(s) missing ({} of {} calibrations produced). Do NOT treat this experiment as finished; '
                  'details in status.json'.format(len(missing), status['calibrations_produced'], status['expected_calibrations']))
        print('\n' + '!' * 100 + '\n!! ' + banner + '\n' + '!' * 100)
        for mm in missing[:12]:
            print('   missing: {} seed {} - {}'.format(mm['method'], mm['seed'], mm['what']))
    print('=== [5/5] report ===')
    protocol = ('checkpoint = best VALIDATION weighted-F1 epoch; test set evaluated once per run after training; settings chosen on validation only '
                '(score "{}", tuning seed(s) {}{}); final seeds {}; calibration: temperature fitted on validation, same ECE bins / MC passes / seed for every checkpoint.'.format(
                    args.select, tune_seeds, '' if not args.final_only else ' - reused', seeds))
    meta = {'notes': notes, 'protocol': protocol, 'settings': best_cfg, 'status': status}
    md = build_final_report(ds, {m: [(r, calib.get((m, r['seed']))) for r in final[m]] for m in final}, R.out, meta)
    json.dump({'dataset': ds, 'meta': meta, 'methods': {m: [{'run': r, 'calib': calib.get((m, r['seed']))} for r in final[m]] for m in final}},
              open(os.path.join(R.out, 'runs.json'), 'w'))

    grid_md = []
    if grid_rows:
        grid_md = ['\n## Grid search (VALIDATION scores only; mean over tuning seed(s) {}; pick = highest "score" = {})\n'.format(tune_seeds, args.select),
                   '| limitation | setting | valid wF1 | valid macro-F1 | score | score std | mean span |', '|---|---|---|---|---|---|---|']
        grid_md.append('| baseline | - | {:.2f} | {:.2f} | {:.2f} | - | - |'.format(
            np.mean([r['valid']['weighted_f1'] for r in base_runs]), np.mean([r['valid']['macro_f1'] for r in base_runs]), base_score))
        for g in grid_rows:
            grid_md.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} |'.format(g['limitation'], g['setting'], g['valid_wf1'], g['valid_macro'], g['score'], g['score_std'],
                                                                                       '{:.1f}'.format(g['mean_span']) if g['mean_span'] is not None else '-'))
        with open(os.path.join(R.out, 'grid.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys())); w.writeheader(); w.writerows(grid_rows)
    head = ['**Chosen settings:** `L1_ARGS="{}"`  `L2_ARGS="{}"`  `L4_ARGS="{}"`\n'.format(best_cfg['L1_ARGS'], best_cfg['L2_ARGS'], best_cfg['L4_ARGS'])]
    open(os.path.join(R.out, 'summary.md'), 'w').write('\n'.join(head) + md + '\n'.join(grid_md) + '\n')

    if args.ensemble:
        os.makedirs(os.path.join(R.out, 'ensemble'), exist_ok=True)
        for m in ('baseline', 'ALL (L1+L2+L4)'):
            if len(final[m]) >= 2:
                tag = 'baseline' if m == 'baseline' else 'all'
                p = subprocess.run([sys.executable, 'ensemble_eval.py', '--ckpts', ','.join(r['ckpt'] for r in final[m]), '--out_dir', os.path.join(R.out, 'ensemble', tag)],
                                   cwd=os.path.join(HERE, ds), capture_output=True, text=True)
                open(os.path.join(R.out, 'ensemble', 'ensemble_{}.txt'.format(tag)), 'w').write(p.stdout + p.stderr)
                print('\n--- ensemble of {} seeds ({}) ---\n{}'.format(len(final[m]), m, p.stdout[-1500:] if p.returncode == 0 else p.stderr[-800:]))
                if p.returncode != 0:
                    status['warnings'].append('optional ensemble of {} failed (see ensemble/ensemble_{}.txt)'.format(m, tag))
                    write_status()
    def write_status():
        json.dump(status, open(os.path.join(R.out, 'status.json'), 'w'), indent=2)
        for marker in ('STATUS_COMPLETE.txt', 'STATUS_INCOMPLETE.txt'):
            if os.path.exists(os.path.join(R.out, marker)):
                os.remove(os.path.join(R.out, marker))
        open(os.path.join(R.out, 'STATUS_COMPLETE.txt' if status['complete'] else 'STATUS_INCOMPLETE.txt'), 'w').write(
            ('All required results exist: {} training runs, {} calibrations.\n'.format(status['trained_runs'], status['calibrations_produced']) if status['complete']
             else banner + '\n'))
    write_status()
    if not args.keep_ckpts:
        keep = {os.path.basename(final[m][0]['ckpt']) for m in final} | {os.path.basename(final[m][0]['ckpt']).replace('.pkl', '_results.json') for m in final}
        for f in os.listdir(os.path.join(R.out, 'ckpts')):
            if f not in keep and not f.endswith('_results.json'):
                os.remove(os.path.join(R.out, 'ckpts', f))
    print('\n' + md)
    print('\nSaved to {}  (final_report.md, final_results.json, final_results.csv, summary.md, status.json, runs.json, grid.csv)'.format(R.out))
    if not status['complete']:
        print('!! ' + banner)
        sys.exit(2)          # a non-zero exit code: scripts / CI can never mistake this for a finished experiment
    print('STATUS: COMPLETE - every required training result and calibration exists.')


if __name__ == '__main__':
    main()
