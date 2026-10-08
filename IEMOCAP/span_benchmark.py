"""
[ADDED] Limitation 4 -- accuracy / weighted-F1 / calibration (ECE) versus attention span.
Identical file in IEMOCAP/ and MELD/. Run it from inside the dataset folder.

MODE A  (fixed-span sweep, no training):
    python span_benchmark.py --load_model_state_dir ECERC_MODEL.pkl
    Takes ONE trained checkpoint and, at test time only, cuts the history each of the four Evidence-Cause
    Interaction branches may attend to (hard span of w utterances, soft ramp as in AdaptiveSpan) for
    w in --spans. 'full' = no cut = the original model (sanity row, must equal the normal test score).
    It shows how much distant context the model really uses. If the checkpoint was trained with --lim4 an
    extra row 'learned' (the spans it learned itself) is added first.

MODE B  (learned-span curve, needs trained checkpoints):
    python span_benchmark.py --learned_ckpts ckpt_lim4_p0.pkl,ckpt_lim4_p0.005.pkl,...
    Evaluates each --lim4 checkpoint AS TRAINED and reports its learned spans next to test accuracy,
    weighted-F1 and ECE ("accuracy vs learned attention span"). run_span_sweep.sh produces such checkpoints
    with different --span_penalty values.

Both modes also print how long the dialogues are (IEMOCAP ~49 utterances/conversation, MELD ~9.6), the
contrast the SOP asks Member 2 to analyse. ECE here is the UNcalibrated ECE (Member 3's comparison of the
adaptive-span model against the full-attention baseline); for before/after temperature scaling run
calibrate.py on the same checkpoint.
"""
import argparse
import csv
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calibrate import (DATASET, CONFIG, ECERC, get_loaders, collect_logits, softmax_np,
                       compute_metrics, configure_span_from_state, parse_confusable_pairs)

SPAN_MAX = {'MELD': 40.0, 'IEMOCAP': 120.0}[DATASET]
DEFAULT_SPANS = {'MELD': '1,2,3,5,8,12,20,full',
                 'IEMOCAP': '1,2,3,5,8,12,20,40,80,full'}[DATASET]


def length_stats(loader, name):
    ds = loader.dataset
    ds = ds.dataset if isinstance(ds, Subset) else ds
    lab = ds.emotion_labels if hasattr(ds, 'emotion_labels') else ds.videoLabels
    lens = np.array([len(lab[k]) for k in ds.keys])
    print('{} dialogue length ({}): n={} mean={:.1f} median={:.0f} max={}'.format(
        DATASET, name, len(lens), lens.mean(), np.median(lens), lens.max()))
    return {'n': int(len(lens)), 'mean': float(lens.mean()), 'median': float(np.median(lens)), 'max': int(lens.max())}


def make_model(args, state, cuda_flag):
    """Build the model so that `state` loads; baseline checkpoints get span modules (identity at init) added."""
    d_t, d_a, d_v = 1024, CONFIG['d_audio'], 342
    input_size = d_t + d_a + d_v if args.feature_type == 'multi' else d_t
    has_span = configure_span_from_state(args, state)
    if not has_span:
        args.adaptive_span = True            # attach (identity) span modules to a full-attention checkpoint
        args.span_per_head = False
    args.span_max, args.span_ramp, args.span_init = SPAN_MAX, args.ramp, 1.0
    model = ECERC(args, d_t=d_t, d_a=d_a, d_v=d_v, base_layer=args.base_layer, input_size=input_size,
                  hidden_size=128, n_speakers=CONFIG['n_speakers'], n_classes=CONFIG['n_classes'], cuda_flag=cuda_flag)
    if has_span:
        model.load_state_dict(state)
    else:
        res = model.load_state_dict(state, strict=False)
        bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if not k.startswith('span_modules.')]
        assert not bad, 'checkpoint does not match the model: {}'.format(bad)
    if cuda_flag:
        model.cuda()
    model.eval()
    return model, has_span


def evaluate(model, loader, args, cuda_flag, conf_classes):
    z, y = collect_logits(model, loader, args, cuda_flag)
    m = compute_metrics(softmax_np(z), y, conf_classes, args.n_bins)
    return {k: m[k] for k in ('acc', 'weighted_f1', 'ece', 'ece_confusable_preds', 'nll', 'avg_conf')}


def fmt_row(r):
    return '{:<24}{:>9}{:>9.2f}{:>9.2f}{:>9.2f}{:>13.2f}{:>9.4f}{:>9.2f}'.format(
        r['label'], r['span'], r['acc'], r['weighted_f1'], r['ece'], r['ece_confusable_preds'], r['nll'], r['avg_conf'])


HEADER = '{:<24}{:>9}{:>9}{:>9}{:>9}{:>13}{:>9}{:>9}'.format(
    'setting', 'span', 'Acc', 'wF1', 'ECE', 'ECE(conf.)', 'NLL', 'AvgConf')


def save_outputs(rows, args, title, xlabel, stem):
    os.makedirs(args.out_dir, exist_ok=True)
    keys = ['label', 'span', 'acc', 'weighted_f1', 'ece', 'ece_confusable_preds', 'nll', 'avg_conf', 'detail']
    with open(os.path.join(args.out_dir, stem + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.out_dir, stem + '.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    pts = [r for r in rows if r.get('x') is not None]
    if len(pts) >= 2:
        pts = sorted(pts, key=lambda r: r['x'])
        xs = [r['x'] for r in pts]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(xs, [r['weighted_f1'] for r in pts], 'o-', color='tab:blue', label='weighted F1')
        ax.plot(xs, [r['acc'] for r in pts], 's--', color='tab:cyan', label='accuracy')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('test F1 / accuracy (%)')
        if min(xs) > 0 and max(xs) / max(min(xs), 1e-9) > 20:
            ax.set_xscale('log')
        ax2 = ax.twinx()
        ax2.plot(xs, [r['ece'] for r in pts], '^-', color='tab:red', label='ECE (uncalibrated)')
        ax2.set_ylabel('ECE (%)', color='tab:red')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, stem + '.png'), dpi=150)
        plt.close(fig)
    print('Saved {}.csv/.json/.png to {}/'.format(stem, args.out_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--load_model_state_dir', type=str, default=None, help='MODE A: one checkpoint (baseline or --lim4)')
    p.add_argument('--spans', type=str, default=DEFAULT_SPANS, help="MODE A: fixed spans in utterances, 'full' = unrestricted")
    p.add_argument('--learned_ckpts', type=str, default=None, help='MODE B: comma-separated --lim4 checkpoints')
    p.add_argument('--ramp', type=float, default=2.0, help='soft-ramp length (utterances) for MODE A fixed spans')
    p.add_argument('--feature_type', type=str, default='multi')
    p.add_argument('--base_layer', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    p.add_argument('--valid_rate', type=float, default=0.1)
    p.add_argument('--n_bins', type=int, default=15)
    p.add_argument('--no_cuda', action='store_true', default=False)
    p.add_argument('--confusable_pairs', type=str, default=CONFIG['pairs'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='span_out')
    args = p.parse_args()
    assert args.load_model_state_dir or args.learned_ckpts, 'give --load_model_state_dir (mode A) and/or --learned_ckpts (mode B)'

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    names = CONFIG['target_names']
    conf_classes = sorted({c for pr in parse_confusable_pairs(args.confusable_pairs, names) for c in pr})
    valid_loader, test_loader = get_loaders(args.batch_size, args.valid_rate)
    stats = {'test': length_stats(test_loader, 'test')}
    print('Dataset: {} | span_max = {} utterances\n'.format(DATASET, SPAN_MAX))

    # ----------------------------- MODE A -----------------------------
    if args.load_model_state_dir:
        state = torch.load(args.load_model_state_dir, map_location='cuda' if cuda_flag else 'cpu')
        model, has_span = make_model(args, state, cuda_flag)
        rows = []
        if has_span:
            r = evaluate(model, test_loader, args, cuda_flag, conf_classes)
            r.update(label='learned', span='%.1f' % model.mean_span(), x=model.mean_span(), detail=model.get_spans_str())
            rows.append(r)
        for tok in [t.strip() for t in args.spans.split(',') if t.strip()]:
            w = SPAN_MAX if tok == 'full' else float(tok)
            model.set_fixed_span(w)
            r = evaluate(model, test_loader, args, cuda_flag, conf_classes)
            r.update(label='fixed' if tok != 'full' else 'full (no cut)', span=tok, x=w, detail='')
            rows.append(r)
        print('=== MODE A: test accuracy / F1 / ECE vs attention span | {} ===\n{}'.format(args.load_model_state_dir, HEADER))
        for r in rows:
            print(fmt_row(r))
        print('(Acc / wF1 / ECE / AvgConf in %; ECE is uncalibrated; "full" must equal the checkpoint\'s normal test score)')
        save_outputs(rows, args, '{}: test performance vs attention span'.format(DATASET),
                     'attention span (utterances; full = {:.0f})'.format(SPAN_MAX), 'span_sweep_{}'.format(DATASET))

    # ----------------------------- MODE B -----------------------------
    if args.learned_ckpts:
        rows = []
        for ck in [c.strip() for c in args.learned_ckpts.split(',') if c.strip()]:
            state = torch.load(ck, map_location='cuda' if cuda_flag else 'cpu')
            if not any(k.startswith('span_modules.') for k in state):
                print('skip {} (no adaptive-span parameters; train it with --lim4)'.format(ck)); continue
            model, _ = make_model(args, state, cuda_flag)
            r = evaluate(model, test_loader, args, cuda_flag, conf_classes)
            r.update(label=os.path.basename(ck).replace('.pkl', '').replace('ckpt_', '')[:23], span='%.1f' % model.mean_span(),
                     x=model.mean_span(), detail=model.get_spans_str())
            rows.append(r)
        print('\n=== MODE B: learned spans vs test performance | {} ===\n{}'.format(DATASET, HEADER))
        for r in rows:
            print(fmt_row(r) + '   [' + r['detail'] + ']')
        save_outputs(rows, args, '{}: performance vs learned span'.format(DATASET),
                     'mean learned span (utterances)', 'learned_span_curve_{}'.format(DATASET))
    with open(os.path.join(args.out_dir, 'dialogue_length_stats_{}.json'.format(DATASET)), 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
