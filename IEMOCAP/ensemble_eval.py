"""
[ADDED] OPTIONAL extra (not one of the four limitations): inference-time ensemble of several trained checkpoints.
Identical file in IEMOCAP/ and MELD/. Run it from inside the dataset folder.

    python ensemble_eval.py --ckpts ckpt_all_s0.pkl,ckpt_all_s1.pkl,ckpt_all_s2.pkl

It loads every checkpoint (any mix of baseline / --lim1 / --lim2 / --lim4 models, same architecture), averages their
class probabilities, and reports test accuracy / weighted-F1 / macro-F1 / ECE for
    * each member on its own,
    * the ensemble (mean of probabilities),
    * the ensemble + temperature scaling (T fitted on the VALIDATION set only; nothing is fitted on test).
No architecture change and no retraining. Report ensemble numbers separately from single-model numbers: ensembling
helps the baseline too, so the fair comparison is "ensemble of baseline seeds" vs "ensemble of fixed-model seeds".
"""
import argparse
import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score

from calibrate import (DATASET, CONFIG, get_loaders, build_model, collect_logits, softmax_np,
                       fit_temperature, compute_metrics, parse_confusable_pairs)


def score(prob, y, conf_classes, n_bins):
    m = compute_metrics(prob, y, conf_classes, n_bins)
    m['macro_f1'] = 100 * f1_score(y, prob.argmax(1), average='macro')
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', type=str, required=True, help='comma-separated checkpoints written by train.py')
    p.add_argument('--feature_type', type=str, default='multi')
    p.add_argument('--base_layer', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    p.add_argument('--valid_rate', type=float, default=0.1, help='IEMOCAP only: validation fraction of train conversations')
    p.add_argument('--n_bins', type=int, default=15)
    p.add_argument('--no_cuda', action='store_true', default=False)
    p.add_argument('--confusable_pairs', type=str, default=CONFIG['pairs'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='ensemble_out')
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    os.makedirs(args.out_dir, exist_ok=True)
    names = CONFIG['target_names']
    conf_classes = sorted({c for pr in parse_confusable_pairs(args.confusable_pairs, names) for c in pr})
    ckpts = [c.strip() for c in args.ckpts.split(',') if c.strip()]
    assert len(ckpts) >= 2, 'give at least two checkpoints in --ckpts'
    valid_loader, test_loader = get_loaders(args.batch_size, args.valid_rate)

    zv_all, zt_all, yv, yt = [], [], None, None
    for ck in ckpts:
        args.load_model_state_dir = ck           # read by calibrate.build_model
        model = build_model(args, cuda_flag)
        model.eval()                             # dropout OFF: deterministic members
        zv, yv = collect_logits(model, valid_loader, args, cuda_flag)
        zt, yt = collect_logits(model, test_loader, args, cuda_flag)
        zv_all.append(zv); zt_all.append(zt)
        del model

    rows = {}
    for ck, zt in zip(ckpts, zt_all):
        rows[os.path.basename(ck)] = score(softmax_np(zt), yt, conf_classes, args.n_bins)
    pv = np.mean([softmax_np(z) for z in zv_all], axis=0)
    pt = np.mean([softmax_np(z) for z in zt_all], axis=0)
    T = fit_temperature(np.log(np.clip(pv, 1e-12, None)), yv)     # validation only
    pt_ts = softmax_np(np.log(np.clip(pt, 1e-12, None)), T)
    rows['ENSEMBLE (mean prob)'] = score(pt, yt, conf_classes, args.n_bins)
    rows['ENSEMBLE + temp scaling'] = score(pt_ts, yt, conf_classes, args.n_bins)

    cols = [('acc', 'Acc'), ('weighted_f1', 'wF1'), ('macro_f1', 'macroF1'), ('ece', 'ECE'), ('nll', 'NLL')]
    print('\n=== {} | TEST | ensemble of {} checkpoints (T={:.3f} fitted on validation) ==='.format(DATASET, len(ckpts), T))
    print('{:<48}'.format('model') + ''.join('{:>10}'.format(c[1]) for c in cols))
    for name, r in rows.items():
        print('{:<48}'.format(name[:47]) + ''.join('{:>10.3f}'.format(r[k]) for k, _ in cols))
    mem = [r for n, r in rows.items() if not n.startswith('ENSEMBLE')]
    print('{:<48}'.format('members: mean') + ''.join('{:>10.3f}'.format(np.mean([r[k] for r in mem])) for k, _ in cols))
    print('{:<48}'.format('members: std') + ''.join('{:>10.3f}'.format(np.std([r[k] for r in mem])) for k, _ in cols))
    with open(os.path.join(args.out_dir, 'ensemble_{}.json'.format(DATASET)), 'w') as f:
        json.dump({'dataset': DATASET, 'ckpts': ckpts, 'T': T, 'rows': rows}, f, indent=2)
    print('Saved to {}/'.format(args.out_dir))


if __name__ == '__main__':
    main()
