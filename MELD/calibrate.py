"""
[ADDED] Limitation 3 -- confidence calibration for ECERC (post-hoc, NO retraining, model.py untouched).

    * Temperature Scaling   (Guo et al., ICML 2017): one scalar T fitted on the VALIDATION set
                            by minimising NLL; test probabilities = softmax(logits / T).
    * Monte-Carlo Dropout   (Gal & Ghahramani, ICML 2016): dropout kept ON at test time,
                            S stochastic forward passes, probabilities averaged.
    * MC-Dropout + TS       : T fitted on the validation MC-averaged probabilities.

Reports Accuracy / weighted-F1 / ECE / NLL / Brier before vs after, and ECE separately for
predictions that fall in the confusable classes of the dataset (MELD: Disgust/Angry; IEMOCAP:
Happy/Excited and Angry/Frustrated) vs. all other predictions. Saves a JSON of results and reliability diagrams.

This file is identical in IEMOCAP/ and MELD/. Run it from inside the dataset folder:

    cd IEMOCAP   (or MELD)
    python calibrate.py --load_model_state_dir ECERC_MODEL.pkl        # authors' released checkpoint
    python calibrate.py --load_model_state_dir ECERC_trained.pkl      # a checkpoint saved by train.py

Note on logits: ECERC's forward() ends with log_softmax. softmax(log_prob / T) is identical to
softmax(logits / T) because log_softmax only subtracts a per-row constant, so log_prob is used
directly as "logits" and the model needs no change.
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ECERC
from loss import parse_confusable_pairs

import warnings
warnings.filterwarnings("ignore")

# The dataset is identified by which loader this folder ships with.
try:
    from dataloader import MELDRobertaDataset
    DATASET = 'MELD'
except ImportError:
    from dataloader import IEMOCAPRobertaDataset
    DATASET = 'IEMOCAP'

CONFIG = {
    'MELD': dict(n_classes=7, n_speakers=9, d_audio=300,
                 target_names=['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang'],
                 pairs='dis-ang', batch_size=32),
    'IEMOCAP': dict(n_classes=6, n_speakers=2, d_audio=1582,
                    target_names=['hap', 'sad', 'neu', 'ang', 'exc', 'fru'],
                    pairs='hap-exc,ang-fru', batch_size=64),
}[DATASET]


# --------------------------------------------------------------------------------------
# data / model (mirrors the authors' inference.py so the same checkpoints load)
# --------------------------------------------------------------------------------------
def get_loaders(batch_size, valid_rate):
    """Deterministic (non-shuffled) validation and test loaders."""
    if DATASET == 'MELD':
        validset, testset = MELDRobertaDataset('valid'), MELDRobertaDataset('test')
        valid_ds = validset
    else:
        trainset = IEMOCAPRobertaDataset(train=True)
        split = int(valid_rate * len(trainset))          # same split rule as the authors' train.py
        valid_ds = Subset(trainset, list(range(len(trainset)))[:split])
        validset, testset = trainset, IEMOCAPRobertaDataset(train=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, collate_fn=validset.collate_fn, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, num_workers=0)
    return valid_loader, test_loader


def configure_span_from_state(args, state):
    """[L4] A checkpoint trained with --lim4 contains span_modules.*; switch the model on and match per-head/per-branch."""
    keys = [k for k in state if k.startswith('span_modules.') and k.endswith('span_ratio')]
    args.adaptive_span = len(keys) > 0
    args.span_per_head = bool(keys) and state[keys[0]].shape[0] > 1
    return args.adaptive_span


def build_model(args, cuda_flag):
    d_t, d_a, d_v = 1024, CONFIG['d_audio'], 342
    input_size = d_t + d_a + d_v if args.feature_type == 'multi' else d_t
    state = torch.load(args.load_model_state_dir, map_location='cuda' if cuda_flag else 'cpu')
    is_l4 = configure_span_from_state(args, state)       # [ADDED] Limitation 4 checkpoints load automatically
    if is_l4:
        print('Checkpoint contains adaptive-span parameters (Limitation 4) -> building the model with spans enabled')
    model = ECERC(args, d_t=d_t, d_a=d_a, d_v=d_v, base_layer=args.base_layer, input_size=input_size,
                  hidden_size=128, n_speakers=CONFIG['n_speakers'], n_classes=CONFIG['n_classes'],
                  cuda_flag=cuda_flag)
    model.load_state_dict(state)
    if is_l4:
        print('Learned spans (utterances): {} | mean {:.2f}'.format(model.get_spans_str(), model.mean_span()))
    if cuda_flag:
        model.cuda()
    return model


def enable_mc_dropout(model):
    """Eval mode everywhere, but every nn.Dropout stays stochastic (LayerNorm has no train/eval difference)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def collect_logits(model, loader, args, cuda_flag):
    """One pass over `loader`; returns (log_prob [N, C], labels [N]) as numpy arrays."""
    all_lp, all_y = [], []
    for data in loader:
        emo, sem, audio, vision, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.feature_type == 'multi':
            emo = torch.cat([emo, audio, vision], dim=-1)
            sem = torch.cat([sem], dim=-1)
        log_prob = model(emo, sem, qmask, umask, seq_lengths)
        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        all_lp.append(log_prob.float().cpu().numpy())
        all_y.append(label.cpu().numpy())
    return np.concatenate(all_lp), np.concatenate(all_y)


# --------------------------------------------------------------------------------------
# calibration methods
# --------------------------------------------------------------------------------------
def _nll_at(z, y, T, mc):
    if mc:   # z: [S, N, C]
        p = F.softmax(z / T, dim=-1).mean(0)
        return -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)).mean()
    return F.cross_entropy(z / T, y)


def fit_temperature(z_np, y_np, mc=False):
    """Fit the scalar T>0 minimising validation NLL (L-BFGS on log T, then a safety grid check)."""
    z = torch.as_tensor(z_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = _nll_at(z, y, log_t.exp(), mc)
        loss.backward()
        return loss
    try:
        opt.step(closure)
        t_lbfgs = float(log_t.exp())
    except Exception:
        t_lbfgs = 1.0
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 300))
    with torch.no_grad():
        grid_nll = [float(_nll_at(z, y, torch.tensor(g, dtype=torch.float64), mc)) for g in grid]
        best_grid = float(grid[int(np.argmin(grid_nll))])
        cands = [t for t in (t_lbfgs, best_grid) if np.isfinite(t) and t > 0]
        return min(cands, key=lambda t: float(_nll_at(z, y, torch.tensor(t, dtype=torch.float64), mc)))


def softmax_np(z, T=1.0):
    return F.softmax(torch.as_tensor(z, dtype=torch.float64) / T, dim=-1).numpy()


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def ece_score(conf, correct, n_bins=15):
    """Expected Calibration Error with equal-width confidence bins (Guo et al., 2017)."""
    if len(conf) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(prob, y, confusable_classes, n_bins):
    pred = prob.argmax(1)
    conf = prob.max(1)
    correct = (pred == y).astype(np.float64)
    in_pair = np.isin(pred, list(confusable_classes))      # prediction falls in a confusable class
    onehot = np.eye(prob.shape[1])[y]
    return {
        'acc': 100 * accuracy_score(y, pred),
        'weighted_f1': 100 * f1_score(y, pred, average='weighted'),
        'ece': 100 * ece_score(conf, correct, n_bins),
        'ece_confusable_preds': 100 * ece_score(conf[in_pair], correct[in_pair], n_bins),
        'ece_other_preds': 100 * ece_score(conf[~in_pair], correct[~in_pair], n_bins),
        'n_confusable_preds': int(in_pair.sum()),
        'n_other_preds': int((~in_pair).sum()),
        'nll': float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None)).mean()),
        'brier': float(((prob - onehot) ** 2).sum(1).mean()),
        'avg_conf': 100 * float(conf.mean()),
    }


def print_table(title, results):
    cols = [('acc', 'Acc'), ('weighted_f1', 'wF1'), ('ece', 'ECE'), ('ece_confusable_preds', 'ECE(conf.)'),
            ('ece_other_preds', 'ECE(other)'), ('nll', 'NLL'), ('brier', 'Brier'), ('avg_conf', 'AvgConf')]
    print('\n' + title)
    print('{:<20}'.format('method') + ''.join('{:>12}'.format(c[1]) for c in cols))
    for name, r in results.items():
        print('{:<20}'.format(name) + ''.join('{:>12.4f}'.format(r[k]) for k, _ in cols))
    first = next(iter(results.values()))
    print('(ECE/Acc/wF1/AvgConf in %; ECE(conf.) over {} predictions in confusable classes, ECE(other) over {})'
          .format(first['n_confusable_preds'], first['n_other_preds']))


def plot_reliability(probs_by_method, y, n_bins, path, title):
    n = len(probs_by_method)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, (name, prob) in zip(axes[0], probs_by_method.items()):
        conf, correct = prob.max(1), (prob.argmax(1) == y).astype(float)
        accs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            accs.append(correct[m].mean() if m.any() else 0.0)
        ax.bar(centers, accs, width=1.0 / n_bins, edgecolor='k', alpha=0.8, label='accuracy')
        ax.plot([0, 1], [0, 1], 'r--', label='perfect calibration')
        ax.set_title('{}\nECE = {:.2f}%'.format(name, 100 * ece_score(conf, correct, n_bins)))
        ax.set_xlabel('confidence')
        ax.set_ylabel('accuracy')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_state_dir', type=str, default='ECERC_MODEL.pkl', help='checkpoint to calibrate')
    parser.add_argument('--feature_type', type=str, default='multi', help='feature type: text/multi')
    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--valid_rate', type=float, default=0.1, help='IEMOCAP only: validation fraction of train conversations')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--n_bins', type=int, default=15, help='ECE bins')
    parser.add_argument('--mc_passes', type=int, default=30, help='MC-Dropout stochastic forward passes')
    parser.add_argument('--confusable_pairs', type=str, default=CONFIG['pairs'],
                        help="confusable class pairs for the breakdown, e.g. 'hap-exc,ang-fru'")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='calibration_out')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    if cuda_flag:
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    names = CONFIG['target_names']
    pairs = parse_confusable_pairs(args.confusable_pairs, names)
    conf_classes = sorted({c for p in pairs for c in p})
    print('Dataset: {} | checkpoint: {} | confusable classes: {}'.format(
        DATASET, args.load_model_state_dir, [names[c] for c in conf_classes]))

    model = build_model(args, cuda_flag)
    valid_loader, test_loader = get_loaders(args.batch_size, args.valid_rate)

    # ---- deterministic logits (dropout OFF) ----
    model.eval()
    zv, yv = collect_logits(model, valid_loader, args, cuda_flag)
    zt, yt = collect_logits(model, test_loader, args, cuda_flag)
    print('validation utterances: {} | test utterances: {}'.format(len(yv), len(yt)))

    # ---- Temperature Scaling: fit T on validation only ----
    T = fit_temperature(zv, yv)
    print('Fitted temperature T = {:.4f}  (T>1 softens over-confident outputs, T<1 sharpens)'.format(T))

    # ---- MC-Dropout: S stochastic passes on validation and test ----
    enable_mc_dropout(model)
    zv_mc = np.stack([collect_logits(model, valid_loader, args, cuda_flag)[0] for _ in range(args.mc_passes)])
    zt_mc = np.stack([collect_logits(model, test_loader, args, cuda_flag)[0] for _ in range(args.mc_passes)])
    model.eval()
    T_mc = fit_temperature(zv_mc, yv, mc=True)
    print('Fitted temperature for MC-Dropout average, T_mc = {:.4f}'.format(T_mc))

    def mc_prob(z_mc, temp=1.0):
        return np.mean([softmax_np(z, temp) for z in z_mc], axis=0)

    probs = {
        'test': {
            'Uncalibrated': softmax_np(zt),
            'Temp scaling': softmax_np(zt, T),
            'MC-Dropout': mc_prob(zt_mc),
            'MC-Dropout + TS': mc_prob(zt_mc, T_mc),
        },
        'valid': {
            'Uncalibrated': softmax_np(zv),
            'Temp scaling': softmax_np(zv, T),
            'MC-Dropout': mc_prob(zv_mc),
            'MC-Dropout + TS': mc_prob(zv_mc, T_mc),
        },
    }
    labels = {'test': yt, 'valid': yv}

    summary = {'dataset': DATASET, 'checkpoint': args.load_model_state_dir, 'T': T, 'T_mc': T_mc,
               'mc_passes': args.mc_passes, 'n_bins': args.n_bins,
               'confusable_classes': [names[c] for c in conf_classes]}
    for split in ('valid', 'test'):
        res = {m: compute_metrics(p, labels[split], conf_classes, args.n_bins) for m, p in probs[split].items()}
        summary[split] = res
        tag = 'TEST (T fitted on validation -> no leakage)' if split == 'test' else 'VALIDATION (the set T was fitted on)'
        print_table('=== {} | {} ==='.format(DATASET, tag), res)

    with open(os.path.join(args.out_dir, 'calibration_{}.json'.format(DATASET)), 'w') as f:
        json.dump(summary, f, indent=2)
    plot_reliability(probs['test'], yt, args.n_bins,
                     os.path.join(args.out_dir, 'reliability_{}_test.png'.format(DATASET)),
                     '{} test set - reliability diagrams'.format(DATASET))
    print('\nSaved results to {}/'.format(args.out_dir))


if __name__ == '__main__':
    main()
