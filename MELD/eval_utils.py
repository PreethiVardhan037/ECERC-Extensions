"""
[ADDED] One implementation of every reported metric (identical file in IEMOCAP/ and MELD/).

train.py, calibrate.py, ensemble_eval.py and the root-level reporting code all import from here, so classification
and calibration numbers are computed the same way everywhere (equal-width ECE bins, same NLL / Brier definition).
All scores that are percentages are in percent; NLL and Brier are raw (lower is better).
"""
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def ece_score(conf, correct, n_bins=15):
    """Expected Calibration Error with equal-width confidence bins (Guo et al., 2017). Returns a fraction in [0, 1]."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll_score(prob, y):
    prob = np.asarray(prob, dtype=np.float64)
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None)).mean())


def brier_score(prob, y):
    """Multi-class Brier score: mean over samples of the squared distance to the one-hot label (range 0..2)."""
    prob = np.asarray(prob, dtype=np.float64)
    return float(((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1).mean())


def probabilistic_metrics(prob, y, n_bins=15):
    prob = np.asarray(prob, dtype=np.float64)
    pred, conf = prob.argmax(1), prob.max(1)
    return {'ece': 100 * ece_score(conf, (pred == y).astype(np.float64), n_bins),
            'nll': nll_score(prob, y), 'brier': brier_score(prob, y), 'avg_conf': 100 * float(conf.mean())}


def classification_metrics(y, pred, target_names):
    """Accuracy, weighted/macro P-R-F1, per-class P/R/F1/support and the confusion matrix (rows = true, cols = predicted)."""
    labels = list(range(len(target_names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {'accuracy': 100 * accuracy_score(y, pred),
           'weighted_f1': 100 * f1_score(y, pred, average='weighted', labels=labels, zero_division=0),
           'macro_f1': 100 * f1_score(y, pred, average='macro', labels=labels, zero_division=0),
           'per_class': {n: {'precision': 100 * float(p[i]), 'recall': 100 * float(r[i]), 'f1': 100 * float(f[i]), 'support': int(s[i])}
                         for i, n in enumerate(target_names)},
           'confusion_matrix': confusion_matrix(y, pred, labels=labels).tolist(), 'class_order': list(target_names)}
    for avg in ('weighted', 'macro'):
        pa, ra, _, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        out[avg + '_precision'], out[avg + '_recall'] = 100 * float(pa), 100 * float(ra)
    return out


def full_metrics(prob, y, target_names, n_bins=15):
    """Everything the final report needs for one set of class probabilities."""
    prob = np.asarray(prob, dtype=np.float64)
    out = classification_metrics(y, prob.argmax(1), target_names)
    out.update(probabilistic_metrics(prob, y, n_bins))
    out['n'] = int(len(y))
    return out


def pair_breakdown(prob, y, pairs, target_names, n_bins=15):
    """
    Per confusable pair (a, b) -- NOT pooled -- with explicit populations and sample counts:
      true_in_pair : utterances whose TRUE label is a or b            (ECE of the model's top-1 confidence)
      pred_in_pair : utterances whose PREDICTED label is a or b       (the population the old pooled figure used)
      confusions   : true a predicted b and true b predicted a, with the model's mean confidence on those errors
    """
    prob = np.asarray(prob, dtype=np.float64)
    pred, conf = prob.argmax(1), prob.max(1)
    correct = (pred == y).astype(np.float64)
    out = {}
    for a, b in pairs:
        t_in, p_in = np.isin(y, [a, b]), np.isin(pred, [a, b])
        ab, ba = (y == a) & (pred == b), (y == b) & (pred == a)
        err = ab | ba
        out['{}-{}'.format(target_names[a], target_names[b])] = {
            'true_in_pair': {'n': int(t_in.sum()), 'ece': 100 * ece_score(conf[t_in], correct[t_in], n_bins),
                             'accuracy': 100 * float(correct[t_in].mean()) if t_in.any() else float('nan'),
                             'nll': nll_score(prob[t_in], y[t_in]) if t_in.any() else float('nan')},
            'pred_in_pair': {'n': int(p_in.sum()), 'ece': 100 * ece_score(conf[p_in], correct[p_in], n_bins),
                             'accuracy': 100 * float(correct[p_in].mean()) if p_in.any() else float('nan')},
            'confusions': {'{}->{}'.format(target_names[a], target_names[b]): int(ab.sum()),
                           '{}->{}'.format(target_names[b], target_names[a]): int(ba.sum()),
                           'rate_of_true_in_pair': 100 * float(err.sum()) / max(int(t_in.sum()), 1),
                           'mean_conf_on_confusions': 100 * float(conf[err].mean()) if err.any() else float('nan')}}
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def dump_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=2)
