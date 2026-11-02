import torch
import torch.nn as nn
from torch.autograd import Variable


class Loss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = input.gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# =====================================================================================
# [ADDED] Limitation 1 (class imbalance) + Limitation 2 (category confusability)
# Everything above this line is the authors' original code, untouched.
# =====================================================================================
class ECERCImprovedLoss(nn.Module):
    """
    One configurable loss that covers both individually-owned loss-level fixes.
    It consumes the model's log-probabilities (the ECERC forward already ends in
    log_softmax) and flattened integer targets, exactly like the authors' `Loss`.

        loss_i = alpha[y_i] * (1 - p_t)^gamma * ( - sum_c q_ic * log p_ic )

    Limitation 1 : alpha = Class-Balanced weights (Cui et al., CVPR 2019),
                   (1 - beta) / (1 - beta^n_c), normalised so sum(alpha) = C,
                   and gamma > 0 turns on the Focal modulating factor (Lin et al., ICCV 2017).
    Limitation 2 : q = Confusion-aware label smoothing (extension of Szegedy et al., CVPR 2016).
                   Smoothing mass `smoothing` is NOT spread uniformly: a fraction
                   `neighbor_mass` goes to the paper's confusable partner class(es) of the
                   true class, the remaining mass is spread uniformly over the NON-partner
                   classes only (e.g. eps=0.1, neighbor_mass=0.7: true 90%, partner 7%, others share 3%).

    With class_counts=None, gamma=0, smoothing=0 this is plain NLL (cross-entropy).
    """

    def __init__(self, n_classes, class_counts=None, cb_beta=0.999, gamma=0.0,
                 smoothing=0.0, confusable_pairs=None, neighbor_mass=0.7, alpha=None):
        # `alpha`: explicit per-class weights, used ONLY when class_counts is None. train.py passes the authors' own
        # class weights here for an L2-only run, so that run differs from the baseline by the label smoothing alone.
        super(ECERCImprovedLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0 and 0.0 <= neighbor_mass <= 1.0
        self.n_classes = n_classes
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

        # ---- Limitation 1: class-balanced weights from training label counts ----
        self.alpha = None
        if class_counts is not None:
            assert 0.0 <= cb_beta < 1.0, 'cb_beta must be in [0, 1) (beta = 1 divides by zero)'
            counts = torch.as_tensor(class_counts, dtype=torch.float)
            present = counts > 0                      # a class with zero training examples gets weight 0 and is left out
            assert bool(present.any()), 'no training examples'   # of the normalisation, instead of a huge fake weight
            effective_num = (1.0 - torch.pow(cb_beta, counts.clamp(min=1.0))) / (1.0 - cb_beta)
            w = torch.where(present, 1.0 / effective_num, torch.zeros_like(effective_num))
            self.alpha = w / w.sum() * int(present.sum())   # mean weight over the PRESENT classes is exactly 1
        elif alpha is not None:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float).detach().clone()

        # ---- Limitation 2: row y of this matrix = target distribution when the label is y ----
        self.soft_targets = None
        if self.smoothing > 0:
            neighbors = {c: [] for c in range(n_classes)}
            for a, b in (confusable_pairs or []):
                if b not in neighbors[a]: neighbors[a].append(b)
                if a not in neighbors[b]: neighbors[b].append(a)
            T = torch.zeros(n_classes, n_classes)
            for y in range(n_classes):
                T[y, y] = 1.0 - self.smoothing
                near = neighbors[y]
                far = [c for c in range(n_classes) if c != y and c not in near]   # NON-partner classes only
                if near and far:
                    near_share, far_share = self.smoothing * neighbor_mass, self.smoothing * (1.0 - neighbor_mass)
                elif near:      # every other class is a partner: all smoothing mass goes to the partners
                    near_share, far_share = self.smoothing, 0.0
                else:           # class without a confusable partner: plain uniform smoothing
                    near_share, far_share = 0.0, self.smoothing
                for c in near:
                    T[y, c] += near_share / len(near)
                for c in far:
                    T[y, c] += far_share / len(far)
            self.soft_targets = T  # each row sums to 1

    def forward(self, input, target):
        target = target.view(-1)
        logpt = input.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.detach().exp()

        if self.soft_targets is not None:
            q = self.soft_targets.to(input.device)[target]          # (N, C)
            nll = -(q * input).sum(dim=1)
        else:
            nll = -logpt

        loss = nll
        if self.gamma > 0:
            loss = (1.0 - pt).clamp(min=0.0) ** self.gamma * loss
        if self.alpha is not None:
            loss = self.alpha.to(input.device)[target] * loss
        return loss.mean()


def class_counts_from_loader(train_loader, n_classes):
    """
    Count training utterances per emotion class straight from the training loader
    (respects IEMOCAP's train/valid conversation split, which is applied via a sampler).
    """
    ds = train_loader.dataset
    sampler = train_loader.sampler
    idxs = list(sampler.indices) if hasattr(sampler, 'indices') else list(range(len(ds)))
    label_dict = ds.emotion_labels if hasattr(ds, 'emotion_labels') else ds.videoLabels  # MELD / IEMOCAP
    counts = [0] * n_classes
    for i in idxs:
        for y in label_dict[ds.keys[i]]:
            counts[int(y)] += 1
    return counts


def parse_confusable_pairs(spec, target_names):
    """'hap-exc,ang-fru' -> [(0, 4), (3, 5)] using the dataset's class-name order."""
    pairs = []
    for item in [s.strip() for s in spec.split(',') if s.strip()]:
        a, b = [s.strip() for s in item.split('-')]
        if a not in target_names or b not in target_names:
            raise ValueError("unknown class in confusable pair '{}'; valid names: {}".format(item, target_names))
        pairs.append((target_names.index(a), target_names.index(b)))
    return pairs


def smoothing_table_str(loss_f, target_names):
    """Printable L2 target distribution for every class (row = true label, columns = target probability, in %)."""
    T = loss_f.soft_targets
    if T is None:
        return '(no label smoothing)'
    head = '{:<6}'.format('true') + ''.join('{:>7}'.format(n) for n in target_names) + '   row sum'
    rows = [head] + ['{:<6}'.format(target_names[y]) + ''.join('{:>7.2f}'.format(100 * float(v)) for v in T[y]) +
                     '   {:.4f}'.format(float(T[y].sum())) for y in range(len(target_names))]
    return '\n'.join(rows)


def build_training_loss(args, n_classes, target_names, class_weights, train_counts, authors_gamma):
    """
    The ONE place that decides which loss train.py uses (also called by tests/).
      neither --lim1 nor --lim2 : the authors' own `Loss` (alpha = their class weights only if --class_weight)
      --lim2 only               : authors' gamma and class weights + confusion-aware label smoothing
      --lim1 (with/without 2)   : class-balanced alpha (REPLACES the authors' weights, never multiplied with them)
                                  + focal gamma [+ label smoothing]
    `class_weights` is the authors' per-class tensor of this dataset; `train_counts` the training label counts.
    """
    if not (args.lim1 or args.lim2):
        return Loss(alpha=class_weights if args.class_weight else None)
    return ECERCImprovedLoss(
        n_classes=n_classes,
        class_counts=train_counts if args.lim1 else None,
        cb_beta=args.cb_beta,
        gamma=args.focal_gamma if args.lim1 else authors_gamma,
        alpha=None if args.lim1 else (class_weights if args.class_weight else None),
        smoothing=args.ls_eps if args.lim2 else 0.0,
        confusable_pairs=parse_confusable_pairs(args.confusable_pairs, target_names) if args.lim2 else None,
        neighbor_mass=args.ls_neighbor_mass)
