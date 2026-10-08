import torch
import torch.nn as nn
from torch.autograd import Variable


class Loss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
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
                 smoothing=0.0, confusable_pairs=None, neighbor_mass=0.7):
        super(ECERCImprovedLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0 and 0.0 <= neighbor_mass <= 1.0
        self.n_classes = n_classes
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

        # ---- Limitation 1: class-balanced weights from training label counts ----
        self.alpha = None
        if class_counts is not None:
            counts = torch.as_tensor(class_counts, dtype=torch.float).clamp(min=1.0)
            effective_num = (1.0 - torch.pow(cb_beta, counts)) / (1.0 - cb_beta)
            w = 1.0 / effective_num
            self.alpha = w / w.sum() * n_classes

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
        pairs.append((target_names.index(a), target_names.index(b)))
    return pairs
