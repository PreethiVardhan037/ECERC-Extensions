"""
Unit tests for the whole project.   Run from the repository root:

        python tests/run_tests.py            (about 1-2 minutes on a CPU; no dataset and no GPU needed)

References are the AUTHORS' OWN official model.py / loss.py (tests/reference/), not a re-implementation:
  * loss      : with the fixes switched off our loss == the official `Loss`, value and gradient (Issues 8, 9, 10)
  * model     : adaptive_span=False == official model;  adaptive_span=True at initialisation == official model, for short and
                long dialogues with padding (Issue 4);  span gradients / numerics / edge cases (Issues 6, 7)
  * metrics   : ECE / NLL / Brier / per-pair breakdown on hand-checkable inputs
"""
import argparse
import importlib.util
import inspect
import os
import re
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, 'tests', 'reference')
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load(path, name, replace=None):
    src = open(path, encoding='utf-8').read().replace('\r', '')
    for a, b in (replace or []):
        assert a in src, 'patch target not found in ' + path
        src = src.replace(a, b)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    exec(compile(src, path, 'exec'), mod.__dict__)
    sys.modules[name] = mod
    return mod


# the official forward() builds imask/mask only inside `if self.cuda_flag:` (UnboundLocalError on a CPU) -> same one-line fix we ship
CPU_FIX = [("            imask, smask, cmask, mask = identity_mask.cuda(), smask.cuda(), cmask.cuda(), (submask & padmask).cuda()\n",
            "            imask, smask, cmask, mask = identity_mask.cuda(), smask.cuda(), cmask.cuda(), (submask & padmask).cuda()\n"
            "        else:\n            imask, mask = identity_mask, (submask & padmask)\n")]

DS = {
    'MELD': dict(n_classes=7, P=9, d_a=300, names=['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang'], pairs='dis-ang', longest=33,
                 authors_cw=torch.log(torch.FloatTensor([1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.073096002,
                                                         1 / 0.168368836, 1 / 0.026334987, 1 / 0.117230814]))),
    'IEMOCAP': dict(n_classes=6, P=2, d_a=1582, names=['hap', 'sad', 'neu', 'ang', 'exc', 'fru'], pairs='hap-exc,ang-fru', longest=110,
                    authors_cw=torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])),
}
MODS = {}
for ds in DS:
    MODS[ds] = dict(
        model=load(os.path.join(ROOT, ds, 'model.py'), ds + '_model'),
        loss=load(os.path.join(ROOT, ds, 'loss.py'), ds + '_loss'),
        ref_model=load(os.path.join(REF, 'official_%s_model.py' % ds), ds + '_ref_model', CPU_FIX),
        ref_loss=load(os.path.join(REF, 'official_%s_loss.py' % ds), ds + '_ref_loss'),
        train_src=open(os.path.join(ROOT, ds, 'train.py')).read())
EU = load(os.path.join(ROOT, 'MELD', 'eval_utils.py'), 'eval_utils_t')


def loss_args(**kw):
    d = dict(lim1=False, lim2=False, class_weight=False, cb_beta=0.999, focal_gamma=1.0, ls_eps=0.1, ls_neighbor_mass=0.7, confusable_pairs='')
    d.update(kw)
    return argparse.Namespace(**d)


def rand_logp(n, c, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, c, generator=g) * 2
    y = torch.randint(0, c, (n,), generator=g)
    lp = torch.log_softmax(z, 1).clone().requires_grad_(requires_grad)
    return lp, y


# ============================================================================================ L1 / L2 loss
class TestLoss(unittest.TestCase):
    def test_authors_loss_class_is_byte_identical_to_official(self):
        for ds, m in MODS.items():
            ours = open(os.path.join(ROOT, ds, 'loss.py')).read().replace('\r', '')
            theirs = open(os.path.join(REF, 'official_%s_loss.py' % ds)).read().replace('\r', '')
            grab = lambda t: re.search(r'class Loss\(nn\.Module\):.*?(?=\n\n\n|\Z)', t, re.S).group(0).strip()
            self.assertEqual(grab(ours), grab(theirs), ds)

    def test_authors_gamma_constant_in_train_py_matches_official_default(self):
        for ds, m in MODS.items():
            g_train = float(re.search(r'AUTHORS_GAMMA = ([\d.]+)', m['train_src']).group(1))
            g_official = re.search(r'class Loss.*?def __init__\(self, gamma=([\d.]+)', open(os.path.join(REF, 'official_%s_loss.py' % ds)).read(), re.S).group(1)
            self.assertEqual(g_train, float(g_official), ds)

    def test_baseline_loss_equivalence_value_and_gradient(self):
        """new baseline (flags off, and the L2-only path with eps=0) == the official Loss for identical inputs"""
        for ds, m in MODS.items():
            cfg = DS[ds]
            for cw in (False, True):
                lp1, y = rand_logp(97, cfg['n_classes'], 3, True)
                lp2 = lp1.detach().clone().requires_grad_(True)
                official = m['ref_loss'].Loss(alpha=cfg['authors_cw'].clone() if cw else None)
                a = loss_args(class_weight=cw)
                gamma = float(re.search(r'AUTHORS_GAMMA = ([\d.]+)', m['train_src']).group(1))
                variants = {'no flags': m['loss'].build_training_loss(a, cfg['n_classes'], cfg['names'], cfg['authors_cw'].clone(), None, gamma),
                            'L2 eps=0': m['loss'].build_training_loss(loss_args(class_weight=cw, lim2=True, ls_eps=0.0, confusable_pairs=cfg['pairs']),
                                                                      cfg['n_classes'], cfg['names'], cfg['authors_cw'].clone(), [1] * cfg['n_classes'], gamma)}
                lo = official(lp1, y); lo.backward()
                for name, lf in variants.items():
                    lp2.grad = None
                    ln = lf(lp2, y); ln.backward()
                    self.assertAlmostEqual(float(lo), float(ln), places=6, msg='%s %s class_weight=%s' % (ds, name, cw))
                    self.assertTrue(torch.allclose(lp1.grad, lp2.grad, atol=1e-7), '%s %s grad' % (ds, name))

    def test_cb_weights_hand_computed(self):
        L = MODS['MELD']['loss']
        counts, beta = [10, 20, 30], 0.9
        eff = [(1 - beta ** n) / (1 - beta) for n in counts]
        w = np.array([1 / e for e in eff]); w = w / w.sum() * 3
        lf = L.ECERCImprovedLoss(3, class_counts=counts, cb_beta=beta)
        self.assertTrue(np.allclose(lf.alpha.numpy(), w, atol=1e-6))
        self.assertAlmostEqual(float(lf.alpha.sum()), 3.0, places=5)

    def test_cb_weight_applied_exactly_once_with_focal(self):
        """loss_i = alpha[y_i] * (1 - p_t)^gamma * (-log p_t): hand-computed; the authors' IEMOCAP weights are NOT multiplied in"""
        for ds, m in MODS.items():
            cfg = DS[ds]
            counts = list(range(50, 50 + 40 * cfg['n_classes'], 40))
            a = loss_args(lim1=True, class_weight=True, focal_gamma=2.0, cb_beta=0.99)
            lf = m['loss'].build_training_loss(a, cfg['n_classes'], cfg['names'], cfg['authors_cw'].clone(), counts, 0.0)
            lp, y = rand_logp(40, cfg['n_classes'], 5)
            eff = np.array([(1 - 0.99 ** n) / 0.01 for n in counts]); alpha = (1 / eff); alpha = alpha / alpha.sum() * len(counts)
            pt = lp.exp()[torch.arange(40), y].numpy()
            manual = (alpha[y.numpy()] * (1 - pt) ** 2 * (-lp[torch.arange(40), y].numpy())).mean()
            self.assertAlmostEqual(float(lf(lp, y)), float(manual), places=5, msg=ds)
            self.assertTrue(np.allclose(lf.alpha.numpy(), alpha, atol=1e-6), ds)

    def test_cb_edge_cases(self):
        L = MODS['MELD']['loss']
        zero = L.ECERCImprovedLoss(4, class_counts=[100, 0, 50, 10], cb_beta=0.999)
        a = zero.alpha.numpy()
        self.assertEqual(a[1], 0.0)                                    # missing class: no fake huge weight
        self.assertAlmostEqual(float(a[[0, 2, 3]].mean()), 1.0, places=5)   # mean over the PRESENT classes is 1
        self.assertTrue(np.all(np.isfinite(a)))
        flat = L.ECERCImprovedLoss(3, class_counts=[5, 500, 5000], cb_beta=0.0).alpha.numpy()
        self.assertTrue(np.allclose(flat, 1.0))                        # beta = 0 -> effective number 1 for every class
        ext = L.ECERCImprovedLoss(3, class_counts=[5, 500, 5000], cb_beta=0.999999).alpha.numpy()
        self.assertTrue(np.all(np.isfinite(ext)) and ext[0] > ext[1] > ext[2])   # -> inverse frequency, finite
        one = L.ECERCImprovedLoss(3, class_counts=[1, 1, 1], cb_beta=0.99).alpha.numpy()
        self.assertTrue(np.allclose(one, 1.0))
        with self.assertRaises(AssertionError):
            L.ECERCImprovedLoss(3, class_counts=[1, 2, 3], cb_beta=1.0)  # beta = 1 would divide by zero
        lp, y = rand_logp(30, 4, 1)
        self.assertTrue(torch.isfinite(zero(lp, y)))

    def test_label_mapping_per_dataset(self):
        for ds, m in MODS.items():
            names = DS[ds]['names']
            got = m['loss'].parse_confusable_pairs(DS[ds]['pairs'], names)
            want = [(names.index(a), names.index(b)) for a, b in (p.split('-') for p in DS[ds]['pairs'].split(','))]
            self.assertEqual(got, want)
        self.assertEqual(MODS['IEMOCAP']['loss'].parse_confusable_pairs('hap-exc,ang-fru', DS['IEMOCAP']['names']), [(0, 4), (3, 5)])
        self.assertEqual(MODS['MELD']['loss'].parse_confusable_pairs('dis-ang', DS['MELD']['names']), [(5, 6)])
        with self.assertRaises(ValueError):
            MODS['MELD']['loss'].parse_confusable_pairs('hap-exc', DS['MELD']['names'])   # IEMOCAP names are invalid for MELD

    def test_smoothing_distribution_every_class_both_datasets(self):
        eps, nm = 0.1, 0.7
        for ds, m in MODS.items():
            cfg, C = DS[ds], DS[ds]['n_classes']
            pairs = m['loss'].parse_confusable_pairs(cfg['pairs'], cfg['names'])
            lf = m['loss'].ECERCImprovedLoss(C, smoothing=eps, confusable_pairs=pairs, neighbor_mass=nm)
            T = lf.soft_targets.numpy()
            print('\n[%s] smoothing target distribution (%%), eps=%.2f neighbor_mass=%.2f\n%s' % (ds, eps, nm, m['loss'].smoothing_table_str(lf, cfg['names'])))
            partners = {c: set() for c in range(C)}
            for a, b in pairs:
                partners[a].add(b); partners[b].add(a)
            for y in range(C):
                self.assertAlmostEqual(float(T[y].sum()), 1.0, places=6)               # sums to 1
                self.assertAlmostEqual(float(T[y, y]), 1 - eps, places=6)              # target class gets 1 - eps
                far = [c for c in range(C) if c != y and c not in partners[y]]
                if partners[y]:
                    for c in partners[y]:
                        self.assertAlmostEqual(float(T[y, c]), eps * nm / len(partners[y]), places=6)
                    for c in far:
                        self.assertAlmostEqual(float(T[y, c]), eps * (1 - nm) / len(far), places=6)
                else:                                                                   # class without a partner: uniform smoothing
                    for c in far:
                        self.assertAlmostEqual(float(T[y, c]), eps / len(far), places=6)
            # mass is not duplicated when a pair is listed twice / in both orders
            lf2 = m['loss'].ECERCImprovedLoss(C, smoothing=eps, confusable_pairs=pairs + [(b, a) for a, b in pairs] + pairs, neighbor_mass=nm)
            self.assertTrue(np.allclose(lf2.soft_targets.numpy(), T))

    def test_smoothing_when_every_other_class_is_a_partner(self):
        L = MODS['MELD']['loss']
        T = L.ECERCImprovedLoss(3, smoothing=0.2, confusable_pairs=[(0, 1), (0, 2)], neighbor_mass=0.7).soft_targets.numpy()
        self.assertTrue(np.allclose(T[0], [0.8, 0.1, 0.1]))

    def test_smoothing_target_is_used_in_the_loss(self):
        L = MODS['IEMOCAP']['loss']
        lf = L.ECERCImprovedLoss(6, smoothing=0.1, confusable_pairs=[(0, 4)], neighbor_mass=0.7)
        lp, y = rand_logp(25, 6, 2)
        manual = -(lf.soft_targets[y] * lp).sum(1).mean()
        self.assertAlmostEqual(float(lf(lp, y)), float(manual), places=6)


# ============================================================================================ model / L4
def make_model(ds, adaptive, span_max=None, seed=0, per_head=True):
    cfg, mm = DS[ds], MODS[ds]
    D = 1024 + cfg['d_a'] + 342
    kw = dict(d_t=1024, d_a=cfg['d_a'], d_v=342, base_layer=1, input_size=D, hidden_size=128, n_speakers=cfg['P'], n_classes=cfg['n_classes'], cuda_flag=False)
    torch.manual_seed(seed)
    ref = mm['ref_model'].ECERC(argparse.Namespace(), **kw)
    args = argparse.Namespace(adaptive_span=adaptive, span_max=span_max or 40.0, span_ramp=2.0, span_init=1.0, span_per_head=per_head)
    ours = mm['model'].ECERC(args, **kw)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('span_modules.') for k in missing), (missing, unexpected)
    return ref.eval(), ours.eval()


def make_batch(ds, lengths, seed=0):
    cfg, g = DS[ds], torch.Generator().manual_seed(seed)
    B, L, D = len(lengths), max(lengths), 1024 + DS[ds]['d_a'] + 342
    emo, sem = torch.randn(L, B, D, generator=g), torch.randn(L, B, 1024, generator=g)
    sp = torch.randint(0, min(cfg['P'], 4), (L, B), generator=g)
    qm = torch.nn.functional.one_hot(sp, cfg['P']).float()
    um = torch.zeros(B, L)
    for b, n in enumerate(lengths):
        um[b, :n] = 1
    return emo, sem, qm, um, list(lengths)


class TestModelAndSpan(unittest.TestCase):
    def test_adaptive_off_is_the_official_model(self):
        for ds in DS:
            ref, ours = make_model(ds, adaptive=False)
            self.assertEqual(list(ref.state_dict().keys()), list(ours.state_dict().keys()))
            batch = make_batch(ds, [7, 4, 9], 1)
            with torch.no_grad():
                self.assertTrue(torch.equal(ref(*batch), ours(*batch)), ds)

    def test_full_history_initialisation_equals_official_short_and_long(self):
        for ds in DS:
            for lengths in ([1], [2, 1], [5, 3, 5], [DS[ds]['longest'], 10, 1]):
                ref, ours = make_model(ds, adaptive=True, span_max=max(lengths))   # train.py sets span_max = longest dialogue
                batch = make_batch(ds, lengths, 7)
                with torch.no_grad():
                    a, b = ref(*batch), ours(*batch)
                self.assertTrue(torch.allclose(a, b, atol=1e-6), '%s lengths=%s max|diff|=%g' % (ds, lengths, float((a - b).abs().max())))

    def test_span_max_smaller_than_dialogue_is_NOT_full_history(self):
        """documents why train.py refuses span_max < longest dialogue - 1: such a model is not the baseline at initialisation"""
        ref, ours = make_model('MELD', adaptive=True, span_max=4)
        batch = make_batch('MELD', [20, 20], 2)
        with torch.no_grad():
            self.assertFalse(torch.allclose(ref(*batch), ours(*batch), atol=1e-4))

    def test_train_py_refuses_too_small_span_max_and_defaults_to_auto(self):
        for ds, m in MODS.items():
            self.assertIn("parser.add_argument('--span_max', type=float, default=0", m['train_src'])
            self.assertIn('elif args.span_max < max_len - 1:', m['train_src'])

    def test_no_nan_and_finite_for_very_short_dialogues_with_spans(self):
        for ds in DS:
            _, ours = make_model(ds, adaptive=True, span_max=3)
            ours.train()
            for lengths in ([1], [1, 1], [2], [3, 1]):
                out = ours(*make_batch(ds, lengths, 3))
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                for p in ours.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())

    def _attn_and_mask(self, L=12, B=2, H=8, lengths=(12, 7), seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.arange(L)
        sub = (idx.view(L, 1) >= idx.view(1, L))
        pad = torch.zeros(B, 1, L, dtype=torch.bool)
        for b, n in enumerate(lengths):
            pad[b, 0, :n] = True
        spk = torch.randint(0, 2, (B, L), generator=g)
        spk[0] = torch.tensor([0] * (L - 1) + [1])        # utterance 0..L-2 are one speaker -> their rows (also LATE ones, key distances
        #                                                   far beyond a small span) have NO allowed cross-speaker key
        cross = (spk.unsqueeze(2) != spk.unsqueeze(1))
        mask = (sub.unsqueeze(0) & pad & cross).unsqueeze(1)             # [B,1,L,L], like cmask: some rows have NO allowed key
        scores = torch.randn(B, H, L, L, generator=g).masked_fill(~mask, -1e9)
        return torch.softmax(scores, -1), mask

    def test_adaptive_span_module_properties(self):
        for ds in DS:      # both datasets' model.py
            self._test_adaptive_span_module_properties(MODS[ds]['model'], ds)

    def _test_adaptive_span_module_properties(self, Mod, ds):
        attn, mask = self._attn_and_mask()
        valid = mask.any(-1, keepdim=True)
        self.assertTrue((~valid).any() and valid.any())
        self.assertTrue((~valid[0, 0, 8:11]).all())          # a late fully-masked row exists (distance 8..10 > span + ramp)
        full = Mod.AdaptiveSpan(8, max_span=12.0, per_head=True)
        self.assertTrue(torch.equal(full(attn, valid), attn))             # full history: EXACT identity
        sp = Mod.AdaptiveSpan(8, max_span=12.0, per_head=True)
        sp.span_ratio.data.fill_(2.0 / 12.0)                              # span 2
        out = sp(attn, valid)
        nv = (~valid).expand_as(out)
        self.assertTrue(torch.equal(out[nv], attn[nv]))                   # rows with no allowed key: untouched
        vr = valid.expand_as(out)
        self.assertEqual(float((out * (~mask).float() * vr.float()).sum()), 0.0)   # rows with allowed keys: no attention on padded / future / same-speaker keys
        rs = out.sum(-1)[valid.expand_as(out[..., :1]).squeeze(-1)]
        self.assertTrue(((rs <= 1 + 1e-6) & (rs >= 0)).all())
        far = (torch.arange(12).view(12, 1) - torch.arange(12).view(1, 12)) > 2 + 2.0   # beyond span + ramp
        self.assertEqual(float((out * far.float() * vr.float()).sum()), 0.0)   # rows with allowed keys: keys beyond span + ramp removed
        self.assertTrue(torch.isfinite(out).all())

    def test_span_gradients_boundary_collapse_saturation(self):
        for ds in DS:      # both datasets' model.py
            self._test_span_gradients_boundary_collapse_saturation(MODS[ds]['model'], ds)

    def _test_span_gradients_boundary_collapse_saturation(self, Mod, ds):
        attn, mask = self._attn_and_mask(L=14, lengths=(14, 9), seed=4)
        valid = mask.any(-1, keepdim=True)
        v = torch.randn(2, 8, 14, 5, generator=torch.Generator().manual_seed(9))
        def task_grad(ratio):
            m = Mod.AdaptiveSpan(8, max_span=14.0, per_head=True)
            m.span_ratio.data.fill_(ratio)
            (m(attn, valid) @ v).pow(2).sum().backward()
            return m.span_ratio.grad
        # 1) at initialisation (span >= every distance) the task loss cannot move the span -> only the penalty does
        self.assertEqual(float(task_grad(1.0).abs().sum()), 0.0)
        # 2) when the span cuts into the attention the gradient exists and is finite
        g = task_grad(4.0 / 14.0)
        self.assertTrue(torch.isfinite(g).all() and float(g.abs().sum()) > 0)
        # 3) exactly at span boundaries / inside the ramp / collapsed (0) / saturated: finite everywhere
        for r in (0.0, 1e-6, 1.0 / 14, 3.0 / 14, 3.5 / 14, 13.0 / 14, 1.0):
            gr = task_grad(r)
            self.assertTrue(torch.isfinite(gr).all(), 'ratio %g' % r)
        # 4) span 0: rows keep (almost) nothing -> bounded output (zero vector), no division blow-up
        m0 = Mod.AdaptiveSpan(8, max_span=14.0, per_head=True)
        m0.span_ratio.data.fill_(0.0)
        out0 = m0(attn, valid)
        self.assertTrue(torch.isfinite(out0).all() and float(out0.sum(-1).max()) <= 1.0 + 1e-6)
        # 5) the penalty is differentiable at the starting value 1.0 (raw, un-clamped ratio)
        model = make_model(ds, adaptive=True, span_max=10)[1]
        model.span_penalty().backward()
        act = [k for k in model.span_modules if k != 'self_event']
        self.assertTrue(all(float(model.span_modules[k].span_ratio.grad.abs().sum()) > 0 for k in act))

    def test_float16_attention_row_is_finite(self):
        for ds in DS:      # both datasets' model.py
            self._test_float16_attention_row_is_finite(MODS[ds]['model'], ds)

    def _test_float16_attention_row_is_finite(self, Mod, ds):
        attn, mask = self._attn_and_mask(seed=5)
        m = Mod.AdaptiveSpan(8, max_span=12.0, per_head=True)
        m.span_ratio.data.fill_(0.1)
        out = m(attn.half().float(), mask.any(-1, keepdim=True)).half()      # emulate fp16 storage of the attention weights
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_self_event_branch_is_provably_inactive(self):
        for ds in DS:
            _, ours = make_model(ds, adaptive=True, span_max=12)
            batch = make_batch(ds, [10, 6], 11)
            with torch.no_grad():
                base = ours(*batch)
                ours.span_modules['self_event'].span_ratio.data.fill_(0.0)     # even a collapsed span changes nothing
                self.assertTrue(torch.equal(base, ours(*batch)), ds)
            ours.span_modules['self_event'].span_ratio.data.fill_(1.0)
            ours.train(); ours.zero_grad()
            ours(*batch).sum().backward()
            self.assertEqual(float(ours.span_modules['self_event'].span_ratio.grad.abs().sum()), 0.0)
            rep = ours.span_report()
            self.assertTrue(rep['self_event']['status'].startswith('inactive'))
            self.assertTrue(all(rep[k]['status'].startswith('active') for k in ('cross_event', 'contagion', 'cross_emotion')))
            self.assertNotIn('self_event', [k for k in ours.span_modules if k not in ('self_event',) and False])
            self.assertTrue(np.isfinite(ours.mean_span()))

    def test_active_branches_do_change_when_the_span_is_cut(self):
        _, ours = make_model('MELD', adaptive=True, span_max=12)
        batch = make_batch('MELD', [12, 12], 13)
        with torch.no_grad():
            base = ours(*batch)
            for k in ('cross_event', 'contagion', 'cross_emotion'):
                ours.span_modules[k].span_ratio.data.fill_(0.0)
            self.assertFalse(torch.allclose(base, ours(*batch), atol=1e-6))


# ============================================================================================ metrics
class TestMetrics(unittest.TestCase):
    def test_ece_known_values(self):
        self.assertAlmostEqual(EU.ece_score([1.0] * 10, [1] * 10), 0.0)
        self.assertAlmostEqual(EU.ece_score([0.9] * 10, [1] * 9 + [0]), 0.0, places=9)        # confidence == accuracy
        self.assertAlmostEqual(EU.ece_score([0.95] * 4, [1, 0, 0, 0]), 0.7, places=9)           # one bin, |0.25-0.95|
        self.assertTrue(np.isnan(EU.ece_score([], [])))

    def test_nll_brier_perfect_and_uniform(self):
        y = np.array([0, 1, 2])
        self.assertAlmostEqual(EU.brier_score(np.eye(3)[y], y), 0.0)
        self.assertAlmostEqual(EU.nll_score(np.full((3, 3), 1 / 3), y), np.log(3), places=9)
        self.assertAlmostEqual(EU.brier_score(np.full((3, 3), 1 / 3), y), (2 / 3) ** 2 + 2 * (1 / 3) ** 2, places=9)

    def test_full_metrics_consistency(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 6, 400); z = rng.normal(size=(400, 6)); z[np.arange(400), y] += 1.2
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        m = EU.full_metrics(p, y, DS['IEMOCAP']['names'])
        cm = np.array(m['confusion_matrix'])
        self.assertEqual(int(cm.sum()), 400)
        self.assertAlmostEqual(m['accuracy'], 100 * np.trace(cm) / 400, places=9)
        sup = np.array([m['per_class'][n]['support'] for n in DS['IEMOCAP']['names']])
        self.assertTrue(np.array_equal(sup, cm.sum(1)))
        self.assertAlmostEqual(m['macro_f1'], np.mean([m['per_class'][n]['f1'] for n in DS['IEMOCAP']['names']]), places=9)
        for k in ('ece', 'nll', 'brier', 'weighted_precision', 'weighted_recall', 'macro_precision', 'macro_recall'):
            self.assertIn(k, m)

    def test_pair_breakdown_populations_and_counts(self):
        y = np.array([0, 0, 4, 4, 4, 1, 2])
        pred = np.array([0, 4, 4, 0, 4, 1, 0])
        p = np.full((7, 6), 0.02); p[np.arange(7), pred] = 0.9; p /= p.sum(1, keepdims=True)
        out = EU.pair_breakdown(p, y, [(0, 4)], DS['IEMOCAP']['names'])['hap-exc']
        self.assertEqual(out['true_in_pair']['n'], 5)       # true hap or exc
        self.assertEqual(out['pred_in_pair']['n'], 6)       # predicted hap or exc (incl. the sad->hap error)
        self.assertEqual(out['confusions']['hap->exc'], 1)
        self.assertEqual(out['confusions']['exc->hap'], 1)
        self.assertAlmostEqual(out['confusions']['rate_of_true_in_pair'], 100 * 2 / 5)


# ============================================================================================ pipeline reliability
class TestPipelineReliability(unittest.TestCase):
    """A failed / invalid calibration can never be mistaken for a result (auto_tune.Runner)."""

    def setUp(self):
        import tempfile
        sys.path.insert(0, ROOT)
        import auto_tune
        self.at = auto_tune
        self.tmp = tempfile.mkdtemp(prefix='ecerc_rel_')
        self.args = argparse.Namespace(resume=False, calib_retries=1, calib_args='', extra='')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_calibration_is_recorded_and_stale_output_is_not_reused(self):
        R = self.at.Runner('MELD', self.args, out_root=self.tmp)
        out_dir = os.path.join(R.out, 'calib', 'fake_s0')
        os.makedirs(out_dir)
        stale = os.path.join(out_dir, 'calibration_MELD.json')
        open(stale, 'w').write('{"T": 1.0, "stale": true}')                       # leftover from an earlier run
        run = {'stem': 'fake_s0', 'ckpt': os.path.join(self.tmp, 'does_not_exist.pkl'), 'seed': 0}
        self.assertIsNone(R.calibrate(run))                                       # the checkpoint does not exist -> must FAIL
        self.assertFalse(os.path.exists(stale))                                   # the stale file was removed, not read
        self.assertEqual(len(R.failures), 1)
        f = R.failures[0]
        self.assertEqual((f['stage'], f['run'], f['attempts']), ('calibration', 'fake_s0', 2))   # 1 try + 1 retry
        self.assertTrue(f['reason'])
        self.assertTrue(os.path.exists(os.path.join(R.out, f['log'])))

    def test_validation_helpers_flag_incomplete_results(self):
        self.assertTrue(self.at.calibration_problems({'T': 1.0}))
        self.assertTrue(self.at.calibration_problems('nonsense'))
        full = {'T': 1, 'T_mc': 1, 'protocol': {}, 'test_full': {v: {} for v in self.at.CAL_VARIANTS}, 'test_pairs': {v: {} for v in self.at.CAL_VARIANTS}}
        self.assertEqual(self.at.calibration_problems(full), [])
        del full['test_pairs']['MC-Dropout']
        self.assertTrue(self.at.calibration_problems(full))
        self.assertTrue(self.at.run_problems({'seed': 0}))
        self.assertTrue(self.at.run_problems({'seed': 0, 'ckpt': 'x', 'best_epoch': 0, 'valid': {}, 'test': {}, 'protocol': {}, 'flags': {}}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
