"""
Writes SMALL SYNTHETIC datasets in the exact pickle layout the dataloaders expect, so train.py / calibrate.py /
auto_tune.py can be smoke-tested without the real IEMOCAP/MELD features.  Numbers obtained on this data mean nothing.

    python tests/make_synthetic_data.py --root /tmp/proj          # writes /tmp/proj/data/{meld,iemocap}/...
"""
import argparse
import os
import pickle
import numpy as np


def _feats(rng, labels, dim, strength=1.5, n_classes=7):
    x = rng.normal(size=(len(labels), dim)).astype(np.float32)
    x[np.arange(len(labels)), np.asarray(labels) % dim] += strength        # weak class signal so training can learn something
    return x


def make_meld(root, seed=0, n_train=30, n_valid=8, n_test=10):
    rng = np.random.default_rng(seed)
    ids = {'train': list(range(n_train)), 'valid': list(range(1000, 1000 + n_valid)), 'test': list(range(2000, 2000 + n_test))}
    speakers, emo, sent, e1, s1, aud, vis, sentences = {}, {}, {}, {}, {}, {}, {}, {}
    for split, keys in ids.items():
        for k in keys:
            L = int(rng.integers(3, 14)) if k != 0 else 33         # one 33-utterance dialogue (MELD's longest)
            n_sp = int(rng.integers(2, 5))
            sp = rng.integers(0, n_sp, size=L)
            lab = [int(x) for x in rng.integers(0, 7, size=L)]
            speakers[k] = np.eye(9)[sp].astype(np.float32); emo[k] = lab; sent[k] = [0] * L
            e1[k] = _feats(rng, lab, 1024); s1[k] = _feats(rng, lab, 1024)
            vis[k] = _feats(rng, lab, 300)      # NB: in the authors' MELD pickle 'videoVisual' is the 300-d block
            aud[k] = _feats(rng, lab, 342)      # and 'videoAudio' the 342-d block (the model slices 300 then 342)
            sentences[k] = ['x'] * L
    blob = (speakers, emo, sent, e1, e1, e1, e1, s1, s1, s1, s1, aud, vis, sentences, ids['train'], ids['test'], ids['valid'])
    os.makedirs(os.path.join(root, 'data', 'meld'), exist_ok=True)
    pickle.dump(blob, open(os.path.join(root, 'data', 'meld', 'meld_emotion_semantic_features_roberta.pkl'), 'wb'))


def make_iemocap(root, seed=1, n_train=30, n_test=8):
    rng = np.random.default_rng(seed)
    train = ['Ses0%d_%03d' % (1 + i % 4, i) for i in range(n_train)]
    test = ['Ses05_%03d' % i for i in range(n_test)]
    spk, lab, txt, aud, vis, sent, r1, s1 = {}, {}, {}, {}, {}, {}, {}, {}
    for k in train + test:
        L = int(rng.integers(5, 41))
        y = [int(x) for x in rng.integers(0, 6, size=L)]
        spk[k] = list(rng.choice(['M', 'F'], size=L)); lab[k] = y; txt[k] = None
        aud[k] = _feats(rng, y, 1582); vis[k] = _feats(rng, y, 342); sent[k] = ['x'] * L
        r1[k] = _feats(rng, y, 1024); s1[k] = _feats(rng, y, 1024)
    d = os.path.join(root, 'data', 'iemocap'); os.makedirs(d, exist_ok=True)
    pickle.dump((list(spk), spk, lab, txt, aud, vis, sent, train, test), open(os.path.join(d, 'IEMOCAP_features.pkl'), 'wb'))
    pickle.dump((None, None, r1, r1, r1, r1, None, None, None, None), open(os.path.join(d, 'iemocap_emotion_features_roberta.pkl'), 'wb'))
    pickle.dump((None,) * 6 + (s1, s1, s1, s1) + (None,) * 6, open(os.path.join(d, 'iemocap_emotion_semantic_features_roberta.pkl'), 'wb'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); a = ap.parse_args()
    make_meld(a.root); make_iemocap(a.root)
    print('synthetic data written under', os.path.join(a.root, 'data'))
