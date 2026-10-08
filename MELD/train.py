import argparse
import os
import re
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import MELDRobertaDataset
from model import ECERC
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from loss import Loss
# [ADDED] Limitations 1 & 2 (loss-level fixes)
from loss import ECERCImprovedLoss, class_counts_from_loader, parse_confusable_pairs

import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_MELD_bert_loaders(path, batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaDataset('train')
    validset = MELDRobertaDataset('valid')
    testset = MELDRobertaDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True  # shuffle for training data
                              )

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None):
    assert not train_flag or optimizer != None
    losses, preds, labels, masks = [], [], [], []

    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()
        # roberta_fea: CLS embedding of last hidden layer in RoBERTa
        emo_roberta, sem_roberta, audio, vision, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]

        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.feature_type == "multi":
            emo_roberta = torch.cat([emo_roberta,audio, vision], dim=-1)
            sem_roberta = torch.cat([sem_roberta], dim=-1)

        log_prob = model(emo_roberta,sem_roberta, qmask, umask, seq_lengths)

        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        losses.append(loss.item())

        if train_flag:
            if args.lim4:  # [ADDED] L4: L1 penalty on the learned spans (logged loss above excludes it)
                loss = loss + args.span_penalty * model.span_penalty()
            loss.backward()
            optimizer.step()
            if args.lim4:  # [ADDED] L4: keep span_ratio in [0, 1]
                model.clamp_spans()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(
        metrics.classification_report(labels, preds, target_names=target_names if target_names else None, digits=4))
    all_matrix.append(metrics.confusion_matrix(labels, preds))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, []


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_type', type=str, default='multi', help='feature type: text/multi')

    parser.add_argument('--data_dir', type=str, default='../data/meld/meld_features_roberta.pkl', help='dataset dir: meld_features_roberta.pkl')

    parser.add_argument('--output_dir', type=str, default='.', help='saved model dir')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers, 1/2')

    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--use_valid_flag', action='store_true', default=True, help='use valid set')

    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate: 0.0005')

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--cls_type', type=str, default='emotion', help='choose between sentiment or emotion')

    parser.add_argument('--Dataset', type=str, default="MELD", help='DATASET')


    # ===================== [ADDED] flags for Limitations 1 & 2 (+ checkpointing) =====================
    # Without --lim1 / --lim2 the authors' original loss is used, i.e. behaviour is unchanged.
    parser.add_argument('--lim1', action='store_true', default=False,
                        help='Limitation 1: Class-Balanced weights (effective number of samples) + Focal loss')
    parser.add_argument('--cb_beta', type=float, default=0.999, help='L1: beta of the effective-number-of-samples weights (lower = flatter weights)')
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='L1: focal-loss gamma (used when --lim1 is set; the authors use 1 on MELD)')
    parser.add_argument('--lim2', action='store_true', default=False,
                        help='Limitation 2: confusion-aware label smoothing')
    parser.add_argument('--ls_eps', type=float, default=0.1, help='L2: total label-smoothing mass')
    parser.add_argument('--ls_neighbor_mass', type=float, default=0.7,
                        help='L2: fraction of the smoothing mass routed to the confusable partner class(es)')
    parser.add_argument('--confusable_pairs', type=str, default='dis-ang',
                        help="L2: confusable class pairs from the paper's Appendix C, e.g. 'hap-exc,ang-fru'")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default = the authors\' seed)')
    parser.add_argument('--save_model_path', type=str, default=None,
                        help='where to save the best-validation-F1 checkpoint (default: <output_dir>/ECERC_trained.pkl)')
    # ===================================================================================================


    # ===================== [ADDED] flags for Limitation 4 (adaptive attention span) =====================
    parser.add_argument('--lim4', action='store_true', default=False,
                        help='Limitation 4 (--lim4): learnable attention span in the 4 Evidence-Cause Interaction branches')
    parser.add_argument('--span_max', type=float, default=40,
                        help='L4: largest span (utterances) a branch may learn; must be >= the longest dialogue (MELD max 33, IEMOCAP max ~110)')
    parser.add_argument('--span_ramp', type=float, default=2.0, help='L4: length (utterances) of the soft ramp at the span edge')
    parser.add_argument('--span_init', type=float, default=1.0, help='L4: initial span as a fraction of span_max (1.0 = start from full history)')
    parser.add_argument('--span_branch_level', action='store_true', default=False,
                        help='L4: ONE span per branch shared by its 8 heads (default is one span per attention head, as in Sukhbaatar et al.)')
    parser.add_argument('--span_per_head', action='store_true', default=False, help=argparse.SUPPRESS)  # deprecated: per-head is the default now
    parser.add_argument('--span_penalty', type=float, default=0.005, help='L4: weight of the L1 penalty on the (normalised) span')
    parser.add_argument('--span_lr', type=float, default=0.005, help='L4: learning rate of the span parameters (they need a larger lr than the network)')
    # ====================================================================================================

    args = parser.parse_args()
    args.adaptive_span = args.lim4  # [ADDED] read by ECERC.__init__
    args.span_per_head = not args.span_branch_level  # [ADDED] L4: per-head spans unless --span_branch_level

    epochs, batch_size, data_path, base_layer, feature_type = \
        args.epochs, args.batch_size, args.data_dir, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda

    # MELD dataset
    n_speakers, hidden_size, input_size = 9, 128, None
    if args.cls_type.strip().lower() == 'emotion':
        n_classes = 7
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']

        # perform 1/weight according to the weight of each label in training data
        class_weights = torch.FloatTensor(
            [1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.073096002, 1 / 0.168368836, 1 / 0.026334987, 1 / 0.117230814])
        class_weights = torch.log(class_weights)
    else:
        n_classes = 3
        target_names = ['0', '1', '2']
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0])

    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    if feature_type == 'text':
        input_size = D_text
    elif feature_type == "multi":
        input_size = D_text + D_audio + D_visual
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(args.seed)  # [CHANGED] was seed_everything(); default value is identical
    model = ECERC(args, d_t=D_text, d_a=D_audio, d_v=D_visual,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        cuda_flag=cuda_flag)

    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'ECERC'
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))

    train_loader, valid_loader, test_loader = get_MELD_bert_loaders(data_path, batch_size=batch_size, classify=args.cls_type, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.lim4:  # [ADDED] L4: span parameters get their own lr and no weight decay
        span_params = [p for n, p in model.named_parameters() if n.startswith('span_modules.')]
        other_params = [p for n, p in model.named_parameters() if not n.startswith('span_modules.')]
        optimizer = optim.Adam([{'params': other_params, 'lr': args.lr, 'weight_decay': args.l2},
                                {'params': span_params, 'lr': args.span_lr, 'weight_decay': 0.0}])
    loss_f = Loss(alpha=class_weights if args.class_weight else None)

    # ===================== [ADDED] Limitations 1 & 2: loss selection =====================
    if args.lim1 or args.lim2:
        train_counts = class_counts_from_loader(train_loader, n_classes)
        print('Training label counts per class {}: {}'.format(target_names, train_counts))
        loss_f = ECERCImprovedLoss(
            n_classes=n_classes,
            class_counts=train_counts if args.lim1 else None,
            cb_beta=args.cb_beta,
            gamma=args.focal_gamma if args.lim1 else 0.0,
            smoothing=args.ls_eps if args.lim2 else 0.0,
            confusable_pairs=parse_confusable_pairs(args.confusable_pairs, target_names) if args.lim2 else None,
            neighbor_mass=args.ls_neighbor_mass)
        print('Using ECERCImprovedLoss | L1 (class-balanced + focal): {} | L2 (confusion-aware LS): {}'.format(args.lim1, args.lim2))
        if args.lim1:
            print('  class-balanced alpha: {}'.format([round(a, 3) for a in loss_f.alpha.tolist()]))
    else:
        print('Using the authors\' original loss')
    save_model_path = args.save_model_path or os.path.join(args.output_dir, 'ECERC_trained.pkl')
    best_test_metrics = None
    best_spans, best_mean_span, best_spans_full = None, None, None
    best_valid_macro = None
    if args.lim4:
        print('Using L4 adaptive attention span | span_max {} ramp {} init {} per_head {} penalty {} span_lr {}'.format(
            args.span_max, args.span_ramp, args.span_init, args.span_per_head, args.span_penalty, args.span_lr))
        print('  (the self-event branch uses the identity mask, distance 0, so its span has no effect and is not reported)')
    # ======================================================================================

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    for e in range(epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, train_flag=True,
                                                                        optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                        target_names=target_names)
        valid_loss, valid_acc, valid_fscore, valid_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader,
                                                                        cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
        test_loss, test_acc, test_fscore, test_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
        all_test_fscore.append(test_fscore)
        all_test_acc.append(test_acc)

        if args.use_valid_flag:
            eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
        else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
            torch.save(model.state_dict(), save_model_path)  # [ADDED] checkpoint for calibrate.py / inference.py
            best_test_metrics = test_metrics                  # [ADDED] per-class report + confusion matrix
            _m = re.search(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', valid_metrics[0])
            best_valid_macro = float(_m.group(1)) * 100 if _m else None   # [ADDED] validation macro-F1 (selection only, never test)
            if args.lim4:
                best_spans, best_mean_span, best_spans_full = model.get_spans_str(), model.mean_span(), model.get_spans()
        else:
            patience += 1
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
            else:
                patience2 += 1

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                       round(time.time() - start_time, 2)))
        # print(test_metrics[0])
        # print(test_metrics[1])
        # print('\n')

        if args.lim4:  # [ADDED] L4: learned spans (utterances) after this epoch
            print('   learned spans (utterances): {} | mean {:.2f}'.format(model.get_spans_str(), model.mean_span()))

        if patience >= args.patience and patience2 >= args.patience:
            print('Early stoping...', patience, patience2)
            break

    print('Final Test performance...')
    print('Early stoping...', patience, patience2)

    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,
                                                                                                all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                                                                all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
    print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch2,
                                                                            all_test_acc[best_epoch2] if best_epoch2 >= 0 else 0,
                                                                            all_test_fscore[best_epoch2] if best_epoch2 >= 0 else 0))

    # ===================== [ADDED] per-class F1 + confusion matrix at the best-validation epoch =====================
    if best_test_metrics is not None:
        print('\nPer-class test report at best-validation-F1 epoch ({}):'.format(best_epoch))
        print(best_test_metrics[0])
        print(best_test_metrics[1])
        print('Validation scores at best-validation epoch: weighted-F1 {:.2f} | macro-F1 {}'.format(
            best_eval_fscore, '{:.2f}'.format(best_valid_macro) if best_valid_macro is not None else 'n/a'))
        print('Checkpoint saved to: {}'.format(save_model_path))
        if args.lim4:
            print('Learned spans at best-validation epoch (utterances): {} | mean {:.2f}'.format(best_spans, best_mean_span))
            for _k, _v in best_spans_full.items():
                if _k != 'self_event' and len(_v) > 1:
                    print('   per-head spans {}: {}'.format(_k, ' '.join('{:.1f}'.format(x) for x in _v)))
