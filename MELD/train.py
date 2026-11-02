import argparse
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from dataloader import MELDRobertaDataset
from model import ECERC
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
# [ADDED] Limitations 1 & 2 (loss-level fixes), shared metric code, structured results
from loss import class_counts_from_loader, build_training_loss, smoothing_table_str
from eval_utils import full_metrics, dump_json

import warnings
warnings.filterwarnings("ignore")

DATASET = 'MELD'


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


def train_or_eval_model(model, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False,
                        feature_type='multi', lim4=False, span_penalty=0.0, keep_probs=False):
    """
    [CHANGED] no global `args`; returns a dict. `log_probs` (only with keep_probs=True) are the model's per-utterance
    log-probabilities, from which every metric (accuracy, F1, NLL, Brier, ECE, per-class P/R/F1, confusion matrix)
    is derived in eval_utils.full_metrics -- nothing has to be parsed back out of terminal output.
    """
    assert not train_flag or optimizer is not None
    losses, preds, labels, log_probs = [], [], [], []

    if train_flag:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train_flag):
        for data in dataloader:
            if train_flag:
                optimizer.zero_grad()
            # roberta_fea: CLS embedding of last hidden layer in RoBERTa
            emo_roberta, sem_roberta, audio, vision, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]

            seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            if feature_type == "multi":
                emo_roberta = torch.cat([emo_roberta, audio, vision], dim=-1)
                sem_roberta = torch.cat([sem_roberta], dim=-1)

            log_prob = model(emo_roberta, sem_roberta, qmask, umask, seq_lengths)

            label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
            loss = loss_f(log_prob, label)

            preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
            labels.append(label.data.cpu().numpy())
            if keep_probs:
                log_probs.append(log_prob.detach().cpu().numpy())

            losses.append(loss.item())

            if train_flag:
                if lim4:  # [ADDED] L4: L1 penalty on the learned spans (logged loss above excludes it)
                    loss = loss + span_penalty * model.span_penalty()
                loss.backward()
                optimizer.step()
                if lim4:  # [ADDED] L4: keep span_ratio in [0, 1]
                    model.clamp_spans()

    if not preds:
        return {'loss': float('nan'), 'acc': float('nan'), 'wf1': float('nan'), 'labels': np.array([]), 'preds': np.array([]), 'log_probs': None}

    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    return {'loss': round(np.sum(losses) / len(losses), 4),
            'acc': round(accuracy_score(labels, preds) * 100, 2),
            'wf1': round(f1_score(labels, preds, average='weighted') * 100, 2),
            'labels': labels, 'preds': preds,
            'log_probs': np.concatenate(log_probs) if keep_probs else None}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_type', type=str, default='multi', help='feature type: text/multi')

    parser.add_argument('--data_dir', type=str, default='../data/meld/meld_features_roberta.pkl', help='dataset dir: meld_features_roberta.pkl')

    parser.add_argument('--output_dir', type=str, default='.', help='saved model dir')

    parser.add_argument('--base_layer', type=int, default=1, help='the number of base model layers, 1/2')

    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')

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
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='L1: focal-loss gamma (used when --lim1 is set)')
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
    parser.add_argument('--legacy_rng_compat', action='store_true', default=False,
                        help='[ADDED] the authors\' loop built a test-loader iterator every epoch, which silently consumed one draw of the global '
                             'RNG per epoch. Test evaluation is gone from the loop; this flag consumes that draw anyway so a run reproduces the '
                             'authors\' training trajectory exactly (default off: cleaner, statistically equivalent)')
    parser.add_argument('--results_json', type=str, default=None,
                        help='[ADDED] structured results (all metrics, history, spans); default: <checkpoint>_results.json')
    # ===================================================================================================


    # ===================== [ADDED] flags for Limitation 4 (adaptive attention span) =====================
    parser.add_argument('--lim4', action='store_true', default=False,
                        help='Limitation 4 (--lim4): learnable attention span in the 4 Evidence-Cause Interaction branches')
    parser.add_argument('--span_max', type=float, default=0,
                        help='L4: largest span (utterances) a branch may learn. 0 (default) = set automatically to the longest dialogue '
                             'of the data, which GUARANTEES that span_init=1 is exactly full-history attention; a value smaller '
                             'than (longest dialogue - 1) is refused')
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
    # [CHANGED] the data loaders are built BEFORE the model (they draw no random numbers) so that the longest dialogue is known
    train_loader, valid_loader, test_loader = get_MELD_bert_loaders(data_path, batch_size=batch_size, classify=args.cls_type, num_workers=0)
    max_len = max(max(l.dataset.dialogue_lengths()) for l in (train_loader, valid_loader, test_loader))
    if args.lim4:
        # [FIXED] full-history initialisation by construction: span_ratio = 1 is full attention only if max_span >= every key distance
        if args.span_max <= 0:
            args.span_max = float(max_len)
        elif args.span_max < max_len - 1:
            raise SystemExit('--span_max {} is smaller than the longest dialogue ({} utterances): the initial span would cut '
                             'history and L4 would not start from the full-attention baseline. Use --span_max 0 (auto) or >= {}.'.format(
                                 args.span_max, max_len, max_len - 1))
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.lim4:  # [ADDED] L4: span parameters get their own lr and no weight decay
        span_params = [p for n, p in model.named_parameters() if n.startswith('span_modules.')]
        other_params = [p for n, p in model.named_parameters() if not n.startswith('span_modules.')]
        optimizer = optim.Adam([{'params': other_params, 'lr': args.lr, 'weight_decay': args.l2},
                                {'params': span_params, 'lr': args.span_lr, 'weight_decay': 0.0}])

    # ===================== [ADDED] Limitations 1 & 2: loss selection (loss.build_training_loss) =====================
    AUTHORS_GAMMA = 1.0  # default gamma of the authors' `Loss` in this dataset's loss.py
    train_counts = None
    if args.lim1 or args.lim2:
        train_counts = class_counts_from_loader(train_loader, n_classes)
        print('Training label counts per class {}: {}'.format(target_names, train_counts))
    loss_f = build_training_loss(args, n_classes, target_names, class_weights, train_counts, AUTHORS_GAMMA)
    if args.lim1 or args.lim2:
        print('Using ECERCImprovedLoss | L1 (class-balanced + focal): {} | L2 (confusion-aware LS): {}'.format(args.lim1, args.lim2))
        if args.lim1:
            print('  class-balanced alpha: {}'.format([round(a, 3) for a in loss_f.alpha.tolist()]))
        if args.lim2:
            print('  label-smoothing target distribution per true class (%):')
            print(smoothing_table_str(loss_f, target_names))
    else:
        print('Using the authors\' original loss')
    save_model_path = args.save_model_path or os.path.join(args.output_dir, 'ECERC_trained.pkl')
    results_json = args.results_json or os.path.splitext(save_model_path)[0] + '_results.json'
    if args.lim4:
        print('Using L4 adaptive attention span | span_max {} (longest dialogue {}) ramp {} init {} per_head {} penalty {} span_lr {}'.format(
            args.span_max, max_len, args.span_ramp, args.span_init, args.span_per_head, args.span_penalty, args.span_lr))
        print('  (the self-event branch uses the identity mask, distance 0, so its span has no effect: reported as inactive)')
    # ======================================================================================

    # [CHANGED] The test set is NOT touched inside the epoch loop (methodological fix): checkpoints are selected on the
    # validation set only, and the test set is evaluated exactly once, after training, with the saved best checkpoint.
    history, best_valid = [], None
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    for e in range(epochs):
        start_time = time.time()
        tr = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, train_flag=True, optimizer=optimizer,
                                 cuda_flag=cuda_flag, feature_type=feature_type, lim4=args.lim4, span_penalty=args.span_penalty)
        va = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, cuda_flag=cuda_flag,
                                 feature_type=feature_type, keep_probs=True)
        if args.legacy_rng_compat:
            torch.empty((), dtype=torch.int64).random_()   # == the draw made by creating the (now removed) per-epoch test iterator
        eval_loss, eval_fscore = va['loss'], va['wf1']
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
            torch.save(model.state_dict(), save_model_path)  # [ADDED] checkpoint for calibrate.py / inference.py
            best_valid = full_metrics(np.exp(va['log_probs']), va['labels'], target_names)   # validation only
            best_valid['epoch'] = e
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

        entry = {'epoch': e, 'train_loss': tr['loss'], 'train_wf1': tr['wf1'], 'valid_loss': va['loss'], 'valid_acc': va['acc'],
                 'valid_wf1': va['wf1'], 'valid_macro_f1': round(100 * f1_score(va['labels'], va['preds'], average='macro'), 2),
                 'seconds': round(time.time() - start_time, 2)}
        print('epoch: {}, train_loss: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, time: {} sec'.format(
            e, tr['loss'], tr['wf1'], va['loss'], va['acc'], va['wf1'], entry['seconds']))
        if args.lim4:  # [ADDED] L4: learned spans (utterances) after this epoch (logged every epoch -> span_history in the JSON)
            entry['spans'] = model.get_spans()
            entry['mean_span'] = model.mean_span()
            print('   learned spans (utterances): {} | mean {:.2f}'.format(model.get_spans_str(), entry['mean_span']))
        history.append(entry)

        if patience >= args.patience and patience2 >= args.patience:
            print('Early stoping...', patience, patience2)
            break

    # ===================== [CHANGED] single, final test evaluation with the best-validation checkpoint =====================
    model.load_state_dict(torch.load(save_model_path, map_location='cuda' if cuda_flag else 'cpu'))
    te = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, cuda_flag=cuda_flag,
                             feature_type=feature_type, keep_probs=True)
    test_res = full_metrics(np.exp(te['log_probs']), te['labels'], target_names)

    print('Final Test performance (test set evaluated ONCE, with the best-validation checkpoint)...')
    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore, te['acc'], te['wf1']))
    print('\nPer-class test report at best-validation-F1 epoch ({}):'.format(best_epoch))
    print(metrics.classification_report(te['labels'], te['preds'], target_names=target_names, digits=4))
    print(metrics.confusion_matrix(te['labels'], te['preds']))
    print('Validation scores at best-validation epoch: weighted-F1 {:.2f} | macro-F1 {:.2f}'.format(best_valid['weighted_f1'], best_valid['macro_f1']))
    print('Test: ECE {:.2f}% | NLL {:.4f} | Brier {:.4f}'.format(test_res['ece'], test_res['nll'], test_res['brier']))
    print('Checkpoint saved to: {}'.format(save_model_path))

    result = {'schema': 2, 'dataset': DATASET, 'seed': args.seed, 'ckpt': save_model_path,
              'flags': {'lim1': args.lim1, 'lim2': args.lim2, 'lim4': args.lim4}, 'args': vars(args),
              'protocol': {'checkpoint_selection': 'best VALIDATION weighted-F1 (authors\' rule, 2-dp rounded)',
                           'test_evaluations_during_training': 0, 'test_evaluations_total': 1},
              'best_epoch': best_epoch, 'epochs_run': len(history), 'valid': best_valid, 'test': test_res, 'history': history,
              'train_counts': train_counts, 'loss_alpha': loss_f.alpha.tolist() if getattr(loss_f, 'alpha', None) is not None and hasattr(loss_f.alpha, 'tolist') else None}
    if args.lim2:
        result['smoothing_targets'] = loss_f.soft_targets.tolist()
    if args.lim4:
        report = model.span_report()   # spans of the loaded best-validation checkpoint
        print('Learned spans at best-validation epoch (utterances): {} | mean {:.2f}'.format(model.get_spans_str(), model.mean_span()))
        for _k, _v in report.items():
            if _v['status'].startswith('active') and len(_v['per_head']) > 1:
                print('   per-head spans {}: {}'.format(_k, ' '.join('{:.1f}'.format(x) for x in _v['per_head'])))
        result['span'] = {'span_max': args.span_max, 'max_dialogue_length': max_len, 'init_is_full_history': args.span_max >= max_len - 1,
                          'branches': report, 'active_branches': [k for k, v in report.items() if v['status'].startswith('active')],
                          'inactive_branches': [k for k, v in report.items() if not v['status'].startswith('active')], 'mean_span_active_branches': model.mean_span(),
                          'history': [{'epoch': h['epoch'], 'mean_span': h['mean_span'], 'spans': h['spans']} for h in history],
                          'note': 'the mask is applied after the full L x L attention is computed: this limits the effective context, '
                                  'it does NOT reduce attention computation or memory'}
    dump_json(results_json, result)
    print('Structured results saved to: {}'.format(results_json))
