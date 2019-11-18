import os
import torch
import numpy as np
import os.path as osp
from load_data import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, BertModel)
from transformers import AdamW, WarmupLinearSchedule

from io_utils import *
from collections import defaultdict
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from collections import Counter

def train(train_dataloader, model,classifier, args):

    # Prepare optimizer and schedule (linear warmup and decay)
    criterion = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.num_train_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    for epoch in range(args.start_epoch,args.num_train_epochs):
        correct,total = 0,0
        avg_loss = 0.
        for i, (sentence, attention_mask, relation) in enumerate(train_dataloader):
            model.train()
            sentence, attention_mask, relation = sentence.cuda(args.cuda), attention_mask.cuda(args.cuda), relation.cuda(args.cuda)
            lam = None
            if args.mix_inp:
                sentence, _, _, lam = mixup_data(sentence, relation, args.alpha, num_cuda=args.cuda, lam=lam)
                attention_mask, _, _, lam = mixup_data(attention_mask, relation, args.alpha, num_cuda=args.cuda, lam=lam)
            feature  = model(input_ids=sentence, attention_mask=attention_mask)[0]
            logit = classifier(feature)

            pred =  torch.argmax(logit, 1)
            correct += (pred == relation).sum().item()
            total += pred.size(0)

            if args.mixup:
                feature, relation_a, relation_b, lam = mixup_data(feature, relation, args.alpha, num_cuda=args.cuda, lam=lam)
                logit = classifier(feature)
                loss = mixup_criterion(criterion, logit, relation_a, relation_b, lam)
            else:
                loss = criterion(logit,relation)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            avg_loss += loss.data.item()
            scheduler.step()

            if i % 50 == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:3f} | Accuracy {:3f}'.format(epoch, i, len(train_dataloader), avg_loss/(i+1), (100.*correct)/total))

        if (epoch % args.save_freq == 0) or (epoch == args.num_train_epochs - 1):
            outfile = os.path.join(args.checkpoint_dir, '{:d}.tar'.format(epoch))
            state_dict = {}
            state_dict['epoch'] = epoch
            state_dict['feature'] = model.state_dict()
            state_dict['classifier'] =  classifier.state_dict()
            torch.save(state_dict, outfile)


tp, fp, fn = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0)
keys = set(); # to manage list of keys - If this could be provided separately, remove this.

# Call this for every tensor
def precision_recall(pred, relation):
    for corr_i, pred_i in zip(relation.float(), pred):
        if(corr_i in pred_i):
            tp[corr_i.item()] += 1
        else:
            fp[pred_i.item()] += 1
            fn[corr_i.item()] += 1
        keys.add(corr_i.item())
        for a in pred_i:
            keys.add(a.item())

# call this at the end
def mean_precision_recall():
    mean_precision, mean_recall = 0, 0
    p, r = 0, 0
    for key in keys:
        if(tp[key] + fp[key] != 0):
            mean_precision += tp[key] / (tp[key] + fp[key]) # Adding precision[key]
            p += 1
        if(tp[key] + fn[key] != 0):
            mean_recall += tp[key] / (tp[key] + fn[key]) # Adding recall[key]
            r += 1
    mean_precision = mean_precision/p
    mean_recall = mean_recall/r
    return mean_precision, mean_recall


def test(train_dataloader, model, classifier, args):

    # Prepare optimizer and schedule (linear warmup and decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    k = args.k
    correct, total = 0, 0
    avg_loss = 0.
    pred_all = torch.tensor([], device=args.cuda, dtype=torch.long)
    relation_all = torch.tensor([], device=args.cuda, dtype=torch.long)
    with torch.no_grad():
        for i, (sentence, attention_mask, relation) in enumerate(train_dataloader):
            model.eval()
            sentence, attention_mask, relation = sentence.cuda(args.cuda), attention_mask.cuda(args.cuda), relation.cuda(args.cuda)
            feature  = model(input_ids=sentence, attention_mask=attention_mask)[0]
            logit = classifier(feature)

            pred =  torch.argmax(logit, 1)
            pred_all = torch.cat([pred_all, pred])
            relation_all = torch.cat([relation_all, relation])
            correct += (pred == relation).sum().item()
            total += pred.size(0)

            loss = criterion(logit,relation)
            avg_loss += loss.data.item()

    rel_cpu = relation_all.cpu()
    pred_cpu = pred_all.cpu()
    print('TRUE COUNT:', Counter(rel_cpu.numpy()))
    print('PREDICTED COUNT:', Counter(pred_cpu.numpy()))

    prec, recall, fbeta_score, support = precision_recall_fscore_support(rel_cpu, pred_cpu, average=None, labels=[i for i in range(15)])

    outfile = os.path.join(args.checkpoint_dir, 'confusion_{}.tar'.format(args.split))
    torch.save({'tp': dict(tp), 'fp': dict(fp), 'fn': dict(fn)}, outfile)
    with open(osp.join(args.checkpoint_dir, 'keys_{}.pkl'.format(args.split)), 'wb+') as f:
        pickle.dump(keys, f)
    print('k={}'.format(k))
    print('Avg. loss: {}'.format(avg_loss / (i + 1)))
    print('Accuracy: {}%'.format((100.0 * correct) / total))

    print('Mean Precision: {}'.format(prec))
    print('Mean Recall: {}'.format(recall))


if __name__ == '__main__':
    
    args = parse_args()
    
    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=True)
    
    # get data loader
    if args.dataset == 'nyt':
        dataset = NYT(root=osp.join(osp.dirname(osp.dirname(os.getcwd())), 'data-mining/data/main_data_splits'), split=args.split, tokenizer=tokenizer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_labels = 15
    else:
        raise Exception("Unknown dataset: {}".format(args.dataset))

    # get bert config
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task="data-mining")
    print('bert configs:')
    print(config)

    # get classifier
    # can be cosine or linear
    classifier =  BertClassifier(config, cos=args.cos)

    # get bert(pretrained)
    model = BertModel.from_pretrained(args.model_name_or_path, config=config)

    model.cuda(args.cuda)
    classifier.cuda(args.cuda)

    # get paths for model weights store/load
    args.checkpoint_dir = '%s/%s/%s_train' %(SAVE_DIR, args.dataset, args.config_name)
    if args.mixup:
        args.checkpoint_dir += 'mixup_%.2f'%(args.alpha)
    if args.cos:
        args.checkpoint_dir += 'cos'
    if args.mix_inp:
        args.checkpoint_dir += 'mix_inp'

    print('checkpoints dir : ',args.checkpoint_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.start_epoch = 0

    if args.resume:
        # for testing, one has to load models from certain path
        if args.iter !=-1:
            resume_file = get_assigned_file(args.checkpoint_dir, args.iter)
        else:
            resume_file = get_resume_file(args.checkpoint_dir)
        if resume_file is not None:
            print('Resume file is: ', resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            classifier.load_state_dict(tmp['classifier'])
            model.load_state_dict(tmp['feature'])
        else:
            raise Exception('Resume file not found')

    if args.do_train:
        train(data_loader, model, classifier, args)
    elif args.do_eval:
        test(data_loader, model, classifier, args)
