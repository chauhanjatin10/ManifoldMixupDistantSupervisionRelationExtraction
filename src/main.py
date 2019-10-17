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
            feature  = model(input_ids=sentence, attention_mask=attention_mask)[0]
            logit = classifier(feature)

            pred =  torch.argmax(logit, 1)
            correct += (pred == relation).sum().item()
            total += pred.size(0)

            if args.mixup:
                feature, relation_a, relation_b, lam = mixup_data(feature, relation, args.alpha, num_cuda=args.cuda)
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

def precision_recall(train_dataloader, model, classifier, args):

    # Prepare optimizer and schedule (linear warmup and decay)
    criterion = torch.nn.CrossEntropyLoss()
    tp, fp, fn = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0)
    with torch.no_grad():
        for i, (sentence, attention_mask, relation) in enumerate(train_dataloader):
            model.eval()
            sentence, attention_mask, relation = sentence.cuda(args.cuda), attention_mask.cuda(args.cuda), relation.cuda(args.cuda)
            feature  = model(input_ids=sentence, attention_mask=attention_mask)[0]
            logit = classifier(feature)

            pred =  torch.argmax(logit, 1)

            # pred is prediction, relation is true value
            for corr_i, pred_i in zip(relation, pred):
                if(corr_i == pred_i):
                    tp[corr_i.item()] += 1
                else:
                    fp[pred_i.item()] += 1
                    fn[corr_i.item()] += 1

    mean_precision, mean_recall = 0, 0
    precision, recall = {}, {}
    for corr_i in tp:
        if(corr_i not in fp):
            fp[corr_i] = 0
        if(corr_i not in fn):
            fn[corr_i] = 0

        if(tp[corr_i] + fp[corr_i] != 0):
            precision[corr_i] = tp[corr_i] / (tp[corr_i] + fp[corr_i])
            mean_precision += precision[corr_i]
        if(tp[corr_i] + fn[corr_i] != 0):
            recall[corr_i] = tp[corr_i] / (tp[corr_i] + fn[corr_i])
            mean_recall += recall[corr_i]
    mean_precision = mean_precision/len(tp)
    mean_recall = mean_recall/len(tp)
    print('Precision: {}'.format(mean_precision))
    print('Recall: {}'.format(mean_recall))


def test(train_dataloader, model, classifier, args):

    # Prepare optimizer and schedule (linear warmup and decay)
    criterion = torch.nn.CrossEntropyLoss()

    correct, total = 0, 0
    avg_loss = 0.
    with torch.no_grad():
        for i, (sentence, attention_mask, relation) in enumerate(train_dataloader):
            model.eval()
            sentence, attention_mask, relation = sentence.cuda(args.cuda), attention_mask.cuda(args.cuda), relation.cuda(args.cuda)
            feature  = model(input_ids=sentence, attention_mask=attention_mask)[0]
            logit = classifier(feature)

            pred =  torch.argmax(logit, 1)
            correct += (pred == relation).sum().item()
            total += pred.size(0)

            loss = criterion(logit,relation)
            avg_loss += loss.data.item()

    print('Avg. loss: {}'.format(avg_loss / (i + 1)))
    print('Accuracy: {}%'.format((100.0 * correct) / total))


if __name__ == '__main__':
    
    args = parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=True)
    
    if args.dataset == 'nyt':
        dataset = NYT(root=osp.join(osp.dirname(osp.dirname(os.getcwd())), 'data-mining/data/main_data_splits'), split=args.split, tokenizer=tokenizer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_labels = 53

    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task="data-mining")
    classifier =  BertClassifier(config)
    model = BertModel.from_pretrained(args.model_name_or_path, config=config)

    model.cuda(args.cuda)
    classifier.cuda(args.cuda)
    args.checkpoint_dir = '%s/%s/%s_train' %(SAVE_DIR, args.dataset, args.config_name)
    if args.mixup:
        args.checkpoint_dir += 'mixup_%.2f'%(args.alpha)

    print('checkpoints dir : ',args.checkpoint_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.start_epoch = 0

    if args.resume:
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
