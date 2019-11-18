#! /usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

import os
import sys
import argparse
import time
import math
import pickle
import os.path as osp
from collections import defaultdict
from bilstm_model import SelfAttention
from load_data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from collections import Counter

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--create_corpus', type=int, default=0,
					help='whether to create or load corpus')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
					help='batch size')
parser.add_argument('--emsize', type=int, default=300,
					help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=300,
					help='size of hidden tensor')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wdecay', type=float, default=1e-8)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--eval', action='store_true', help='evaluate model')

args = parser.parse_args()

sys.stdout = open('train.log', 'w+')

# create a corpus to save time
if args.create_corpus == 1:
	print("Creating New Corpus")
	corpus = Dataset()
	corpus_save_path = "nyt_corpus.pkl"
	with open(corpus_save_path, 'wb') as output:
		pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)
	print("Number of training samples = ", len(corpus.train_sentences))
	print("Number of testing samples = ", len(corpus.test_sentences))
	assert len(corpus.train_sentences) == len(corpus.train_labels)
	assert len(corpus.test_sentences) == len(corpus.test_labels)

else:
	print("Loading saved Corpus")
	corpus_save_path = "nyt_corpus.pkl"
	with open(corpus_save_path, 'rb') as input_:
		corpus = pickle.load(input_)
	print("Number of training samples = ", len(corpus.train_sentences))
	print("Number of testing samples = ", len(corpus.test_sentences))
	assert len(corpus.train_sentences) == len(corpus.train_labels)
	assert len(corpus.test_sentences) == len(corpus.test_labels)


# create model

MODEL = SelfAttention(args.batch_size, len(corpus.relation_dict), args.hidden_size, len(corpus.vocab),
				args.emsize, None)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MODEL.parameters(), lr=args.lr, betas=(0, 0.999), eps=1e-9, 
					weight_decay=args.wdecay)

MODEL = MODEL.cuda()
MODEL = nn.DataParallel(MODEL, range(args.cuda))


def test():

    # Prepare optimizer and schedule (linear warmup and decay)
    criterion = torch.nn.CrossEntropyLoss()
    ckpt_path = osp.join(osp.dirname(os.getcwd()), 'data-mining', 'nyt', 'bilstm', 'ckpt.tar')
    
    correct, total = 0, 0
    avg_loss = 0.
    pred_all = torch.tensor([], device=args.cuda, dtype=torch.long)
    relation_all = torch.tensor([], device=args.cuda, dtype=torch.long)
    MODEL.load_state_dict(torch.load(ckpt_path))
    MODEL.train()
    num_iters = math.ceil(len(corpus.train_sentences)/args.batch_size)

    for iter_ in range(num_iters):
        batch, labels = corpus.get_batch("train", iter_, args.batch_size)
        batch, labels = batch.cuda(), labels.cuda()

        logit = MODEL(batch, batch.shape[0])

        # call metrics
        pred =  torch.argmax(logit, 1)
        pred_all = torch.cat([pred_all, pred])
        relation_all = torch.cat([relation_all, labels])
        correct += (pred == labels).sum().item()
        total += pred.size(0)
        
        loss = criterion(logit,labels)
        avg_loss += loss.data.item()

    rel_cpu = relation_all.cpu()
    pred_cpu = pred_all.cpu()
    print('TRUE COUNT:', Counter(rel_cpu.numpy()))
    print('PREDICTED COUNT:', Counter(pred_cpu.numpy()))
    prec, recall, fbeta_score, support = precision_recall_fscore_support(rel_cpu, pred_cpu, average=None, labels=[i for i in range(15)])

    print('Avg. loss: {}'.format(avg_loss / (iter_ + 1)))
    print('Accuracy: {}%'.format((100.0 * correct) / total))

    print('Mean Precision: {}'.format(prec))
    print('Mean Recall: {}'.format(recall))


def train():
	for epoch in range(args.num_epochs):
		MODEL.train()
		num_iters = math.ceil(len(corpus.train_sentences)/args.batch_size)
        
		for iter_ in range(num_iters):
			batch, labels = corpus.get_batch("train", iter_, args.batch_size)
			batch, labels = batch.cuda(), labels.cuda()

			output_logits = MODEL(batch, batch.shape[0])

			optimizer.zero_grad()
			loss_ = criterion(output_logits, labels)
			loss_.backward()
			optimizer.step()

			if iter_%100 == 0:
				print("loss = ", loss_.detach().cpu().numpy())

		test()
	torch.save(MODEL.state_dict(), '../data-mining/nyt/bilstm/ckpt.tar')

if __name__ == '__main__':
    
    if not args.eval:
        train()
    else:
        test()
