import os
import glob
import torch
import argparse
import torch.nn as nn
import numpy as np
import os.path as osp
from torch.nn.utils.weight_norm import WeightNorm

SAVE_DIR = osp.join(osp.dirname(osp.dirname(os.getcwd())), 'data-mining')

# parser
def parse_args():
    parser = argparse.ArgumentParser(description= 'ATCNN')
    parser.add_argument("--dataset", default="nyt", type=str,
                        help="dataset")
    parser.add_argument("--config_name", default='bert-base-uncased', type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--split", default="train", type=str, choices=['train', 'test', 'example'],
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default='bert-base-uncased', type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--mixup", action='store_true',
                        help="Train using mixup")
    parser.add_argument("--alpha", default=2., type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--resume", action='store_true',
                        help="Train using mixup")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--save_freq', type=int, default=10,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--iter', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--mix_inp', action='store_true',
                        help='mixup of input')
    parser.add_argument('--cuda', type=int, help='gpu number')
    parser.add_argument('--cos', action='store_true', 
                        help='cosine classifier')
    parser.add_argument('--k', type=int, help='k for prec and recall')

    
    return parser.parse_args()


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class BertClassifier(nn.Module):
    def __init__(self, config, cos=False):
        super(BertClassifier, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if cos:
            self.classifier = distLinear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [x for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def mixup_data(x, y, alpha=1.0, num_cuda=0, lam=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(num_cuda)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
