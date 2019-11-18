#!/bin/sh

###################################################################################
########################## BASELINE Att-BiLSTM ####################################
###################################################################################

# train
python train.py --num_epochs 10 --batch_size 32 --cuda 0 

# eval
python train.py --num_epochs 10 --batch_size 32 --cuda 0 --eval

###################################################################################
################################### MixBERT #######################################
###################################################################################

# train ordinary model
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 10 

# train mixup model
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10

# test oridiary model
python main.py --split test --do_eval --resume --iter 9 --cuda 0 

# test mixup model
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0

# train ordinary model with cosine classifier
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 10 --cos 

# train mixup+cos model
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10 --cos 

# test oridiary model
python main.py --split test --do_eval --resume --iter 9 --cuda 0 --cos

# test mixup model
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0 --cos 

# train bert+mixup(inputs)+cos
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10 --cos --mix_inp

# test bert+mixup(inputs)+cos
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0 --cos
