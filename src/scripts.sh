#!/bin/sh

# train ordinary model
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 10 &> train_bert.log

# train mixup model
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10 &> train_mixup.log

# test oridiary model
python main.py --split test --do_eval --resume --iter 9 --cuda 0 &> test_bert.log

# test mixup model
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0 &> test_mixup.log

# train ordinary model with cosine classifier
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 10 --cos &> train_bert_cos.log

# train mixup model
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10 --cos &> train_mixup_cos.log

# test oridiary model
python main.py --split test --do_eval --resume --iter 9 --cuda 0 --cos &> test_bert_cos.log

# test mixup model
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0 --cos &> test_mixup_cos.log

# train bert+mixup(inputs)+cos
python main.py --split train --do_train --mixup --batch-size 32 --cuda 0 --num_train_epochs 10 --cos --mix_inp &> train_mixup_inp_cos.log

# test bert+mixup(inputs)+cos
python main.py --split test --do_eval --resume --iter 9 --mixup --cuda 0 --cos &> test_mixup_inp_cos.log
