#!/bin/sh

# train ordinary model
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 3

# train mixup model
python main.py --split train --do_train --batch-size 32 --cuda 0 --num_train_epochs 3

# test oridiary model
python main.py --split test --do_eval --resume --iter 2

# test mixup model
python main.py --split test --do_eval --resume --iter 2 --mixup
