#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python cluster.py \
       --dir "./data" \
       --batch-size 256 \
       --input-dim 27 \
       --n-classes 10 \
       --lr 3e-4 \
       --wd 1e-6 \
       --epoch 100 \
       --pre-epoch 10 \
       --pretrain \
       --lamda 1 \
       --beta 1 \
       --hidden-dims 32 64 128 \
       --latent-dim 8 \
       --n-clusters 8 \
       --n-jobs 1 \
       --cuda \
       --log-interval 10
