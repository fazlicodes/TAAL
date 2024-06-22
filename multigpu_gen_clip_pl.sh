#!/bin/bash
# custom config
dset="$1"
CUDA_VISIBLE_DEVICES=2,3 python multigpu_gen_clip_pl.py \
--dataset ${dset} \
--model_subtype ViT-B/32 \
--imgs_per_label 16
