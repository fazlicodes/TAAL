#!/bin/bash
# custom config
dset="$1"
CUDA_VISIBLE_DEVICES=1 python fast_gen_clip_pl.py \
--dataset ${dset} \
--model_subtype ViT-B/32 \
--imgs_per_label 32
# --model_subtype RN50 \ViT-B/32
# CUDA_VISIBLE_DEVICES=0 python generate_clip_pseudolabels.py \
# CUDA_VISIBLE_DEVICES=3 python fast_gen_clip_pl.py \