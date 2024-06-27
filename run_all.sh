#!/bin/bash
# custom config
CUDA_VISIBLE_DEVICES=4
dset="$1"
# python data_preparation.py 

# python faster_gen_clip_pl.py --dataset ${dset} --model_subtype ViT-B/32 --imgs_per_label 16

