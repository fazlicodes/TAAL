#!/bin/bash
# custom config
# CUDA_VISIBLE_DEVICES=2
dset="$1"
num_pl=64
pseudo_model="ViT-B/32"

python gen_pls_from_saved.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}