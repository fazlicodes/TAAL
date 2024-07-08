#!/bin/bash
# custom config
dset="$1"
num_pl=64
pseudo_model="ViT-B/32"
pseudo_model2="ViT-B_32"
# python data_preparation.py --dataset ${dset}

python gen_pls_datapar2.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}