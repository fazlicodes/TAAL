#!/bin/bash
# custom config
dset="$1"
CUDA_VISIBLE_DEVICES=1 python generate_pretrained_clip_pseudolabels.py \
--dataset ${dset} \
--model_subtype "GeoRSCLIP_ViTB32" \
--imgs_per_label 16
# --model_subtype "pretrained_model/GeoRSCLIP/RS5M_ViT-L-16.pt" \
