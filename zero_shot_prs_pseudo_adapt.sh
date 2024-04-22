#!/bin/bash
# custom config
dset="$1"
epochs=50
CUDA_VISIBLE_DEVICES=3 python zero_shot_pseudo_adapt.py \
--dataset ${dset} \
--feature_path data/${dset}/pretrained_features/ \
--pretrained_model "clip_pretrained_model_GeoRSCLIP_RS5M_ViT-B-32.pt" \
--pseudolabel_model 'georsclip' \
--pseudo_conf "16shot" \
--finetune_type "mlp_adapter" \
--epochs $epochs \
--bb "RN50"
# --pseudolabel_model 'clip_ViT-B_32' \
