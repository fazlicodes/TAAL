#!/bin/bash
# custom config
dset="$1"
model_subtype=ViT-B/32  #vitb16 or resnet50
CUDA_VISIBLE_DEVICES=1 python zero_shot_pseudo_adapt.py \
--dataset ${dset} \
--feature_path data/${dset}/pretrained_features_bb_clip/ \
--pretrained_model "clip_ViT-B_32" \
--pseudolabel_model 'clip_ViT-B_32' \
--pseudo_conf '16shot' \
--finetune_type mlp_adapter \
--bb ViT-B/32

# --bb vitb16
#--pseudolabel_model 'clip_ViT-B_32' or georsclip \
# --pretrained_model dino_vitb16 or simclr_RN50
# --feature_path data/${dset}/${model_sub}_${model_subtype}/pretrained_features/ \
# --pseudolabel_model georsclip \
