#!/bin/bash
# custom config
dset="$1"
model_type=dino
model_subtype=vitb16 #vitb16 or resnet50
CUDA_VISIBLE_DEVICES=4 python feature_extraction_clip.py \
--dataset ${dset} \
--model_type ${model_type} \
--model_subtype ${model_subtype} \
--feature_path data/${dset}/pretrained_features/ \
--use_pseudo \
--pseudo_conf "16shot" \
--pseudolabel_model "clip_ViT-B_32" \
--model_dir "all_weights/dino_resnet50_pretrain.pth"
#--pseudolabel_model georsclip
# --pseudolabel_model clip_ViT-B_32 or clip_GeoRSCLIP_ViTB32\
# --model_dir "all_weights/dino_resnet50_pretrain.pth"
# --model_dir "/l/users/sanoojan.baliah/Felix/RS_zero_shot/svl_adapter_models/text_desc_grsb32_svl_adapter_models/${dset}/${dset}_pretrained_encoder.pt"
# --pseudolabel_model "clip_RN50"
# --model_subtype "pretrained_model/GeoRSCLIP/RS5M_ViT-B-32.pt" \
# --pseudolabel_model "clip_ViT-B_32"
