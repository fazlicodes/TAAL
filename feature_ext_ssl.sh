#!/bin/bash
# custom config
dset="$1"
model_type=ssl
folder_name=dino_rn50
CUDA_VISIBLE_DEVICES=1 python feature_extraction.py \
--dataset ${dset} \
--model_type ${model_type} \
--model_subtype RN50 \
--feature_path data/${dset}/pretrained_features/ \
--model_dir ${folder_name}/${dset} \
--use_pseudo \
--pseudo_conf "16shot" \
--pseudolabel_model "georsclip"
#--pseudolabel_model "clip_ViT-B_32"
#--pseudolabel_model "clip_RN50"
# --model_subtype RN50 \
# folder_name=ssl_pretraining_2024-02-29_00_26_42_mlrsnet_resnet50_112imres_100ep_512bs
# --model_dir data/${dset}/${folder_name}/models/models/100 \
