#!/bin/bash
# custom config
CUDA_VISIBLE_DEVICES=4
dset="$1"
python data_preparation.py --dataset ${dset}

python faster_gen_clip_pl.py --dataset ${dset} --model_subtype ViT-B/32 --imgs_per_label 16

model_type=dino
model_subtype=vitb16 
python feature_extraction_clip.py --dataset ${dset} --model_type ${model_type} --model_subtype ${model_subtype} \
--feature_path data/${dset}/pretrained_features/ \
--use_pseudo \
--pseudo_conf "16shot" \
--pseudolabel_model "clip_ViT-B_32" \
--model_dir "all_weights/dino_resnet50_pretrain.pth"
