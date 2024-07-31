#!/bin/bash
set -e

# custom config
dset="oxford_flowers"
pseudo_model="ViT-B/32"
pseudo_model2="ViT-B_32"
model_type=dino
model_subtype=vitb16

num_pl=8
# python data_preparation.py --dataset ${dset}
python generate_pls.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}

python feature_extraction_clip.py --dataset ${dset} --model_type ${model_type} --model_subtype ${model_subtype} \
--feature_path data/${dset}/pretrained_features/ \
--pseudo_conf ""${num_pl}"shot" \
--pseudolabel_model ${pseudo_model2}

python zero_shot_pseudo_adapt.py \
--dataset ${dset} \
--feature_path data/${dset}/pretrained_features/ \
--pretrained_model ${model_type}_${model_subtype} \
--pseudolabel_model ${pseudo_model2} \
--pseudo_conf ""${num_pl}"shot" \
--finetune_type mlp_adapter

echo "-----------------------------------"


num_pl=64
# python data_preparation.py --dataset ${dset}
python generate_pls.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}

python feature_extraction_clip.py --dataset ${dset} --model_type ${model_type} --model_subtype ${model_subtype} \
--feature_path data/${dset}/pretrained_features/ \
--pseudo_conf ""${num_pl}"shot" \
--pseudolabel_model ${pseudo_model2}

python zero_shot_pseudo_adapt.py \
--dataset ${dset} \
--feature_path data/${dset}/pretrained_features/ \
--pretrained_model ${model_type}_${model_subtype} \
--pseudolabel_model ${pseudo_model2} \
--pseudo_conf ""${num_pl}"shot" \
--finetune_type mlp_adapter
