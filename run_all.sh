#!/bin/bash
# custom config
# CUDA_VISIBLE_DEVICES=5
dset="$1"
num_pl=128
pseudo_model="ViT-B/32"
pseudo_model2="ViT-B_32"
# python data_preparation.py --dataset ${dset}

# python faster_gen_clip_pl.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}
# python gen_pls_from_saved.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}
python gen_pls_datapar2.py --dataset ${dset} --model_subtype ${pseudo_model} --imgs_per_label ${num_pl}

model_type=dino
model_subtype=vitb16
python feature_extraction_clip.py --dataset ${dset} --model_type ${model_type} --model_subtype ${model_subtype} \
--feature_path data/${dset}/pretrained_features/ \
--pseudo_conf ""${num_pl}"shot" \
--pseudolabel_model ${pseudo_model2}

echo "Feature extraction done"


python zero_shot_pseudo_adapt.py \
--dataset ${dset} \
--feature_path data/${dset}/pretrained_features/ \
--pretrained_model ${model_type}_${model_subtype} \
--pseudolabel_model ${pseudo_model2} \
--pseudo_conf ""${num_pl}"shot" \
--finetune_type mlp_adapter \
