#!/bin/bash
# custom config
dset="$1"
backbone="resnet50"
backbone_model="resnet50" #CHANGE IN CODE
epochs=200
current_datetime=$(date +"%Y-%m-%d_%H:%M:%S")
formatted_datetime=$(echo "$current_datetime" | sed 's/:/_/g')
batch_size=128 #450 or 850
im_res=112 #112, 224
exp_name=ssl_pretraining_${formatted_datetime}_${dset}_${backbone_model}_${im_res}imres_${epochs}ep_${batch_size}bs
CUDA_VISIBLE_DEVICES=0 python ssl_pretraining.py > "${exp_name}".txt 2>&1 \
--dataset ${dset} \
--output_dir data/${dset}/${exp_name}/models/ \
--backbone $backbone \
--batch_size $batch_size \
--epochs $epochs \
--im_res $im_res \
--wandb_run_name "${exp_name}"