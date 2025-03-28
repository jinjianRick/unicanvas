#!/usr/bin/env bash

init_ctx=("A")
init_end_ctx=(".")

caption='<new1> <new2> <new3> teddybear'
task_token=3
datapath='data/subject/teddybear'
reg_data='real_reg/samples_teddybear'
class_name='teddybear' #
pretrain_model='/opt/data/private/model/sd-v1-4.ckpt'
caption_background="Photo of <new4> sofa <new5> wall <new6> floor."
backgroundpath='data/source/chair2'
task_name='teddybear+chair2'

length=${#init_ctx[@]}

i=0
while [ $i -lt $length ]; do

    python -u  train.py \
            --base configs/finetune_joint.yaml  \
            -t --gpus 0,1 \
            --resume-from-checkpoint-custom  "$pretrain_model" \
            --caption "$caption" \
            --datapath "$datapath" \
            --reg_datapath "$reg_data/images.txt" \
            --reg_caption "$reg_data/caption.txt" \
            --type1 "concept" \
            --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>+<new6>" \
            --name "$task_name-sdv4" \
            --init_ctx "${init_ctx[$i]}" \
            --ctx_end_init "${init_end_ctx[$i]}" \
            --class_name "$class_name" \
            --task_token "$task_token" \
            --layout_sup True \
            --caption2 "$caption_background" \
            --datapath2 "$backgroundpath" \
            --reg_datapath2 "$reg_data/images.txt" \
            --reg_caption2 "$reg_data/caption.txt" \
            --type2 'background' \
            --save_epoch_interval 3

    i=$((i + 1))
done
