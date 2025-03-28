#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where retrieved images are saved
# 4rth arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path
# (4,2) (3,2) (2,2) (2,1) (3,1) (4,1)
#init_ctx=("A photo of a" "Photo of a" "Photo of" "A photo of a" "Photo of a" "Photo of")
#init_end_ctx=(". ." ". ." ". ." "." "." ".")
init_ctx=("A photo of a" "Photo of a" "Photo of" "A photo of a" "Photo of a" "Photo of")
init_end_ctx=(". ." "." ". ." "." ". ." ".")

#init_ctx=("A photo of")
init_ctx=("A")
init_end_ctx=(".")

#token_pos='{"concept":[2,3,4,5,6,7], "background":[1, 8]}'
#token_pos='{"concept":[2,3,4,5,6], "background":[1, 7]}'
token_pos='{"concept":[2,3,4,5], "background":[1, 6]}'
#token_pos='{"concept":[4,5,6], "background":[1,2,3,7]}'
#token_pos='{"concept":[4,5], "background":[1,2,3,6]}'
#token_pos='{"concept":[4,5,6,7,8], "background":[1,2,3,9]}'

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done

length=${#init_ctx[@]}

#python src/retrieve.py --target_name "${ARRAY[0]}" --outpath ${ARRAY[2]}

i=0
while [ $i -lt $length ]; do

    python -u  train.py \
            --base configs/custom-diffusion/${ARRAY[4]}  \
            -t --gpus 0,1,2 \
            --resume-from-checkpoint-custom  ${ARRAY[5]} \
            --caption "${ARRAY[0]}" \
            --datapath ${ARRAY[1]} \
            --reg_datapath "${ARRAY[2]}/images.txt" \
            --reg_caption "${ARRAY[2]}/caption.txt" \
            --type1 "concept" \
            --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>+<new6>" \
            --name "${ARRAY[8]}-sdv4" \
            --init_ctx "${init_ctx[$i]}" \
            --ctx_end_init "${init_end_ctx[$i]}" \
            --figure_num 7 \
            --class_name "${ARRAY[3]}" \
            --task_token 3 \
            --layout_sup True \
            --caption2 "${ARRAY[6]}" \
            --datapath2 ${ARRAY[7]} \
            --reg_datapath2 "${ARRAY[2]}/images.txt" \
            --reg_caption2 "${ARRAY[2]}/caption.txt" \
            --type2 'background' \
            --token_pos "$token_pos"



    i=$((i + 1))
done
