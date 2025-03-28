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

init_ctx=("A")
init_end_ctx=(".")

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
            --modifier_token "<new1>+<new2>+<new3>" \
            --name "${ARRAY[3]}-sdv4" \
            --init_ctx "${init_ctx[$i]}" \
            --ctx_end_init "${init_end_ctx[$i]}" \
            --figure_num 5 \
            --class_name "${ARRAY[3]}" \
            --task_token 3 \
            --layout_sup True


    i=$((i + 1))
done