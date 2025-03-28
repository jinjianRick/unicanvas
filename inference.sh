#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where retrieved images are saved
# 4rth arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path

#epoch_list=("319" "479" "639" "799" "899" "319" "479" "639" "799" "899") #cat

#epoch_list=("335" "503" "671" "839" "899" "335" "503" "671" "839" "899" "335" "503" "671" "839" "899")
#file_list=("logs/2023-10-07T03-39-52_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T03-39-52_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T03-39-52_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T03-39-52_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T03-39-52_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-18-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-18-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-18-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-18-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-18-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-59-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-59-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-59-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-59-50_dog-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-07T04-59-50_dog-sdv4/checkpoints/embeddings_gs-")
#epoch_list=("899" "187" "375" "563" "751" "899")
#file_list=("logs/2023-10-10T09-35-21_castle-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-10T10-13-38_castle-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-10T10-13-38_castle-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-10T10-13-38_castle-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-10T10-13-38_castle-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-10T10-13-38_castle-sdv4/checkpoints/embeddings_gs-")
#epoch_list=("719" "899")
#file_list=("logs/2023-10-11T07-09-51_shoes2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-11T07-09-51_shoes2-sdv4/checkpoints/embeddings_gs-")

#epoch_list=("899" "799" "639" "479" "319" "899" "799" "639" "479" "319")
#file_list=("logs/2023-10-14T07-11-23_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-11-23_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-11-23_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-11-23_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-11-23_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-49-57_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-49-57_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-49-57_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-49-57_cat2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-14T07-49-57_cat2-sdv4/checkpoints/embeddings_gs-")

#epoch_list=("899" "755" "899" "755")
#file_list=("logs/2023-10-15T08-32-35_backpack-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T08-32-35_backpack-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T09-09-43_backpack-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T09-09-43_backpack-sdv4/checkpoints/embeddings_gs-")
epoch_list=("899" "863" "647" "899" "863" "647")
file_list=("logs/2023-10-15T12-07-39_sunglasses-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T12-07-39_sunglasses-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T12-07-39_sunglasses-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T13-21-41_sunglasses-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T13-21-41_sunglasses-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-15T13-21-41_sunglasses-sdv4/checkpoints/embeddings_gs-")
epoch_list=("899" "803" "535" "267")
file_list=("logs/2023-10-17T06-08-45_purse2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-17T06-08-45_purse2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-17T06-08-45_purse2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-17T06-08-45_purse2-sdv4/checkpoints/embeddings_gs-")

epoch_list=("899" "815" "611")
file_list=("logs/2023-10-18T03-42-51_chair3-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-18T03-42-51_chair3-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-18T03-42-51_chair3-sdv4/checkpoints/embeddings_gs-")

epoch_list=("899" "803" "535" "267")
file_list=("logs/2023-10-20T07-54-28_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T07-54-28_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T07-54-28_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T07-54-28_dish2-sdv4/checkpoints/embeddings_gs-")

epoch_list=("899" "803" "535" "267" "899" "803" "535" "267")
file_list=("logs/2023-10-20T10-55-30_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-55-30_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-55-30_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-55-30_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-18-13_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-18-13_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-18-13_dish2-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T10-18-13_dish2-sdv4/checkpoints/embeddings_gs-")

epoch_list=("899" "803" "535" "267")
file_list=("logs/2023-10-20T04-39-00_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T04-39-00_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T04-39-00_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T04-39-00_dish-sdv4/checkpoints/embeddings_gs-" )

epoch_list=("803" "535" "267" "899" "803" "535" "267")
file_list=("logs/2023-10-20T06-04-01_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-04-01_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-04-01_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-39-33_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-39-33_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-39-33_dish-sdv4/checkpoints/embeddings_gs-" "logs/2023-10-20T06-39-33_dish-sdv4/checkpoints/embeddings_gs-")

epoch_list=("791" "791")
file_list=("logs/2023-10-15T01-57-06_skull_mug-sdv4/checkpoints/embeddings_gs-"  "logs/2023-10-15T03-08-56_skull_mug-sdv4/checkpoints/embeddings_gs-")


suffix=".ckpt"

ARRAY=()

# for i in "$@"
# do 
#     echo $i
#     ARRAY+=("${i}")
# done

length=${#epoch_list[@]}

#python src/retrieve.py --target_name "${ARRAY[0]}" --outpath ${ARRAY[2]}

i=0
while [ $i -lt $length ]; do

    echo "${file_list[$i]}${epoch_list[$i]}${suffix}"

    python3 sample.py \
    --from_file customconcept101/prompts/supply_tmp.txt \
    --delta_ckpt "${file_list[$i]}${epoch_list[$i]}${suffix}" \
    --ckpt ./pretrain_model/sd-v1-4.ckpt \
    --device "cuda:0" \
    --specific_epoch "${epoch_list[$i]}" \
    --add_prompt "<new1> <new2> skull mug"

    i=$((i + 1))
done