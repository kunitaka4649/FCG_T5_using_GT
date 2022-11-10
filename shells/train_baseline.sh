#! /bin/bash

out_path=$1
seed=$2
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python src/simple_train.py \
--insert_offset_bracket_into_source \
--out_path $out_path \
--seed $seed