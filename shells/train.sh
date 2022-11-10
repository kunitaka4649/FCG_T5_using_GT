#! /bin/bash

pgt_train=$1
pgt_dev=$2
out_path=$3
seed=$4
gpu=$5

CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python src/simple_train.py \
--given_pred_grammar_terms_train $pgt_train \
--given_pred_grammar_terms_dev $pgt_dev \
--insert_offset_bracket_into_source \
--out_path $out_path \
--seed $seed