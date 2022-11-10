#! /bin/bash

model_path=$1
out_path=$2
pgt_dev=$3
gpu=$4

CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python src/simple_predict_test.py \
--given_pred_grammar_terms $pgt_dev \
--insert_offset_bracket_into_source \
--out_path $out_path \
--model_path $model_path \
--test_set '/home/lr/kunitaka/project/fbc/t5_genchal/data/train_dev/TEST.prep_feedback_comment.public.tsv'