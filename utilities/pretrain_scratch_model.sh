#!/bin/bash

## Example of pretraining BERT model from scratch, but making use of an existing tokenizer.

echo "~~~ Executing Pretraining Procedure ~~~"
CUDA_VISIBLE_DEVICES="1" python -u scripts/model/pretrain.py \
    --setting train \
    --model_dir "./data/models/pretraining/synthetic-wnut_2017/scratch-bert-base-cased/" \
    --tokenizer_init bert-base-cased \
    --data_train "./data/annotations/synthetic-wnut_2017/processed/annotations.formatted.json" \
    --data_eval_p 0.05 \
    --sample_random_state 42 \
    --learn_batch 16 \
    --learn_gradient_accumulation_steps 4 \
    --learn_sequence_length_data 128 \
    --learn_sequence_length_model 512 \
    --learn_sequence_overlap 0 \
    --learn_lr 0.00005 \
    --learn_max_steps 1000 \
    --learn_mask_probability 0.15 \
    --learn_mask_max_per_sequence 10 \
    --learn_eval_strategy steps \
    --learn_eval_frequency 50 \
    --learn_random_state 42 \
    --learn_early_stopping \
    --learn_early_stopping_patience 3 \
    --learn_early_stopping_tol 0.0 \
    --learn_save_total_limit 5

echo "~~~ Visualizing Pretraining Loss Curves ~~~"
python -u scripts/model/pretrain.py \
    --setting plot \
    --model_dir "./data/models/pretraining/synthetic-wnut_2017/scratch-bert-base-cased/" \
    --plot_prop 1.0
