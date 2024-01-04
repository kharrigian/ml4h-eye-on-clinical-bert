#!/bin/bash

echo "~~~ Beginning Training Script ~~~"
python -u scripts/model/train.py \
        --output_dir "data/models/classifiers/synthetic-notes/" \
        --rm_existing \
        --datasets "data/annotations/synthetic-wnut_2017/labeled-processed/annotations.formatted.json" \
        --encoder bert-base-cased \
        --tokenizer bert-base-cased \
        --split_input \
        --sequence_overlap_input 0 \
        --sequence_overlap_model 0 \
        --sequence_overlap_type_model "first" \
        --max_sequence_length_input 128 \
        --max_sequence_length_model 512 \
        --sequence_split_type_input continuous \
        --entity_key label \
        --attribute_keys wnut \
        --include_attribute \
        --exclude_autolabel \
        --exclude_non_specified \
        --eval_cv 5 \
        --eval_cv_fold 0 \
        --eval_mc_split_frac 3 1 1 \
        --eval_strategy stratified_k_fold \
        --eval_cv_groups user_id \
        --eval_train \
        --eval_test \
        --limit_group user_id \
        --limit_spans_per_group 25 \
        --limit_spans_per_group_label_stratify \
        --model_eval_batch_size 16 \
        --model_save_frequency 0 \
        --model_lr_adaptive_method "step" \
        --model_lr_warmup 100 \
        --model_lr_step_size 50 \
        --model_lr_gamma 0.9 \
        --model_opt_method "adamw" \
        --model_grad_clip_norm 1 \
        --model_lr 0.00001 \
        --model_dropout_p 0.1 \
        --model_weight_decay 0.01 \
        --model_n_steps 100 \
        --model_eval_strategy "steps" \
        --model_eval_frequency 50 \
        --attributes_hidden_size 256 \
        --model_train_batch_size 8 \
        --model_train_gradient_accumulation 1 \
        --weighting_attribute "balanced" \
        --use_attribute_concept_bias \
        --early_stopping_tol 0.01 \
        --early_stopping_patience 5 \
        --early_stopping_warmup 50 \
        --baseline_use_char \
        --random_state 42 \
        --gpu_id -1


