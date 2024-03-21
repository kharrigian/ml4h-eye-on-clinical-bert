#!/bin/bash

echo "~~~ Beginning Entity Model Training ~~~"
OUTPUT_DIR="temp-out/"
python -u resources/ml4h-clinical-bert/scripts/model/train.py \
        --output_dir $OUTPUT_DIR \
        --rm_existing \
        --datasets "data/raw/kharrigian/annotations/internal-personalizing-language/processed-v2/personalizing-language-annotations.json" \
        --encoder bert-base-uncased \
        --tokenizer bert-base-uncased \
        --split_input \
        --sequence_overlap_input 0 \
        --sequence_overlap_model 0 \
        --sequence_overlap_type_model "first" \
        --max_sequence_length_input 512 \
        --max_sequence_length_model 512 \
        --sequence_split_type_input continuous \
        --exclude_autolabel \
        --exclude_non_specified \
        --eval_cv 5 \
        --eval_cv_fold 0 \
        --eval_mc_split_frac 3 1 1 \
        --eval_strategy stratified_k_fold \
        --eval_cv_groups enterprise_mrn \
        --limit_group enterprise_mrn \
        --model_eval_batch_size 16 \
        --model_lr_adaptive_method "step" \
        --model_lr_warmup 100 \
        --model_lr_step_size 250 \
        --model_lr_gamma 0.9 \
        --model_opt_method "adamw" \
        --model_grad_clip_norm 1 \
        --model_lr 0.00005 \
        --model_dropout_p 0.1 \
        --model_weight_decay 0.01 \
        --model_n_steps 500 \
        --model_eval_strategy "steps" \
        --model_eval_frequency 10 \
        --model_train_batch_size 8 \
        --model_train_gradient_accumulation 8 \
        --early_stopping_tol 0.01 \
        --early_stopping_patience 5 \
        --early_stopping_warmup 50 \
        --display_batch_loss \
        --random_state 42 \
        --baseline_skip \
        --gpu_id 0 \
        --include_attribute \
        --attribute_keys personalizing_language_type \
        --weighting_attribute "balanced" \
        --attributes_hidden_size 256
        # --include_entity \
        # --entity_key label \
        # --weighting_entity balanced \
        # --entity_hidden_size 256 \
        # --model_save_criteria "all" \
        # --model_save_predictions \
        # --use_crf \

        # --freeze_encoder
        # --gpu_id 2 \
