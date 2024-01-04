#!/bin/bash

python -u scripts/annotation/prepare_worksheet.py \
    --input_file data/resources/synthetic-notes/synthetic.csv \
    --output_dir data/annotations/synthetic-notes/unlabeled/ \
    --output_file synthetic.xlsx \
    --loader_handle_surgical merge \
    --loader_handle_codes all \
    --loader_handle_headers all \
    --loader_handle_expand all \
    --loader_handle_anti_vegf \
    --window_size 10 \
    --batch_size 10 \
    --n_batches 1 \
    --random_state 42 \
    --n_jobs 1

