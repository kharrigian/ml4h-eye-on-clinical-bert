#!/bin/bash

python -u scripts/annotation/transform_worksheet.py \
    --input_file data/annotations/synthetic-notes/synthetic.batch-0.xlsx \
    --input_file_src data/annotations/synthetic-notes/synthetic.source-annotations.json \
    --input_file_metadata data/resources/synthetic-notes/metadata.csv \
    --output_dir data/annotations/synthetic-notes/processed/ \
    --rm_existing
