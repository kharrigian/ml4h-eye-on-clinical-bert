#!/bin/bash

python -u scripts/annotation/transform_worksheet.py \
    --input_file data/annotations/synthetic-notes/labeled/synthetic.batch-0.xlsx \
    --input_file_src data/annotations/synthetic-notes/labeled/synthetic.source-annotations.json \
    --input_file_metadata data/resources/synthetic-notes/metadata.csv \
    --output_dir data/annotations/synthetic-notes/labeled-processed/ \
    --rm_existing
