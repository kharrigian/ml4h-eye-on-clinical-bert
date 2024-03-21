#!/bin/bash

# echo "~~~ Beginning Entity Model Visualization ~~~"
# python -u scripts/model/train_aggregate.py \
#         --results_dir "data/models/classifiers/synthetic-wnut_2017-entity/"\
#         --eval_strategy "steps" \
#         --model "model" \
#         --output_dir "data/models/classifiers/synthetic-wnut_2017-entity/summary/" \
#         --rm_existing \
#         --dpi 150 \
#         --plot

# echo "~~~ Beginning Attribute Model Visualization ~~~"
# python -u scripts/model/train_aggregate.py \
#         --results_dir "data/models/classifiers/synthetic-wnut_2017-attributes/"\
#         --eval_strategy "steps" \
#         --model "model" \
#         --output_dir "data/models/classifiers/synthetic-wnut_2017-attributes/summary/" \
#         --rm_existing \
#         --dpi 150 \
#         --plot

python -u resources/ml4h-clinical-bert/scripts/model/train_aggregate.py \
        --results_dir "temp-out/"\
        --eval_strategy "steps" \
        --model "model" \
        --output_dir "temp-out/summary/" \
        --rm_existing \
        --dpi 150 \
        --plot