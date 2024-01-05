
"""
Run this script to generate formatted annotations that can be passed to the training script
"""

## Imports
import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from cce.util.helpers import apply_hash

## Load Dataset
wnut = load_dataset("wnut_17",
                    cache_dir="./data/resources/synthetic-wnut_2017/raw/")

## W-Nut Tags
tag_map = {
    0:None,
    1:"B-corporation",
    2:"I-corporation",
    3:"B-creative-work",
    4:"I-creative-work",
    5:"B-group",
    6:"I-group",
    7:"B-location",
    8:"I-location",
    9:"B-person",
    10:"I-person",
    11:"B-product",
    12:"I-product"
}

## Random seed
random_seed = np.random.RandomState(42)

## Formatted Instane Cache
formatted_instances = []

## Iterate Through Splits
for split in ["train","validation","test"]:
    ## Iterate Through Instances in the Split
    for instance in wnut[split]:
        ## Initialize Text String and Spans
        text_str = ""
        text_spans = []
        ## Track Span Start and Type
        cur_span_start = None
        cur_span_type = None
        ## Iterate Through Tokens
        for token, tag in zip(instance["tokens"], instance["ner_tags"]):
            ## Identify Tag Type (if present)
            tag_m = tag_map[tag]
            tag_m_type = tag_m.split("-",1)[-1] if tag_m is not None else None
            ## Case 1: No Tag Associated with Current Token
            if tag_m is None:
                ## Case 1a: Existing span needs to end
                if cur_span_start is not None:
                    text_spans.append((cur_span_start, len(text_str)-1, cur_span_type))
                    cur_span_start = None
                    cur_span_type = None
            ## Case 2: Tag Associated with Current Token
            else:
                ## Case 2a: Token starts a new entity
                if tag_m.startswith("B-"):
                    ## Case 2ai: Need to end previous tag
                    if cur_span_start is not None:
                        text_spans.append((cur_span_start, len(text_str)-1, cur_span_type))
                    ## Indicate start of new tag
                    cur_span_start = len(text_str)
                    cur_span_type = tag_m_type
                ## Case 2b: Token continues current span
                elif tag_m.startswith("I-"):
                    pass
                ## Case 3b: Invalid tag
                else:
                    raise ValueError("Invalid tag")
            ## Update String
            text_str += f"{token} "
        ## Add and hanging spans
        if cur_span_start is not None:
            text_spans.append((cur_span_start, len(text_str)-1, cur_span_type))
        ## Remove trailing whitespace
        text_str = text_str[:-1]
        ## Reformat Text Spans
        text_spans_fmt = []
        for ts in text_spans:
            text_spans_fmt.append({
                "label":"X1 - Emerging Entity",
                "start":ts[0],
                "end":ts[1],
                "severity_type":ts[2].title(),
                "laterality":"N/A",
                "status":"N/A",
                "valid":True,
                "in_header":False,
                "in_autolabel_postprocess":False
            })
        ## Updated Document ID
        hashed_document_id = apply_hash("{}-{}".format(instance["id"], split))
        ## Cache
        formatted_instances.append({
            "document_id":hashed_document_id,
            "text":text_str,
            "labels":text_spans_fmt,
            "metadata":{
                "split":split,
                "source":"wnut_2017",
                "user_id":random_seed.randint(0, 150)
            }
        })

## Cache
annotation_cache_dir = "./data/annotations/synthetic-wnut_2017/processed/"
if not os.path.exists(annotation_cache_dir):
    _ = os.makedirs(annotation_cache_dir)
with open(f"{annotation_cache_dir}/annotations.formatted.json", "w") as the_file:
    for example in formatted_instances:
        the_file.write(f"{json.dumps(example)}\n")

## Flatten
formatted_instances_flat = []
for instance in formatted_instances:
    for span in instance["labels"]:
        formatted_instances_flat.append({
            "document_id":instance["document_id"],
            "text":instance["text"][span["start"]:span["end"]],
            "start":span["start"],
            "end":span["end"],
            "severity_type":span["severity_type"]
        })
formatted_instances_flat = pd.DataFrame(formatted_instances_flat)

## Cache
_ = formatted_instances_flat.to_csv(f"{annotation_cache_dir}/annotations.formatted.spans.csv", index=False)


