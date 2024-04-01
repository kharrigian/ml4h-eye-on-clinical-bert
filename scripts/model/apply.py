
"""
Note: This has been setup currently for the separate entity approach. No
guarantees that this will work with other task setups. 
"""

#######################
### Imports
#######################

## Standard Libary
import os
import sys
import json
import argparse
from copy import deepcopy

## External
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

## Local
from cce.util.helpers import flatten
from cce.util import labels as label_utils
from cce.model.datasets import EntityAttributeDataset
from cce.model.architectures import NERTaggerModel
from cce.model.train_utils import get_device
from cce.model.tokenizers import AttributeTaggingTokenizer, PostProcessTokenizer, AutoTokenizer
from cce.model.labelers import LabelEncoder
from cce.model.eval import evaluate, format_entity_predictions

## Shared
_ = sys.path.append(os.path.dirname(__file__))
from train import _separate_entities

#######################
### Functions
#######################

def parse_command_line():
    """

    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--dataset", type=str, nargs="+", default=None)
    _ = parser.add_argument("--model_dir", type=str, default=None)
    _ = parser.add_argument("--output_dir", type=str, default=None)
    _ = parser.add_argument("--rm_existing", action="store_true")
    _ = parser.add_argument("--keep_existing", action="store_true")
    _ = parser.add_argument("--splits", type=str)
    _ = parser.add_argument("--splits_subset", type=str, nargs="+", default=["test"], choices={"train","dev","test"})
    _ = parser.add_argument("--batch_size", type=int, default=16)
    _ = parser.add_argument("--align_inputs_and_outputs", action="store_true", default=False)
    _ = parser.add_argument("--gpu_id", type=int, nargs="*", default=None, help="Which GPU IDs to use.")
    _ = parser.add_argument("--gpu_hold", action="store_true", default=False, help="If true, initialize a tensor to acquire a GPU lock.")
    args = parser.parse_args()
    ## Validate
    if args.dataset is None or len(args.dataset) == 0:
        raise FileNotFoundError("Must provide at least one --dataset file.")
    if not all(os.path.exists(filename) for filename in args.dataset):
        raise FileNotFoundError("Unable to find all provided --dataset files.")
    if args.model_dir is None or not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Unable to find model directory: '{args.model_dir}'")
    if args.output_dir is None:
        raise ValueError("Must provide an --output_dir")
    if os.path.exists(args.output_dir) and not (args.rm_existing or args.keep_existing):
        raise FileExistsError("Please specify --rm_existing or --keep_existing to handle existing output directory.")
    ## Return
    return args

def initialize_gpu_args(args):
    """

    """
    ## Format GPU ID and Claim
    if args.gpu_id is None or not torch.cuda.is_available():
        return args, None
    ## Format ID
    if len(args.gpu_id) > 1:
        args.gpu_id = ",".join(str(i) for i in args.gpu_id)
    else:
        args.gpu_id = args.gpu_id[0]
    ## Initialize Device Hold
    hold = None
    if args.gpu_hold:
        if isinstance(args.gpu_id, str) or args.gpu_id != -1:
            hold = torch.ones(1).to(f"cuda:{args.gpu_id}")
        else:
            hold = torch.ones(1).to(f"cuda")
    ## Return
    return args, hold

def _load_base_dataset_file(dataset_file,
                            lbl2task,
                            attr_label_rename_map,
                            ignore_non_specified=False,
                            separate_negation=None,
                            separate_validity=None,
                            separate_entity=None,
                            exclude_icd=False,
                            exclude_autolabel=False,
                            entity_types_exclude=None):
    """
    
    """
    ## Cache
    metadata = []
    data = []
    ## Open File
    with open(dataset_file,"r") as the_file:
        for line in the_file:
            ## Load Line
            line_data = json.loads(line)
            ## Cache Metadata
            metadata.append({"document_id":line_data["document_id"], **line_data["metadata"]})
            ## Entity Separation
            if separate_entity is not None:
                line_data["labels"] = _separate_entities(labels=line_data["labels"],
                                                         handle=separate_entity)
            ## Rename Labels
            if attr_label_rename_map is not None:
                ## Rename
                for lbl in line_data["labels"]:
                    _ = label_utils.rename_labels(lbl,
                                                  attr_label_rename_map,
                                                  ignore_non_specified=ignore_non_specified,
                                                  separate_negation=separate_negation)
                ## Translate
                labels_updated = []
                for lbl in line_data["labels"]:
                    ## Exclusion Criteria
                    if exclude_icd and lbl.get("in_header",False):
                        continue
                    if exclude_autolabel and lbl.get("in_autolabel_postprocess",False):
                        continue
                    ## Exclusion of Entity Types
                    if entity_types_exclude is not None and any(lbl["label"].startswith(e) for e in entity_types_exclude):
                        continue
                    ## Make Update
                    lbl_updated ={
                        "start":lbl["start"],
                        "end":lbl["end"],
                        "in_header":lbl.get("in_header",False),
                        "in_autolabel_postprocess":lbl.get("in_autolabel_postprocess", False),
                        "valid":lbl["valid"],
                        "label":lbl["label"]
                    }
                    for task_name, task_field in lbl2task[lbl["label"]]:
                        if task_name == "named_entity_recognition":
                            continue
                        if not lbl["valid"]:
                            lbl_updated[task_name] = None
                        else:
                            if task_field == "negation" and lbl["status"] is None:
                                pass
                            else:
                                lbl_updated[task_name] = lbl[task_field]
                    if separate_validity:
                        lbl_updated["validity_{}".format(lbl["label"].replace(" ","_"))] = {True:"valid",False:"invalid"}[lbl["valid"]]
                    labels_updated.append(lbl_updated)
                ## Update Data
                line_data["labels_original"] = line_data["labels"]
                line_data["labels"] = labels_updated
            ## Cache
            data.append(line_data)
    ## Return
    return metadata, data

def load_base_dataset(args,
                      train_config):
    """

    """
    ## Relabeling Map
    task_map = label_utils.CLASSIFIER_TASK_MAP
    ## Label to Task Map
    lbl2task = label_utils.create_task_map(task_map=task_map,
                                           separate_negation=train_config["separate_negation"])
    ## Attribute Relabeling Map
    attr_label_rename_map = label_utils.create_attribute_rename_map(task_map=task_map,
                                                                    ignore_non_specified=train_config["exclude_non_specified"])
    ## Load
    data = []
    metadata =[]
    for filename in args.dataset:
        ## Load
        _metadata, _data = _load_base_dataset_file(dataset_file=filename,
                                                   lbl2task=lbl2task,
                                                   attr_label_rename_map=attr_label_rename_map,
                                                   ignore_non_specified=train_config["exclude_non_specified"],
                                                   separate_negation=train_config["separate_negation"],
                                                   separate_validity=train_config["separate_validity"] is not None,
                                                   separate_entity=train_config["separate_entity"],
                                                   exclude_icd=train_config["exclude_icd"],
                                                   exclude_autolabel=train_config["exclude_autolabel"],
                                                   entity_types_exclude=train_config["entity_types_exclude"])
        ## Cache
        metadata.extend(_metadata)
        data.extend(_data)
    ## Format Metadata
    metadata = pd.DataFrame(metadata)
    ## Load Splits for Filtering
    if args.splits is not None:
        ## Load
        with open(f"{args.splits}","r") as the_file:
            splits = json.load(the_file)
        ## Filter
        if args.splits_subset is not None:
            splits = {s:splits[s] for s in args.splits_subset} 
        ## Reverse Mapping
        doc2split = {}
        for subset, docs in splits.items():
            for doc in docs:
                assert doc not in doc2split
                doc2split[doc] = subset
        ## Assign Splits
        metadata["split"] = metadata["document_id"].map(doc2split.get)
        ## Filter
        data = [data[ind] for ind in (~metadata["split"].isnull()).values.nonzero()[0]]
        metadata = metadata.dropna(subset=["split"])
    ## Return
    return metadata, data

def load_label_encoders(args):
    """

    """
    ## Entitiy Encoder
    entity_encoder = LabelEncoder.load(f"{args.model_dir}/../encoder.entity.joblib")
    ## Attribute Order
    with open(f"{args.model_dir}/../encoder.attribute-order.json","r") as the_file:
        attribute_order = json.load(the_file)["attributes"]
    ## Load Attribute Encoders
    attribute_encoders = {}
    for attribute in attribute_order:
        attribute_encoders[attribute] = LabelEncoder.load(f"{args.model_dir}/../encoder.attribute.{attribute}.joblib")
    return entity_encoder, attribute_encoders

def create_dataset(tokenizer=None,
                   tagging_tokenizer=None,
                   data=None,
                   preprocessed_data=None,
                   entity_encoder=None,
                   attribute_encoders=None,
                   use_labels=True):
    """

    """
    if data is None and preprocessed_data is None:
        raise ValueError("Missing minimal input")
    elif data is not None and preprocessed_data is not None:
        raise ValueError("Ambiguous input")
    elif data is not None:
        ## Preprocessing
        n_split = 0
        preprocessed_data = {"tags":[], "tokens":[], "document_id":[], "split":[], "text":[]}
        for d, datum in tqdm(enumerate(data), total=len(data), desc="[Preprocessing]", file=sys.stdout):
            ## Run Tagger
            tagged_sequences = tagging_tokenizer.tokenize(datum["text"],
                                                          datum["labels"] if use_labels else [],
                                                          cls_token=tokenizer.cls_token,
                                                          sep_token=tokenizer.sep_token)
            n_split += int(len(tagged_sequences) > 1)
            ## Caching + Token Conversion
            for sequence in tagged_sequences:                
                ## Token Conversion
                sequence_token_ids = tokenizer.convert_tokens_to_ids([i[0] for i in sequence])
                ## Caching
                preprocessed_data["tags"].append(sequence)
                preprocessed_data["tokens"].append(sequence_token_ids)
                preprocessed_data["document_id"].append(datum["document_id"])
                preprocessed_data["split"].append(datum["metadata"].get("split",None))
                preprocessed_data["text"].append(datum["text"])
    elif preprocessed_data is not None:
        pass
    ## Labels
    entity_labels, entity_task_tokens, attribute_labels = None, None, None
    if entity_encoder is not None:
        entity_labels, entity_task_tokens = entity_encoder.transform(preprocessed_data["tags"])
    if attribute_encoders is not None:
        attribute_labels = {}
        for attribute, attribute_encoder in attribute_encoders.items():
            attribute_labels[attribute] = attribute_encoder.transform(preprocessed_data["tags"])[0]
    ## Format Dataset
    dataset = EntityAttributeDataset(document_ids=preprocessed_data["document_id"],
                                     token_ids=preprocessed_data["tokens"],
                                     labels_entity=entity_labels,
                                     labels_attributes=attribute_labels,
                                     task_tokens_entity=entity_task_tokens,
                                     encoder_entity=entity_encoder if entity_labels is not None else None,
                                     encoder_attributes=attribute_encoders if attribute_labels is not None else None,
                                     tagged_sequences=preprocessed_data["tags"],
                                     text=preprocessed_data["text"])
    ## Return
    return dataset, preprocessed_data

def initialize_model(args,
                     train_config,
                     vocab2ind,
                     entity_encoder=None,
                     attribute_encoders=None):
    """

    """
    ## Initialize Class
    model = NERTaggerModel(encoder_entity=entity_encoder,
                           encoder_attributes=attribute_encoders,
                           token_encoder=train_config["encoder"],
                           token_vocab=vocab2ind,
                           freeze_encoder=False, ## Doesn't matter for inference
                           use_crf=train_config["use_crf"],
                           use_lstm=train_config["use_lstm"],
                           use_entity_token_bias=train_config["use_entity_token_bias"],
                           entity_token_bias_type=train_config["entity_token_bias_type"],
                           use_attribute_concept_bias=train_config["use_attribute_concept_bias"],
                           max_sequence_length=train_config["max_sequence_length_model"],
                           sequence_overlap=train_config["sequence_overlap_model"],
                           sequence_overlap_type=train_config["sequence_overlap_type_model"],
                           lstm_hidden_size=train_config["lstm_hidden_size"],
                           lstm_num_layers=train_config["lstm_num_layers"],
                           lstm_bidirectional=train_config["lstm_bidirectional"],
                           entity_hidden_size=train_config["entity_hidden_size"],
                           attributes_hidden_size=train_config["attributes_hidden_size"],
                           dropout=train_config["model_dropout_p"], ## Turned off during inference
                           random_state=train_config["random_state"])
    ## Load Parameters
    model_state_dict = model.state_dict()
    pretrained_state_dict = {k:v for k, v in torch.load(f"{args.model_dir}/model.pt", map_location=torch.device('cpu')).items() if k in model_state_dict}
    model_state_dict.update(pretrained_state_dict) ## Overwrite Model Dict Weights with Pretrained Weights
    _ = model.load_state_dict(model_state_dict) ## Re-load Updated Model Dict
    ## Return
    return model

def translate_preprocessed_data(preprocessed_data,
                                separate_entity=True):
    """

    """
    ## Compile by Document ID
    translated_data = {}
    for document_id, tags, in zip(preprocessed_data["document_id"], preprocessed_data["tags"]):
        ## Copy for Safety
        tags = deepcopy(tags)
        ## Get Token Spans (Start/Stop)
        sequence_spans = {}
        for t, (tok, tok_lbls) in enumerate(tags):
            for (lbl, is_start, is_valid) in tok_lbls:
                lbl_key = (lbl["label"], lbl["start"], lbl["end"])
                if is_start:
                    sequence_spans[lbl_key] = []
                sequence_spans[lbl_key].append(t)
        ## Create Conversion Map (Necessary for Character Span Case)
        sequence_spans_conversion = {x:(min(y), max(y)+1) for x, y in sequence_spans.items()}
        ## Apply Conversion
        for t, (tok, tok_lbls) in enumerate(tags):
            for (lbl, is_start, is_valid) in tok_lbls:
                if is_start:
                    lbl_key = (lbl["label"], lbl["start"], lbl["end"])
                    lbl["start"], lbl["end"] = sequence_spans_conversion[lbl_key]
        ## Cache
        if document_id not in translated_data:
            translated_data[document_id] = []
        translated_data[document_id].append(tags)
    ## Merge
    merged_translated_data = []
    for document_id, tags in translated_data.items():
        ## Token Formatting
        tokens = []
        sequence_spans = {}
        tags_offset = 0
        for s, sequence in enumerate(tags):
            ## Get Unique Spans
            for t, (tok, tok_lbls) in enumerate(sequence):
                for lbl, lbl_is_start, lbl_is_valid in tok_lbls:
                    if lbl_is_start:
                        key = (lbl["label"], lbl["start"]-1+tags_offset, lbl["end"]-1+tags_offset)
                        if key in sequence_spans:
                            raise ValueError("This shouldn't happen")
                        sequence_spans[key] = {x:y for x, y in lbl.items() if x not in ["label","start","end"]}
            ## Remove Special Tokens and Cache
            tokens.extend([s[0] for s in sequence[1:-1]])
            ## Update Offset
            tags_offset = len(tokens)
        ## Character Offsets
        text = ""
        tok2text = {}
        for t, token in enumerate(tokens):
            if token.startswith("##"):
                t_start = len(text)
                text = text + token[2:]
            else:
                text = (text + f" {token}").lstrip()
                t_start = len(text) - len(token)
            t_end = len(text)
            tok2text[t] = (t_start, t_end)
        ## Translate Spans 
        sequence_character_spans = []
        for (concept, tok_start, tok_end), lbl in sequence_spans.items():    
            lbl["start"] = tok2text[tok_start][0]
            lbl["end"] = tok2text[tok_end-1][1]
            lbl["label"] = concept
            sequence_character_spans.append(lbl)
        ## Entity Separation Formatting
        if not separate_entity:
            merged_translated_data.append({"document_id":document_id, "text":text, "labels":sequence_character_spans})
        else:
            sequence_character_spans_flat = []
            for span in sequence_character_spans:
                span_keys = list(filter(lambda k: k.startswith("entity_type="), span.keys()))
                span_keys_true = list(filter(lambda sk: span[sk]=="B-True", span_keys))
                span_keys_true = ["entity_type=Z0 - General"] if len(span_keys_true) == 0 else span_keys_true
                for sk in span_keys_true:
                    sk_lbl = {
                        "label":sk[12:],
                        **{x:y for x, y in span.items() if x != "label" and not x.startswith("entity_type=")}
                    }
                    sequence_character_spans_flat.append(sk_lbl)
            merged_translated_data.append({"document_id":document_id, "text":text, "labels":sequence_character_spans_flat})
    ## Return
    return merged_translated_data

def update_entity_predictions(preprocessed_data,
                              entity_output,
                              entity_encoder,
                              attribute_encoders):
    """

    """
    for (datum_ind, tok_start, tok_end, tok_ent_lbl) in entity_output["entity"]["predictions"][1]:
        ent_lbl = {
                "label":entity_encoder.get_tasks()[tok_ent_lbl][0],
                "start":tok_start,
                "end":tok_end,
                "valid":True,
                "in_header":False,
                "in_autolabel_postprocess":False
                }
        _ = ent_lbl.update({x:False for x in attribute_encoders.keys()}) ## Placeholder
        for t, (token, token_lbls) in enumerate(preprocessed_data["tags"][datum_ind][tok_start:tok_end]):
            token_lbls.append((ent_lbl, t==0, True))
    ## Return
    return preprocessed_data

def update_attribute_predictions(preprocessed_data,
                                 attribute_output,
                                 entity_encoder,
                                 attribute_encoders):
    """

    """
    for attribute, attribute_preds in attribute_output["attributes"]["predictions"].items():
        for row in attribute_preds:
            row_doc_ind, row_tok_start, row_tok_end = list(map(int, row[3:6]))
            row_concept = attribute_encoders[attribute]._id2task[int(row[2])][0]
            row_pred = attribute_encoders[attribute]._classes[int(row[0])]
            for (tok, tok_lbls) in preprocessed_data["tags"][row_doc_ind][row_tok_start:row_tok_end]:
                for (lbl, _, _) in tok_lbls:
                    if lbl["label"] != row_concept:
                        continue
                    lbl[attribute] = row_pred
    return preprocessed_data

def _get_window(text,
                start,
                end,
                window_size=10):
    """
    
    """
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)
    ## Update White-space Limits
    while start > 0 and text[start - 1] != " ":
        start -= 1
    while end < len(text) - 1 and text[end] != " ":
        end += 1
    ## Get Left/Right/Center
    text_left = text[:start].strip().split(" ")[-window_size[0]:]
    text_span = text[start:end].strip().split(" ")
    text_right = text[end:].strip().split(" ")[:window_size[1]]
    ## Merge
    text_window = " ".join(text_left + text_span + text_right)
    return text_window

def _compare_instances(predicted,
                       labeled):
    """

    """
    ## Text
    assert predicted["document_id"] == labeled["document_id"]
    assert predicted["text"] == labeled["text"]
    ## Labeled Spans
    labeled_grps = {}
    for span in labeled["labels"]:
        if span["label"] not in labeled_grps:
            labeled_grps[span["label"]] = {}
        labeled_grps[span["label"]][(span["start"], span["end"])] = []
    ## Predicted
    predicted_grps = {}
    for span in predicted["labels"]:
        if span["label"] not in predicted_grps:
            predicted_grps[span["label"]] = {}
        predicted_grps[span["label"]][(span["start"], span["end"])] = []
    ## Overlap Beteen Labeled and Predicted
    for lgroup, lspans in labeled_grps.items():
        ## Base Case: No Label Overlap with Predicted
        if lgroup not in predicted_grps:
            continue
        for (lbl_span_start, lbl_span_end) in lspans.keys():
            for span_start, span_end in predicted_grps[lgroup]:
                if lbl_span_start <= span_end and lbl_span_end >= span_start:
                    labeled_grps[lgroup][(lbl_span_start, lbl_span_end)].append((span_start, span_end))
    ## Overlap Betwen Predicted and Labeled
    for pgroup, pspans in predicted_grps.items():
        ## Base Case: No Label Overlap with Labels
        if pgroup not in labeled_grps:
            continue
        ## Look for Overlap
        for (span_start, span_end) in pspans.keys():
            for lbl_span_start, lbl_span_end in labeled_grps[pgroup]:
                if span_start <= lbl_span_end and span_end >= lbl_span_start:
                    predicted_grps[pgroup][(span_start, span_end)].append((lbl_span_start, lbl_span_end))
    ## Build Relationship Graph
    g = nx.Graph()
    for lgroup in labeled_grps.keys():
        for labeled_span, predicted_spans in labeled_grps[lgroup].items():
            g.add_node((lgroup, "labeled", *labeled_span))
            for pspan in predicted_spans:
                g.add_edge((lgroup, "labeled", *labeled_span), (lgroup, "predicted", *pspan))
    for pgroup in predicted_grps.keys():
        for predicted_span, labeled_spans in predicted_grps[pgroup].items():
            g.add_node((pgroup, "predicted", *predicted_span))
            for lspan in labeled_spans:
                g.add_edge((pgroup, "predicted", *predicted_span), (pgroup, "labeled", *lspan))
    ## Identify Entity Groups
    combined_groups = list(nx.connected_components(g))
    ## Format
    formatted = []
    for group in combined_groups:
        group = list(group)
        lgroup = sorted([i for i in group if i[1] == "labeled"], key=lambda x: x[2])
        pgroup = sorted([i for i in group if i[1] == "predicted"], key=lambda x: x[2])        
        union = (min(l[2] for l in group), max(l[3] for l in group))
        formatted.append({
            "document_id":labeled["document_id"],
            "label":group[0][0],
            "start_union":union[0],
            "end_union":union[1],
            "text_union":labeled["text"][union[0]:union[1]],
            "text_union_context":_get_window(labeled["text"], union[0], union[1], window_size=10),
            "labeled":[(l[2], l[3], labeled["text"][l[2]:l[3]]) for l in lgroup],
            "predicted":[(p[2], p[3], predicted["text"][p[2]:p[3]]) for p in pgroup],
        })
    ## Return
    return formatted

def compare_labeled_and_predicted(preprocessed_data_translated,
                                  labeled_preprocessed_data_translated):
    """

    """
    ## Check Lengths
    assert len(preprocessed_data_translated) == len(labeled_preprocessed_data_translated)
    ## Iterate Through Instances
    comparison = []
    for predicted, labeled in zip(preprocessed_data_translated, labeled_preprocessed_data_translated):
        comparison.append(_compare_instances(predicted=predicted,
                                             labeled=labeled))
    ## Return
    return comparison

def main():
    """

    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## GPU Init
    print("[Setting Up Compute Resources]")
    args, gpu_hold = initialize_gpu_args(args=args)
    ## Output Directory
    print("[Initializing Output Directory]")
    if os.path.exists(args.output_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.output_dir}")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Load Model Parameters
    print("[Loading Model Training Parameters]")
    with open(f"{args.model_dir}/train.cfg.json","r") as the_file:
        train_config = json.load(the_file)
    ## Generate Dataset
    print("[Loading Base Dataset]")
    metadata, data = load_base_dataset(args,
                                       train_config=train_config)
    ## Initialize Tokenizers
    print("[Initializing Tokenizers]")
    tokenizer = PostProcessTokenizer(tokenizer=AutoTokenizer.from_pretrained(train_config["tokenizer"]))
    tagging_tokenizer = AttributeTaggingTokenizer(tokenizer.tokenize,
                                                  max_sequence_length=train_config["max_sequence_length_input"] if train_config["split_input"] else None,
                                                  sequence_overlap=train_config["sequence_overlap_input"] if train_config["split_input"] else None,
                                                  split_type=train_config["sequence_split_type_input"])
    ## Load Encoders
    print("[Loading Label Encoders]")
    entity_encoder, attribute_encoders = load_label_encoders(args)
    ## Load Model
    print("[Loading Model]")
    model = initialize_model(args,
                             train_config=train_config,
                             vocab2ind=tokenizer.get_vocab(),
                             entity_encoder=entity_encoder,
                             attribute_encoders=attribute_encoders)
    ## Entity Dataset
    print("[Initializing Entity Discovery Dataset]")
    entity_dataset, preprocessed_data = create_dataset(tokenizer=tokenizer,
                                                       tagging_tokenizer=tagging_tokenizer,
                                                       data=data,
                                                       preprocessed_data=None,
                                                       entity_encoder=entity_encoder,
                                                       attribute_encoders=attribute_encoders,
                                                       use_labels=False)
    ## Run Entity Discovery
    print("[Beginning Entity Discovery]")
    entity_output = evaluate(model=model.set_encoder_switch(entity=True, attributes=False),
                             dataset=entity_dataset,
                             batch_size=args.batch_size,
                             entity_weights=None,
                             attribute_weights=None,
                             desc="Entity Discovery",
                             device=get_device(args.gpu_id),
                             use_first_index=train_config["attribute_use_first_index"],
                             return_predictions=True)
    ## Update Preprocessed Data
    print("[Assigning Entity Predictions to Preprocessed Data]")
    preprocessed_data = update_entity_predictions(preprocessed_data=preprocessed_data,
                                                  entity_output=entity_output,
                                                  entity_encoder=entity_encoder,
                                                  attribute_encoders=attribute_encoders)
    ## Attribute Classification Data
    print("[Initializing Attribute Classification Dataset]")
    attribute_dataset, preprocessed_data = create_dataset(tokenizer=tokenizer,
                                                          tagging_tokenizer=tagging_tokenizer,
                                                          data=None,
                                                          preprocessed_data=preprocessed_data,
                                                          entity_encoder=entity_encoder,
                                                          attribute_encoders=attribute_encoders,
                                                          use_labels=False)
    ## Run Attribute Classification
    print("[Beginning Attribute Classification]")
    attribute_output = evaluate(model=model.set_encoder_switch(entity=False, attributes=True),
                                dataset=attribute_dataset,
                                batch_size=args.batch_size,
                                entity_weights=None,
                                attribute_weights=None,
                                desc="Attribute Classification",
                                device=get_device(args.gpu_id),
                                use_first_index=train_config["attribute_use_first_index"],
                                return_predictions=True)
    ## Update Preprocessed Dataset With Predictions
    print("[Assigning Attribute Predictions to Preprocessed Data]")
    preprocessed_data = update_attribute_predictions(preprocessed_data=preprocessed_data,
                                                     attribute_output=attribute_output,
                                                     entity_encoder=entity_encoder,
                                                     attribute_encoders=attribute_encoders)
    ## Translate Preprocessed Dataset, Align
    print("[Translating Predictions]")
    preprocessed_data_translated = translate_preprocessed_data(preprocessed_data,
                                                               separate_entity=train_config["separate_entity"] is not None)
    ## Cache
    print("[Caching Predictions]")
    with open(f"{args.output_dir}/tags.predictions.json","w") as the_file:
        for instance in preprocessed_data_translated:
            the_file.write(json.dumps(instance) + "\n")
    ## Alignment
    if args.align_inputs_and_outputs:
        print("[Initializing Existing Labeled Dataset for Alignment]")
        _, labeled_preprocessed_data = create_dataset(tokenizer=tokenizer,
                                                      tagging_tokenizer=tagging_tokenizer,
                                                      data=data,
                                                      preprocessed_data=None,
                                                      entity_encoder=None,
                                                      attribute_encoders=None,
                                                      use_labels=True)
        ## Translate
        print("[Translating Labels]")
        labeled_preprocessed_data_translated = translate_preprocessed_data(labeled_preprocessed_data,
                                                                           separate_entity=False)
        ## Compare
        print("[Running Label/Prediction Alignment]")
        aligned_comparison = compare_labeled_and_predicted(preprocessed_data_translated=preprocessed_data_translated,
                                                           labeled_preprocessed_data_translated=labeled_preprocessed_data_translated)
        ## Format for NER
        print("[Formatting Output NER Annotations]")
        aligned_comparison_l = []
        for document, document_labels_updated in zip(preprocessed_data_translated, aligned_comparison):
            document_c = deepcopy(document)
            document_c["labels"] = [
                {
                    "label":l["label"],
                    "start":l["start_union"],
                    "end":l["end_union"],
                    "valid":True,
                    "in_autolabel_postprocess":False,
                    "in_header":False
                } for l in document_labels_updated
            ]
            aligned_comparison_l.append(document_c)
        ## DataFrame Output Format
        print("[Formatting Output DataFrame]")
        aligned_comparison_df = pd.DataFrame(flatten(aligned_comparison))
        aligned_comparison_df["span_is_novel"] = aligned_comparison_df["labeled"].map(len) == 0
        aligned_comparison_df["span_is_missed"] = aligned_comparison_df["predicted"].map(len) == 0
        aligned_comparison_df["span_is_exact"] = aligned_comparison_df.apply(lambda row: len(row["labeled"])==len(row["predicted"])==1 and row["labeled"][0][:2]==row["predicted"][0][:2], axis=1)
        aligned_comparison_df["span_is_partial"] = aligned_comparison_df.apply(lambda row: len(row["labeled"]) > 0 and len(row["predicted"]) > 0 and not row["span_is_exact"], axis=1)
        aligned_comparison_df["span_is_expanded"] = aligned_comparison_df.apply(lambda row: len(row["labeled"])>0 and (len(row["labeled"])>1 or row["labeled"][0][0]<row["start_union"] or row["labeled"][0][1]>row["end_union"]), axis=1)
        ## Cache
        print("[Caching Aligned]")
        with open(f"{args.output_dir}/tags.labels-predictions.json","w") as the_file:
            for instance in aligned_comparison_l:
                the_file.write(json.dumps(instance) + "\n")
        _ = aligned_comparison_df.to_json(f"{args.output_dir}/comparison.json",orient="records",indent=1)
    ## Done
    print("[Script Complete]")

####################
### Execute
####################

if __name__ == "__main__":
    _ = main()