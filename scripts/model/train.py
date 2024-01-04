
"""

"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import argparse
from math import gcd
from glob import glob
from collections import Counter

## External
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

## Internal
from cce.util import labels as label_utils
from cce.model import eval
from cce.model.datasets import EntityAttributeDataset
from cce.model.architectures import NERTaggerModel
from cce.model.train_utils import train, train_baseline, get_device, initialize_class_weights
from cce.model.tokenizers import AttributeTaggingTokenizer, PostProcessTokenizer, AutoTokenizer
from cce.model.labelers import LabelEncoder, display_task_summary, display_data_summary, apply_attribute_limit

#######################
### Globals
#######################

## Configuration Changes That Prevent Model Warm Starts
WARM_START_BREAKING_KEYS = [
    "encoder",
    "tokenizer",
    "merge_subwords",
    "vocab_min_freq",
    "entity_key",
    "attribute_keys",
    "include_entity",
    "include_attribute",
    "split_input",
    "max_sequence_length_model",
    "sequence_overlap_model",
    "sequence_overlap_type_model",
    "use_crf",
    "use_lstm",
    "use_entity_token_bias",
    "use_attribute_concept_bias",
    "entity_token_bias_type",
    "lstm_hidden_size",
    "lstm_bidirectional",
    "lstm_num_layers",
    "entity_hidden_size",
    "attributes_hidden_size",
    "exclude_non_specified",
    "separate_negation",
]

## Specific Layer Breaking Keys
SPECIFIC_WARM_START_BREAKING_KEYS = {
    "encoder":[
        "encoder",
        "tokenizer",
        "merge_subwords",
        "vocab_min_freq",
    ],
    "combiner":[
        "encoder",
        "lstm_hidden_size",
        "lstm_bidirectional",
        "lstm_num_layers",
    ],
    "entity_heads":[
        "encoder",
        "tokenizer",
        "entity_key",
        "use_lstm",
        "use_entity_token_bias",
        "entity_token_bias_type",
        "lstm_hidden_size",
        "lstm_bidirectional",
        "lstm_num_layers",
        "entity_hidden_size",
    ],
    "entity_crf":[

    ],
    "attribute_heads":[
        "encoder",
        "tokenizer",
        "use_lstm",
        "lstm_hidden_size",
        "lstm_bidirectional",
        "lstm_num_layers",
        "attributes_hidden_size",
        "attribute_keys",
        "separate_negation",
        "exclude_non_specified",
        "use_attribute_concept_bias"
    ]
}

#######################
### Functions
#######################

def parse_command_line():
    """
    
    """
    ## Parse Command Line
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--datasets",type=str,nargs="+",default=None,help="Path to input JSON file(s) with annotations.")
    _ = parser.add_argument("--encoder", type=str, default="bert-base-cased",help="Which token encoder to use. Should align with tokenizer.")
    _ = parser.add_argument("--tokenizer",type=str,default="bert-base-cased",help="Which tokenizer to use. Should align with embeddings.")
    _ = parser.add_argument("--split_input", action="store_true", default=False, help="If included, will split input instances by length before passing the into the model.")
    _ = parser.add_argument("--max_sequence_length_input", type=int, default=None, help="Maximimum number of tokens in a single example. Default, no splitting prior to input into model.")
    _ = parser.add_argument("--max_sequence_length_model", type=int, default=512, help="Maximum number of tokens processed by model encoder at a time.")
    _ = parser.add_argument("--sequence_overlap_input", type=int, default=None, help="If desired, will include this much left-hand overlap when splitting long examples into sequences.")
    _ = parser.add_argument("--sequence_overlap_model", type=int, default=None, help="If desired, will include this much left-hand overlap when splitting long sequences into encoder inputs..")
    _ = parser.add_argument("--sequence_overlap_type_model", type=str, default="mean", choices={"first","last","mean"}, help="How to combine overlap within the encoder inputs.")
    _ = parser.add_argument("--sequence_split_type_input", type=str, default="centered", choices={"continuous","centered"}, help="How to split sequences relative to spans.")
    _ = parser.add_argument("--entity_key",type=str,default="label",help="Which label should be used as the primary entity type identifer.")
    _ = parser.add_argument("--attribute_keys",type=str,nargs="*",default="default",help="Attribute token classification tasks.")
    _ = parser.add_argument("--include_entity", action="store_true", default=False, help="Whether to model entities (NER)")
    _ = parser.add_argument("--include_attribute", action="store_true", default=False, help="Whether to model attributes.")
    _ = parser.add_argument("--cache_label_encoders", action="store_true", default=False, help="If included, store the dataset label encoders.")
    _ = parser.add_argument("--limit_spans_per_document", type=int, default=None, help="If included, restrict to maximimally this many spans per attribute per document.")
    _ = parser.add_argument("--limit_documents_per_group", type=int, default=None, help="If included, restrict to maximally this many documents per group.")
    _ = parser.add_argument("--limit_spans_per_group", type=int, default=None, help="If include, restrict to maximally this many spans per group.")
    _ = parser.add_argument("--limit_spans_per_document_label_stratify", action="store_true", default=False, help="If include, count spans in a document by label instead of overall")
    _ = parser.add_argument("--limit_spans_per_group_label_stratify", action="store_true", default=False, help="If include, count spans in a group by label instead of overall")
    _ = parser.add_argument("--limit_group", type=str, nargs="+", default=["patient_id"], help="Which metadata attributes to use as grouping variables.")
    _ = parser.add_argument("--model_init", type=str, default=None, help="If desired, can provide a cached model to use as the starting point of training.")
    _ = parser.add_argument("--model_init_ignore_fold", action="store_true", default=False, help="By default, we assume the model_init folds align with what is being trained.")
    _ = parser.add_argument("--model_init_reset_training", action="store_true", default=False, help="If including a model_init to initialize weights but you don't want to continue training.")
    _ = parser.add_argument("--model_lr", type=float, default=1e-5,help="Model learning rate")
    _ = parser.add_argument("--model_lr_adaptive_method", type=str, default=None, help="Include if you want to use an adapative learning rate scheduler.", choices={"step","exponential","cosine_annealing_restart"})
    _ = parser.add_argument("--model_lr_warmup", type=int, default=None, help="Whether to wait a warmup number of steps before modifying learning rate.")
    _ = parser.add_argument("--model_lr_step_size", type=int, default=50, help="Number of steps between learning rate modification.")
    _ = parser.add_argument("--model_lr_gamma", type=float, default=0.9, help="Adaptive decrease factor.")
    _ = parser.add_argument("--model_opt_method", type=str, default="adamw", choices={"adam","adamw","sgd"}, help="Model optimizer.")
    _ = parser.add_argument("--model_nesterov", action="store_true", default=False, help="If using SGD, whether to use nesterov momentum.")
    _ = parser.add_argument("--model_momentum", type=float, default=0.9, help="Momentum level.")
    _ = parser.add_argument("--model_weight_decay", type=float, default=1e-5,help="Regularization strength")
    _ = parser.add_argument("--model_grad_clip_norm", type=float, default=None, help="If include, applies gradient clipping with this max norm.")
    _ = parser.add_argument("--model_dropout_p", type=float, default=0.1, help="Dropout probability")
    _ = parser.add_argument("--model_n_epochs", type=int, default=None, help="Number of training epochs")
    _ = parser.add_argument("--model_n_steps", type=int, default=None, help="Number of training steps.")
    _ = parser.add_argument("--model_train_batch_size", type=int, default=16,help="Training batch size")
    _ = parser.add_argument("--model_train_gradient_accumulation", type=int, default=None, help="How many batches to accumulate gradients for before taking an optimization step.")
    _ = parser.add_argument("--model_eval_batch_size", type=int, default=16,help="Evaluation batch size (can be larger than training since no gradients)")
    _ = parser.add_argument("--model_eval_frequency", type=int, default=1, help="Number of update steps between evaluation subroutines")
    _ = parser.add_argument("--model_eval_strategy", type=str, default="epochs", choices={"epochs","steps"},help="Whether evaluation frequency is interpreted as steps or epochs")
    _ = parser.add_argument("--model_save_frequency", type=int, default=1, help="Number of evaluations to run before saving model.")
    _ = parser.add_argument("--early_stopping_tol", type=float, default=0.001, help="Relative loss decrease requirement.")
    _ = parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of evaluations before deeming the loss as stagnant/increasing.")
    _ = parser.add_argument("--early_stopping_warmup", type=int, default=None, help="If included, wait this many steps before checking early stopping criteria.")
    _ = parser.add_argument("--weighting_entity", type=str, default=None, choices={"balanced"}, help="If desired, what type of class weighting to use for entity loss.")
    _ = parser.add_argument("--weighting_attribute", type=str, default=None, choices={"balanced"}, help="If deisred, what type of class weighting to use for attribute loss.")
    _ = parser.add_argument("--use_crf", action="store_true", default=False,help="If included, will stack a CRF on-top of linear entity heads")
    _ = parser.add_argument("--use_lstm", action="store_true", default=False, help="If included, will stack an LSTM on top-of encoding..")
    _ = parser.add_argument("--use_entity_token_bias", action="store_true", default=False, help="Whether to bias entity models with knowledge of regex token matches.")
    _ = parser.add_argument("--entity_token_bias_type", type=str, choices={"uniform","positional"}, default="uniform", help="Uniform bias just indicates that a match is present. Positional indicates whether it's the first token in the matched span.")
    _ = parser.add_argument("--use_attribute_concept_bias", action="store_true", default=False, help="Whether to indicate which concept is present during attribute classification.")
    _ = parser.add_argument("--lstm_hidden_size", type=int, default=768, help="Hidden layer size for LSTM")
    _ = parser.add_argument("--lstm_bidirectional", action="store_true", default=False, help="If included, will use bidirectional LSTM output.")
    _ = parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of LSTM layers.")
    _ = parser.add_argument("--entity_hidden_size", type=int, default=None, help="Hidden layer size for entity classification head.")
    _ = parser.add_argument("--attributes_hidden_size", type=int, default=None, help="Hidden layer size for attribute classification heads.")
    _ = parser.add_argument("--freeze_encoder", action="store_true", default=False, help="If included, do not run backprop on the token encoder.")
    _ = parser.add_argument("--attribute_use_first_index", action="store_true", default=False, help="If included, don't pool the full entity. Instead use first token as representation.")
    _ = parser.add_argument("--exclude_non_specified", action="store_true", default=False,help="If true, don't model unspecified statuses. Treat as missing data instead.")
    _ = parser.add_argument("--exclude_icd", action="store_true", default=False, help="If included, will not use ICD labeled spans for training or evaluation.")
    _ = parser.add_argument("--exclude_autolabel", action="store_true", default=False, help="If included, will not include entity labels which were added based on autolabeler without human review.")
    _ = parser.add_argument("--separate_negation", type=str, nargs="*", default=None, help="If True, model negation independently from status.")
    _ = parser.add_argument("--separate_validity", type=str, nargs="*", default=None, help="If True, models each entity's validity outside of the NER context.")
    _ = parser.add_argument("--random_state", type=int, default=42,help="Random seed for sampling/training/initialization")
    _ = parser.add_argument("--eval_cv", type=int, default=5, help="If desired, can set-up K-fold cross validation. This specifies the K. Otherwise, uses presampled train/dev/test split.")
    _ = parser.add_argument("--eval_cv_groups", type=str, nargs="*", default=None, help="If doing cross-validation, use this to specify which document metadata attributes should be used for splitting.")
    _ = parser.add_argument("--eval_cv_fold", type=int, nargs="*", default=None, help="If doing K-Fold cross-validation, you can specify which folds you want to run on.")
    _ = parser.add_argument("--eval_mc_split_frac", type=int, nargs=3, default=[7, 2, 1], help="If using monte carlo cross-validation (stratified or default), specify train, dev, test ratios.")
    _ = parser.add_argument("--eval_train", action="store_true", default=False, help="If included, run evaluation on training data.")
    _ = parser.add_argument("--eval_test", action="store_true", default=False,help="If included, run evaluation on test data")
    _ = parser.add_argument("--eval_strategy", type=str, default="k_fold", choices={"k_fold","monte_carlo","stratified_k_fold","stratified_monte_carlo","redundant"})
    _ = parser.add_argument("--output_dir", type=str, default=None,help="Where to store outputs/models")
    _ = parser.add_argument("--rm_existing", action="store_true", default=False,help="If using an existing output directory, must include to overwrite.")
    _ = parser.add_argument("--keep_existing", action="store_true", default=False,help="If using an existing output directory, must include to keep data that already exists.")
    _ = parser.add_argument("--no_training", action='store_true', default=False, help="Load an existing checkpoint / training logs and generate plots without additional training.")
    _ = parser.add_argument("--display_batch_loss", action="store_true", default=False, help="Whether to print cumulative batch loss.")
    _ = parser.add_argument("--gpu_id", type=int, nargs="*", default=None, help="Which GPU IDs to use.")
    _ = parser.add_argument("--gpu_hold", action="store_true", default=False, help="If true, initialize a tensor to acquire a GPU lock.")
    _ = parser.add_argument("--evaluate", action="store_true", default=False, help="If included, run evaluation without training.")
    _ = parser.add_argument("--baseline_only", action="store_true", default=False, help="If included, only run baseline training procedure.")
    _ = parser.add_argument("--baseline_use_char", action="store_true", default=False, help="If included, token baseline will actually just use raw regex text.")
    args = parser.parse_args()
    ## Check Arguments
    if not args.include_entity and not args.include_attribute:
        raise ValueError("Must include at least one of: --include_entity, --include_attribute")
    if args.rm_existing and args.keep_existing:
        raise ValueError("Cannot ask to --rm_existing and --keep_existing.")
    if args.datasets is None or any(not os.path.exists(f) for f in args.datasets):
        raise FileNotFoundError("Must provide a valid dataset path (--dataset)")
    if args.output_dir is None:
        raise FileNotFoundError("Must provide an output directory (--output_dir)")
    if args.model_init is not None and not os.path.exists(args.model_init):
        raise FileNotFoundError(f"Missing required file to perform model initialization: {args.model_init}")
    if args.no_training and args.model_init is None:
        raise ValueError("If generating plots only (--no_training), must provide a model checkpoint (--model_init).")
    if args.evaluate and args.model_init is None:
        raise ValueError("If evaluating a model, you must provide a --model_init.")
    if args.sequence_split_type_input == "centered" and args.include_entity:
        raise ValueError("You can only use 'continuous' for sequence_split_type_input if training an NER model.")
    if not args.baseline_only and (args.model_n_steps is None and args.model_n_epochs is None):
        raise ValueError("Must specify either --model_n_steps or --model_n_epochs.")
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

def _load_annotations(annotation_file,
                      lbl2task,
                      attr_label_rename_map=None,
                      ignore_non_specified=False,
                      separate_negation=None,
                      separate_validity=None,
                      exclude_icd=False,
                      exclude_autolabel=False):
    """
    
    """
    ## Cache
    metadata = []
    data = []
    ## Open File
    with open(annotation_file,"r") as the_file:
        for line in the_file:
            ## Load Line
            line_data = json.loads(line)
            ## Cache Metadata
            metadata.append({"document_id":line_data["document_id"], **line_data["metadata"]})
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

def _sample_splits_redundant(preprocessed_data,
                             **kwargs):
    """
    
    """
    document_ids = preprocessed_data["document_id"]
    splits = {0:{
        "train":set(document_ids),
        "dev":set(document_ids),
        "test":set(document_ids)
        }
    }
    return splits

def _sample_splits_monte_carlo(preprocessed_data,
                               metadata=None,
                               eval_cv=10,
                               eval_cv_groups=None,
                               split_frac=[7,2,1],
                               random_state=42,
                               verbose=True):
    """

    """
    ## Format Split Frac
    if not all(isinstance(i, int) for i in split_frac):
        raise ValueError("Expected integer ratios for split_frac")
    split_frac_sum = sum(split_frac)
    split_frac = [x / split_frac_sum for x in split_frac]
    ## Get Groups of Documents Which Should Be Kept Together
    if eval_cv_groups is None:
        document_groups = {x:i for i, x in enumerate(preprocessed_data["document_id"])}
    else:
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        metadata = metadata.loc[metadata["document_id"].isin(set(preprocessed_data["document_id"]))].copy()
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple,axis=1).to_dict()
    ## Reverse Document Groups
    document_groups_r = {}
    for x, y in document_groups.items():
        if y not in document_groups_r:
            document_groups_r[y] = set()
        document_groups_r[y].add(x)
    if verbose:
        print("[Note: Found {:,d} Groups for {:,d} Documents]".format(len(document_groups_r), len(document_groups)))
    groups = sorted(document_groups_r.keys())
    ## Run Sampling
    seed = np.random.RandomState(random_state)
    splits = {}
    for k in range(eval_cv):
        ## Sample Assignments to Train/Dev/Test Set
        fold_assignments = seed.choice(["train","dev","test"], len(groups), replace=True, p=split_frac)
        ## Group Assignments
        train_g = sorted([groups[ind] for ind in (fold_assignments == "train").nonzero()[0]])
        dev_g = sorted([groups[ind] for ind in (fold_assignments == "dev").nonzero()[0]])
        test_g = sorted([groups[ind] for ind in (fold_assignments == "test").nonzero()[0]])
        ## Translate to Data Indices
        train_doc = set.union(*[document_groups_r[g] for g in train_g])
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g])
        test_doc = set.union(*[document_groups_r[g] for g in test_g])
        ## Check Mutual Exclusion
        assert len(train_doc & dev_doc) == 0
        assert len(train_doc & test_doc) == 0
        assert len(dev_doc & test_doc) == 0        
        ## Store
        splits[k] = {"train":train_doc, "dev":dev_doc, "test":test_doc}
    ## Return
    return splits

def _sample_splits(preprocessed_data,
                   metadata=None,
                   eval_cv=5,
                   eval_cv_groups=None,
                   random_state=42,
                   verbose=True):
    """

    """
    ## Get Groups of Documents Which Should Be Kept Together
    if eval_cv_groups is None:
        document_groups = {x:i for i, x in enumerate(preprocessed_data["document_id"])}
    else:
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        metadata = metadata.loc[metadata["document_id"].isin(set(preprocessed_data["document_id"]))].copy()
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple,axis=1).to_dict()
    ## Reverse Document Groups
    document_groups_r = {}
    for x, y in document_groups.items():
        if y not in document_groups_r:
            document_groups_r[y] = set()
        document_groups_r[y].add(x)
    if verbose:
        print("[Note: Found {:,d} Groups for {:,d} Documents]".format(len(document_groups_r), len(document_groups)))
    ## Generate Groups and Fold Assignments
    seed = np.random.RandomState(random_state)
    groups = sorted(document_groups_r.keys())
    groups = sorted(groups, key=lambda x: seed.random())
    groups_assign = np.array([i % eval_cv for i, g in enumerate(groups)])
    ## Construct Splits
    splits = {}
    for k in range(eval_cv):
        ## K Fold Assignments (1 Test, 1 Dev, eval_cv - 2 Train)
        k_test = k
        k_dev = (k_test - 1) % eval_cv
        k_train = [i % eval_cv for i in range(k_dev - (eval_cv - 2), k_dev)]
        ## Group Assignments
        test_g = sorted([groups[ind] for ind in (groups_assign == k_test).nonzero()[0]])
        dev_g = sorted([groups[ind] for ind in (groups_assign == k_dev).nonzero()[0]])
        train_g = sorted(groups[ind] for ind in np.logical_or.reduce([groups_assign == f for f in k_train]).nonzero()[0])
        ## Translate to Data Indices
        train_doc = set.union(*[document_groups_r[g] for g in train_g])
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g])
        test_doc = set.union(*[document_groups_r[g] for g in test_g])
        ## Check Mutual Exclusion
        assert len(train_doc & dev_doc) == 0
        assert len(train_doc & test_doc) == 0
        assert len(dev_doc & test_doc) == 0
        ## Store
        splits[k] = {"train":train_doc, "dev":dev_doc, "test":test_doc}
    ## Return
    return splits

def _sample_splits_stratified(preprocessed_data,
                              data,
                              metadata,
                              eval_cv=5,
                              eval_cv_groups=None,
                              max_sample_per_iter=None,
                              random_state=42,
                              verbose=True):
    """

    """
    ## Valdiate
    if max_sample_per_iter is None:
        max_sample_per_iter = eval_cv
    if max_sample_per_iter < eval_cv:
        raise ValueError("Must specify a max_sample_per_iter of at least the number of folds.")
    ## Get Groups of Documents Which Should Be Kept Together
    if eval_cv_groups is None:
        document_groups = {x:i for i, x in enumerate(preprocessed_data["document_id"])}
    else:
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        metadata = metadata.loc[metadata["document_id"].isin(set(preprocessed_data["document_id"]))].copy()
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple,axis=1).to_dict()
    ## Reverse Document Groups
    document_groups_r = {}
    for x, y in document_groups.items():
        if y not in document_groups_r:
            document_groups_r[y] = set()
        document_groups_r[y].add(x)
    if verbose:
        print("[Note: Found {:,d} Groups for {:,d} Documents]".format(len(document_groups_r), len(document_groups)))
    ## Translate Data to Label DataFrame
    document_labels_oh = []
    for datum in data:
        datum_lbl_counts = Counter()
        for lbl in datum["labels"]:
            for key, value in lbl.items():
                if key in ["start","end","in_header","in_autolabel_postprocess","valid","label"]:
                    continue
                if pd.isnull(value):
                    continue
                datum_lbl_counts[(key, lbl["label"], value)] += 1
        document_labels_oh.append({"document_id":datum["document_id"], **datum_lbl_counts})
    document_labels_oh = pd.DataFrame(document_labels_oh).set_index("document_id").fillna(0).astype(int)
    ## Aggregate by Group
    document_labels_oh["document_group"] = document_labels_oh.index.map(document_groups.get)
    group_labels_oh = document_labels_oh.groupby(["document_group"]).sum()
    group_labels_oh = (group_labels_oh > 0).astype(int)
    group_labels_oh = group_labels_oh.sort_index(0).sort_index(1)
    ## Isolate Groups Without any Labels
    groups_no_lbl = group_labels_oh.loc[group_labels_oh.sum(axis=1) == 0].index.tolist()
    group_labels_oh = group_labels_oh.loc[~group_labels_oh.index.isin(groups_no_lbl)].copy()
    ## Assignments
    unassigned = set(group_labels_oh.index.tolist())
    assignments = {fold:set() for fold in range(eval_cv)}
    assignment_counts = {target:{fold:0 for fold in range(eval_cv)} for target in group_labels_oh.columns.tolist()}
    seed = np.random.RandomState(random_state)
    while len(unassigned) > 0:
        ## Find the Label With the Smallest Sample Size
        smallest = group_labels_oh.loc[group_labels_oh.index.isin(unassigned)].sum(axis=0)
        smallest = smallest.loc[smallest>0].nsmallest(1)
        ## Identify Candidates with That Label and Shuffle Randomly
        candidates = group_labels_oh.loc[(group_labels_oh.index.isin(unassigned))&
                                         (group_labels_oh[smallest.index.item()] > 0)].index.tolist()
        candidates = sorted(candidates, key=lambda x: seed.random())
        ## Sample Per Iteration Limit
        candidates = candidates[:max_sample_per_iter]
        ## Order Assignment Priority Based on Current Smallest Group for Target (Break Ties Based on Overall Group Size)
        assignment_priority = sorted(list(range(eval_cv)), key=lambda f: (assignment_counts[smallest.index.item()][f], len(assignments[f])))
        ## Add Candidates To Folds Based on Priority and Then Remove From Pool
        for c, candidate in enumerate(candidates):
            c_assign = assignment_priority[c % eval_cv]
            assignments[c_assign].add(candidate)
            unassigned.remove(candidate)
            for task, present in group_labels_oh.loc[[candidate]].iloc[0].items():
                assignment_counts[task][c_assign] += present
    ## Randomly Assign Those Without Any Target Labels
    groups_no_lbl = sorted(groups_no_lbl)
    groups_no_lbl = sorted(groups_no_lbl, key=lambda x: seed.random())
    for f, g in enumerate(groups_no_lbl):
        assignments[f % eval_cv].add(g)
    ## Reverse Assignments
    assignments_r = {}
    for g, gass in assignments.items():
        for ass in gass:
            assignments_r[ass] = g    
    ## Translate
    groups = sorted(document_groups_r.keys())
    groups_assign = np.array([assignments_r[g] for g in groups])
    ## Construct Splits
    splits = {}
    for k in range(eval_cv):
        ## K Fold Assignments (1 Test, 1 Dev, eval_cv - 2 Train)
        k_test = k
        k_dev = (k_test - 1) % eval_cv
        k_train = [i % eval_cv for i in range(k_dev - (eval_cv - 2), k_dev)]
        ## Group Assignments
        test_g = sorted([groups[ind] for ind in (groups_assign == k_test).nonzero()[0]])
        dev_g = sorted([groups[ind] for ind in (groups_assign == k_dev).nonzero()[0]])
        train_g = sorted(groups[ind] for ind in np.logical_or.reduce([groups_assign == f for f in k_train]).nonzero()[0])
        ## Translate to Data Indices
        train_doc = set.union(*[document_groups_r[g] for g in train_g])
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g])
        test_doc = set.union(*[document_groups_r[g] for g in test_g])
        ## Check Mutual Exclusion
        assert len(train_doc & dev_doc) == 0
        assert len(train_doc & test_doc) == 0
        assert len(dev_doc & test_doc) == 0
        ## Store
        splits[k] = {"train":train_doc, "dev":dev_doc, "test":test_doc}
    ## Return
    return splits

def _sample_splits_stratified_monte_carlo(preprocessed_data,
                                          data,
                                          metadata,
                                          eval_cv=1,
                                          eval_cv_groups=None,
                                          split_frac=[7,2,1],
                                          max_sample_per_iter=None,
                                          random_state=42,
                                          verbose=True):
    """

    """
    ## Check Arguments
    if eval_cv < 1:
        raise ValueError("eval_cv should be integer at least 1.")
    ## Format/Reduce Split Frac
    if not all(isinstance(i, int) for i in split_frac):
        raise ValueError("Expected integer ratios for split_frac")
    if len(split_frac) != 3:
        raise ValueError("Expected split_frac of length 3.")
    div_ = gcd(*split_frac)
    split_frac = [int(x / div_) for x in split_frac]
    n_folds = sum(split_frac)
    ## Shuffler
    shuffler = np.random.RandomState(random_state)
    ## Initialize Fold to Group Map
    fold_2_group = []
    for nf, g in zip(split_frac,["train","dev","test"]):
        for _ in range(nf):
            fold_2_group.append(g)
    ## Valdiate
    if max_sample_per_iter is None:
        max_sample_per_iter = n_folds
    if max_sample_per_iter < n_folds:
        raise ValueError(f"Must specify a max_sample_per_iter of at least the number of folds ({n_folds}).")
    ## Sample
    splits = {}
    for sample in range(eval_cv):
        ## Shuffle Fold 2 Group
        _ = shuffler.shuffle(fold_2_group)
        ## Generate a Stratified Sample
        sample_splits = _sample_splits_stratified(preprocessed_data=preprocessed_data,
                                                  data=data,
                                                  metadata=metadata,
                                                  eval_cv=n_folds,
                                                  eval_cv_groups=eval_cv_groups,
                                                  max_sample_per_iter=max_sample_per_iter,
                                                  random_state=random_state+sample,
                                                  verbose=verbose and sample==0)
        ## Generate Sample Groups
        sample_groups = {"train":set(),"dev":set(),"test":set()}
        for g, gf in enumerate(fold_2_group):
            sample_groups[gf].update(sample_splits[g]["test"])
        ## Validate
        assert len(sample_groups["train"] & sample_groups["dev"]) == 0
        assert len(sample_groups["train"] & sample_groups["test"]) == 0
        assert len(sample_groups["dev"] & sample_groups["test"]) == 0
        ## Cache Sample Groups
        splits[sample] = sample_groups
    ## Return
    return splits

def create_datasets(args):
    """
    
    """
    ## Relabeling Map
    task_map = label_utils.CLASSIFIER_TASK_MAP
    ## Label to Task Map
    lbl2task = label_utils.create_task_map(task_map=task_map,
                                           separate_negation=args.separate_negation)
    ## Attribute Relabeling Map
    attr_label_rename_map = label_utils.create_attribute_rename_map(task_map=task_map,
                                                                    ignore_non_specified=args.exclude_non_specified)
    ## Load Data
    print("[Loading Raw Data]")
    data = []
    metadata =[]
    for filename in args.datasets:
        ## Load
        _metadata, _data = _load_annotations(annotation_file=filename,
                                             lbl2task=lbl2task,
                                             attr_label_rename_map=attr_label_rename_map,
                                             ignore_non_specified=args.exclude_non_specified,
                                             separate_negation=args.separate_negation,
                                             separate_validity=args.separate_validity is not None,
                                             exclude_icd=args.exclude_icd,
                                             exclude_autolabel=args.exclude_autolabel)
        ## Cache
        metadata.extend(_metadata)
        data.extend(_data)
    ## Format Metadata
    metadata = pd.DataFrame(metadata)
    ## Inferred Attribute Keys
    if args.attribute_keys is not None and len(args.attribute_keys) == 1 and args.attribute_keys == ["default"]:
        ## Identify
        attribute_keys = set()
        for instance in data:
            for lbl in instance["labels"]:
                lbl_attributes = [i for i in lbl.keys() if i != args.entity_key and i not in set(["start","end","in_header","in_autolabel_postprocess","valid"])]
                attribute_keys.update(lbl_attributes)
        attribute_keys = sorted(attribute_keys)
        ## Update User
        print("Inferred The Following Attributes: {}".format(attribute_keys))
        ## Update
        args.attribute_keys = attribute_keys 
    elif args.attribute_keys is not None and len(args.attribute_keys) == 1 and args.attribute_keys == ["default-validity"]:
        ## Identify
        attribute_keys = set()
        for instance in data:
            for lbl in instance["labels"]:
                attribute_keys.update([i for i in lbl.keys() if i.startswith("validity_")])
        attribute_keys = sorted(attribute_keys)
        ## Update User
        print("Inferred The Following Attributes: {}".format(attribute_keys))
        ## Update
        args.attribute_keys = attribute_keys
    ## Filter Validity
    if args.separate_validity is not None:
        if len(args.separate_validity) == 1 and args.separate_validity[0] == "all":
            pass
        elif len(args.separate_validity) == 0:
            pass
        else:
            args.attribute_keys = list(filter(lambda i: not i.startswith("validity_") or i.split("validity_")[1].split("_")[0] in args.separate_validity, args.attribute_keys))
    ## Negation
    if args.separate_negation is not None:
        for s in args.separate_negation:
            if f"negation_{s}" not in args.attribute_keys:
                args.attribute_keys.append(f"negation_{s}")
    ## Initialize Tokenizer
    print("[Initializing Primary Tokenizer]")
    base_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ## Initialize Secondary Tokenizer
    print("[Initializing Secondary Tokenizer]")
    tokenizer = PostProcessTokenizer(tokenizer=base_tokenizer)
    ## Initialize Tagging Tokenizer
    print("[Initializing Tagging Tokenizer]")
    tagging_tokenizer = AttributeTaggingTokenizer(tokenizer.tokenize,
                                                  max_sequence_length=args.max_sequence_length_input if args.split_input else None,
                                                  sequence_overlap=args.sequence_overlap_input if args.split_input else None,
                                                  split_type=args.sequence_split_type_input)
    ## Preprocessing
    n_split = 0
    preprocessed_data = {"tags":[], "tokens":[], "document_id":[], "split":[], "text":[]}
    for d, datum in tqdm(enumerate(data), total=len(data), desc="[Preprocessing]", file=sys.stdout):
        ## Run Tagger
        tagged_sequences = tagging_tokenizer.tokenize(datum["text"],
                                                      datum["labels"],
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
    ## Update User to Preprocessing Output
    print("[Preprocessing Complete. Started with {} Examples. Ended with {} Examples. {} Were Split]".format(len(data), len(preprocessed_data["tags"]), n_split))
    ## Token Distribution
    token_length_distribution = pd.Series(list(map(len, preprocessed_data["tokens"]))).describe()
    print("[Token Length Distribution]")
    print(token_length_distribution.to_string())
    ## Format Entity Labels
    entity_encoder, entity_labels, entity_task_tokens = None, None, None
    if args.include_entity:
        print("[Encoding Entity Labels]")
        entity_encoder = LabelEncoder(primary_key=args.entity_key,
                                      secondary_key=None,
                                      many_to_one=True,
                                      encode_label_position=True)
        entity_labels, entity_task_tokens = entity_encoder.fit_transform(preprocessed_data["tags"],
                                                                         sort=True,
                                                                         align_label_space=False)
    ## Format Attribute Labels
    attribute_encoders, attribute_labels = None, None
    if args.include_attribute:
        print("[Encoding Attribute Labels]")
        attribute_encoders, attribute_labels = {}, {}
        for k in args.attribute_keys:
            attribute_encoders[k] = LabelEncoder(primary_key=args.entity_key,
                                                 secondary_key=k,
                                                 many_to_one=False,
                                                 encode_label_position=False)
            attribute_labels[k] = attribute_encoders[k].fit_transform(preprocessed_data["tags"],
                                                                      sort=True,
                                                                      align_label_space=True)[0]
        ## For Attributes - Apply Either a Span or Document Limit for patients. E.g., Patient X can only contribute 2 spans per attribute.
        if args.limit_group is not None:
            attribute_labels = apply_attribute_limit(attribute_encoders=attribute_encoders,
                                                     attribute_labels=attribute_labels,
                                                     document_groups=metadata.set_index("document_id").loc[preprocessed_data["document_id"]][args.limit_group].apply(tuple,axis=1).tolist(),
                                                     spans_per_document=args.limit_spans_per_document,
                                                     spans_per_document_label_stratify=args.limit_spans_per_document_label_stratify,
                                                     documents_per_group=args.limit_documents_per_group,
                                                     spans_per_group=args.limit_spans_per_group,
                                                     spans_per_group_label_stratify=args.limit_spans_per_group_label_stratify)
    ## Optional: Cache Encoders
    if args.cache_label_encoders:
        print("[Caching Label Encoders]")
        if entity_encoder is not None:
            _ = entity_encoder.save(f"{args.output_dir}/encoder.entity.joblib")
        if attribute_encoders is not None:
            for attr, attr_encoder in attribute_encoders.items():
                _ = attr_encoder.save(f"{args.output_dir}/encoder.attribute.{attr}.joblib")
            with open(f"{args.output_dir}/encoder.attribute-order.json","w") as the_file:
                json.dump({"attributes":list(attribute_encoders.keys())}, the_file)
    ## Display Tasks
    print("[Task Summary]")
    _ = display_task_summary(entity_encoder=entity_encoder,
                             attribute_encoders=attribute_encoders)
    ## Display Data Summary
    print("[Dataset Summary]")
    _ = display_data_summary(entity_encoder=entity_encoder,
                             entity_labels=entity_labels,
                             attribute_encoders=attribute_encoders,
                             attribute_labels=attribute_labels,
                             show_concept_breakdown=False)
    ## Identify Train/Dev/Test Splits
    print("[Identifying Dataset Splits]")
    if args.eval_strategy == "k_fold":
        splits = _sample_splits(preprocessed_data=preprocessed_data,
                                metadata=metadata,
                                eval_cv=args.eval_cv,
                                eval_cv_groups=args.eval_cv_groups,
                                random_state=args.random_state)
    elif args.eval_strategy == "monte_carlo":
        splits = _sample_splits_monte_carlo(preprocessed_data=preprocessed_data,
                                            metadata=metadata,
                                            eval_cv=args.eval_cv,
                                            eval_cv_groups=args.eval_cv_groups,
                                            split_frac=args.eval_mc_split_frac,
                                            random_state=args.random_state)
    elif args.eval_strategy == "stratified_k_fold":
        splits = _sample_splits_stratified(preprocessed_data=preprocessed_data,
                                           data=data,
                                           metadata=metadata,
                                           eval_cv=args.eval_cv,
                                           eval_cv_groups=args.eval_cv_groups,
                                           max_sample_per_iter=None,
                                           random_state=args.random_state)
    elif args.eval_strategy == "stratified_monte_carlo":
        splits = _sample_splits_stratified_monte_carlo(preprocessed_data=preprocessed_data,
                                                       data=data,
                                                       metadata=metadata,
                                                       eval_cv=args.eval_cv,
                                                       eval_cv_groups=args.eval_cv_groups,
                                                       split_frac=args.eval_mc_split_frac,
                                                       max_sample_per_iter=None,
                                                       random_state=args.random_state)
    elif args.eval_strategy == "redundant":
        splits = _sample_splits_redundant(preprocessed_data=preprocessed_data)
    ## Isolate Subset (If Desired)
    if args.eval_cv_fold is not None:
        splits = {x:splits[x] for x in args.eval_cv_fold}
    ## Support Filter - No Entities, Exclude Examples Without Any Relevant Task Data
    if not args.include_entity:
        ## Initialize Support Filter - At Least One Relevant Entity
        support_filter = [False for _ in preprocessed_data["tags"]]
        ## Iterate Through Tasks
        for attribute, labels in attribute_labels.items():
            for i, lbl in enumerate(labels):
                if any([len(j.nonzero()[0]) > 0 for j in lbl]):
                    support_filter[i] = True
        ## Convert Filter
        support_filter = np.nonzero(support_filter)[0]
        if len(support_filter) == 0:
            raise ValueError("Could not find any relevant data after task-filtering.")
        ## Apply Filter
        for key in list(preprocessed_data.keys()):
            preprocessed_data[key] = [preprocessed_data[key][ind] for ind in support_filter]
        for task in list(attribute_labels.keys()):
            attribute_labels[task] = [attribute_labels[task][ind] for ind in support_filter]
    ## Build Datasets
    print("[Initializing Datasets]")
    datasets = {}
    for fold, fold_splits in splits.items():
        ## Get Indices
        train_ind = [i for i, d in enumerate(preprocessed_data["document_id"]) if d in fold_splits["train"]]
        dev_ind = [i for i, d in enumerate(preprocessed_data["document_id"]) if d in fold_splits["dev"]]
        test_ind = [i for i, d in enumerate(preprocessed_data["document_id"]) if d in fold_splits["test"]]
        ## Check Lengths
        for inds, inds_n in zip([train_ind, dev_ind, test_ind],["train","dev","test"]):
            if len(inds) == 0:
                raise ValueError("No samples were included in {} split for fold {}".format(inds_n, fold))
        ## Create Dataset Objects
        datasets[fold] = {}
        for split, split_ind in zip(["train","dev","test"],[train_ind, dev_ind, test_ind]):
            datasets[fold][split] = EntityAttributeDataset(document_ids=[preprocessed_data["document_id"][ind] for ind in split_ind],
                                                           token_ids=[preprocessed_data["tokens"][ind] for ind in split_ind],
                                                           labels_entity=[entity_labels[ind] for ind in split_ind] if entity_labels is not None else None,
                                                           labels_attributes={atr:[atr_labels[ind] for ind in split_ind] for atr, atr_labels in attribute_labels.items()} if attribute_labels is not None else None,
                                                           task_tokens_entity=[entity_task_tokens[ind] for ind in split_ind] if entity_labels is not None else None,
                                                           encoder_entity=entity_encoder if entity_labels is not None else None,
                                                           encoder_attributes=attribute_encoders if attribute_labels is not None else None,
                                                           tagged_sequences=[preprocessed_data["tags"][ind] for ind in split_ind],
                                                           text=[preprocessed_data["text"][ind] for ind in split_ind])
    ## Return Datasets
    return datasets, tokenizer.get_vocab()

def _initialize_model_warm_start(args,
                                 fold,
                                 model):
    """
    
    """
    ## Determine Init Path
    init_path = f"{args.model_init}/" if args.model_init_ignore_fold else f"{args.model_init}/fold-{fold}/"
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Model warm start initialization not found: '{init_path}'")
    ## Determine if Training Completed or Recent Checkpoint
    if os.path.exists(f"{init_path}/model.pt"):
        pass
    elif os.path.exists(f"{init_path}/checkpoints/"):
        checkpoint_dirs = glob(f"{init_path}/checkpoints/checkpoint-*/model.pt")
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("/checkpoint-")[1].split("/")[0]))
        if len(checkpoint_dirs) == 0:
            raise FileNotFoundError(f"Model warm start initialization not found: '{init_path}")
        init_path = os.path.dirname(checkpoint_dirs[-1]) + "/"
    else:
        FileNotFoundError(f"Model warm start initialization not found: '{init_path}'")
    ## Find Desired Init Configuration
    cur_back = ""
    n_back = 0
    while True:
        ## Format Proposed Filename
        model_init_cfg_file = f"{init_path}/{cur_back}/train.cfg.json"
        ## Check for File
        if not os.path.exists(model_init_cfg_file):
            cur_back += "/../"
        else:
            model_init_cfg_file = os.path.abspath(model_init_cfg_file)
            break
        ## Update Recursion Directory Count
        n_back += 1
        if n_back > 10:
            raise FileNotFoundError("Maximum recursion search for config file reached. Initialization may have been provided incorrectly.")
    ## Load Init Configuration
    with open(model_init_cfg_file,"r") as the_file:
        model_init_config = json.load(the_file)
    ## Load Current Script Configuration
    fold_dir = f"{args.output_dir}/fold-{fold}/"
    with open(f"{fold_dir}/train.cfg.json","r") as the_file:
        current_config = json.load(the_file)
    ## Identify Differences
    config_keys = set(model_init_config.keys()) | set(current_config.keys())
    config_differences = list(filter(lambda ck: model_init_config.get(ck,None) != current_config.get(ck, None), config_keys))
    ## Ensure No Breaking Differences
    broken = set(WARM_START_BREAKING_KEYS) & set(config_differences)
    if broken:
        if not args.model_init_reset_training:
            raise ValueError(f"Cannot use warm start model initialization due to differences in parameters: {broken}")
        else: ## Situtationally an issue, don't want to always raise errors
            print("[WARNING: There are known differences in the configuration of the pretrained init model and the current model. These may cause issues.]")
            print("[WARNING: Known differences: {}]".format(broken))
    ## Load Pretrained Model Weights
    model_state_dict = model.state_dict() ## Target Model State Dict
    pretrained_state_dict = torch.load(f"{init_path}/model.pt", map_location=torch.device('cpu')) ## Pretrained Model State Dict
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict} ## Filter Out Irrelevant Keys
    for prefix, prefix_breaking in SPECIFIC_WARM_START_BREAKING_KEYS.items():
        if any(i in broken for i in prefix_breaking):
            print(f"[WARNING {prefix} from warm start model appears to be incompatible. Removing from update.]")
            pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if not k.startswith(prefix)}
    model_state_dict.update(pretrained_state_dict) ## Overwrite Model Dict Weights with Pretrained Weights
    _ = model.load_state_dict(model_state_dict) ## Re-load Updated Model Dict
    ## Return Initialized Model
    return model, init_path
    
def initialize_model(args,
                     encoder_entity,
                     encoder_attributes,
                     vocab2ind,
                     fold):
    """
    
    """
    ## Initialize Model Class
    print("[Creating Model Class]")
    model_init = None
    model = NERTaggerModel(encoder_entity=encoder_entity,
                           encoder_attributes=encoder_attributes,
                           token_encoder=args.encoder,
                           token_vocab=vocab2ind,
                           freeze_encoder=args.freeze_encoder,
                           use_crf=args.use_crf,
                           use_lstm=args.use_lstm,
                           use_entity_token_bias=args.use_entity_token_bias,
                           entity_token_bias_type=args.entity_token_bias_type,
                           use_attribute_concept_bias=args.use_attribute_concept_bias,
                           max_sequence_length=args.max_sequence_length_model,
                           sequence_overlap=args.sequence_overlap_model,
                           sequence_overlap_type=args.sequence_overlap_type_model,
                           lstm_hidden_size=args.lstm_hidden_size,
                           lstm_num_layers=args.lstm_num_layers,
                           lstm_bidirectional=args.lstm_bidirectional,
                           entity_hidden_size=args.entity_hidden_size,
                           attributes_hidden_size=args.attributes_hidden_size,
                           dropout=args.model_dropout_p,
                           random_state=args.random_state)
    ## Warm-Start
    if args.model_init is not None:
        print("[Loading Model Warm Start]")
        model, model_init = _initialize_model_warm_start(args=args,
                                                         fold=fold,
                                                         model=model)
    ## Ensure Encoder is Frozen if Desired
    if args.freeze_encoder:
        _ = model._freeze_encoder()
    ## Return
    return model, model_init
    
def extract_dataset_labels(dataset,
                           vocab2ind):
    """

    """
    ## Get Vocabulary Token Map
    ind2vocab = sorted(vocab2ind.keys(), key=lambda x: vocab2ind.get(x))
    ## Initialize Cache
    labels = []
    ## Iterate Through Splits
    for s, split in enumerate(["train","dev","test"]):
        for ii, x in enumerate(dataset[split]):
            x_toks = [ind2vocab[ind] for ind in x["input_ids"]]
            if dataset[split]._encoder_entity is not None:
                x_sequence_entities = {}
                for tt, (tok, tok_spans) in enumerate(dataset[split]._tagged_sequences[ii]):
                    for (lbl, lbl_start, lbl_valid) in tok_spans:
                        if lbl_start:
                            x_sequence_entities[(lbl["label"],lbl["start"],lbl["end"])] = {"tokens":[tt], "metadata":lbl, "valid":lbl_valid}
                        else:
                            x_sequence_entities[(lbl["label"],lbl["start"],lbl["end"])]["tokens"].append(tt)
                for (ent, char_start, char_end), lbl_span_info in x_sequence_entities.items():
                    labels.append({
                        "split":split,
                        "document_id":dataset[split]._document_ids[ii],
                        "task":"named_entity_recognition",
                        "entity":ent,
                        "label":{True:"Valid",False:"Invalid"}[lbl_span_info["valid"]],
                        "tokens":" ".join(x_toks[tind] for tind in lbl_span_info["tokens"]),
                        "tokens_boundaries":(lbl_span_info["tokens"][0], lbl_span_info["tokens"][-1]+1),
                        "start":char_start,
                        "end":char_end,
                        "in_header":lbl_span_info["metadata"]["in_header"],
                        "in_autolabel_postprocess":lbl_span_info["metadata"]["in_autolabel_postprocess"],
                        "valid":lbl_span_info["metadata"]["valid"]
                    })
            if dataset[split]._encoder_attributes is not None:
                for attr, attr_lbls in zip(dataset[split]._get_attribute_names(), x["attribute_labels"]):
                    for entity, entity_spans in zip(dataset[split]._encoder_attributes[attr].get_tasks(), dataset[split]._encoder_attributes[attr]._extract_spans(attr_lbls.numpy())):
                        entity_classes = dataset[split]._encoder_attributes[attr].get_classes(entity)
                        for attr_lbl, (tok_start, tok_end) in entity_spans:
                            span_meta = [j for j, _, _ in dataset[split]._tagged_sequences[ii][tok_start][1] if j["label"] == entity[0]]
                            if len(span_meta) > 1:
                                raise ValueError("Found a span with multiple labels assigned to same initial token.")
                            elif len(span_meta) == 0:
                                raise ValueError("Unable to find metadata for span.")
                            span_meta = span_meta[0]
                            labels.append({
                                "split":split,
                                "document_id":dataset[split]._document_ids[ii],
                                "task":attr,
                                "entity":entity[0],
                                "label":entity_classes[attr_lbl],
                                "tokens":" ".join(x_toks[tok_start:tok_end]),
                                "tokens_boundaries":(tok_start, tok_end),
                                "start":span_meta["start"],
                                "end":span_meta["end"],
                                "in_header":span_meta["in_header"],
                                "in_autolabel_postprocess":span_meta["in_autolabel_postprocess"],
                                "valid":span_meta["valid"]
                            })
    ## Format
    labels = pd.DataFrame(labels)
    ## Stack
    labels_stacked = pd.pivot_table(labels,
                                    index=["split","document_id","entity","tokens_boundaries","tokens","start","end","in_header","in_autolabel_postprocess","valid"],
                                    columns="task",
                                    values="label",
                                    aggfunc=lambda x: x.iloc[0]).reset_index().rename(columns={"entity":"label"})
    ## Sort
    labels_stacked = labels_stacked.sort_values(["label","document_id","valid","in_autolabel_postprocess","in_header"])
    labels_stacked = labels_stacked.reset_index(drop=True)
    return labels_stacked

def run_fold_train(args,
                   dataset,
                   vocab2ind,
                   fold):
    """
    
    """
    ## Initialize Output Directory
    print("[Initializing Output Directory]")
    fold_output_dir = f"{args.output_dir}/fold-{fold}/"
    if os.path.exists(fold_output_dir) and args.keep_existing:
        if not os.path.exists(f"{fold_output_dir}/train.log.pt"):
            print(">> Fold exists, but hasn't completed. Will proceed as normal.")
        else:
            print(">> Fold complete. Exiting.")
            return None
    if os.path.exists(fold_output_dir) and not args.keep_existing and not args.no_training:
        if args.rm_existing:
            _ = os.system(f"rm -rf {fold_output_dir}")
        else:
            raise FileExistsError(f"Fold output directory exists ('{fold_output_dir}'). Include --rm_existing flag to overwrite.")
    if not os.path.exists(fold_output_dir):
        _ = os.makedirs(fold_output_dir)
        with open(f"{fold_output_dir}/train.cfg.json","w") as the_file:
            json.dump(vars(args), the_file, indent=1)
    ## Extract Dataset Statistics
    print("[Extracting Dataset Split Labels]")
    dataset_labels = extract_dataset_labels(dataset,
                                            vocab2ind)
    _ = dataset_labels.to_csv(f"{fold_output_dir}/labels.csv",index=False)
    ## Baseline Performance Metrics
    print("[Gathering Baseline Performance Metrics]")
    baseline_training_logs, _ = train_baseline(dataset=dataset,
                                               eval_train=args.eval_train,
                                               eval_test=args.eval_test,
                                               weighting_entity=args.weighting_entity,
                                               weighting_attribute=args.weighting_attribute,
                                               use_char=args.baseline_use_char)
    print("[Caching Baseline Training Logs]")
    for mode, mode_logs in baseline_training_logs.items():
        _ = torch.save(mode_logs, f"{fold_output_dir}/baseline.{mode}.train.log.pt")
    if args.baseline_only:
        print(f"[Baseline Only Run. Fold {fold} Complete.]")
        return None
    ## Initialize Model
    print("[Initializing Primary Model]")
    model, model_init = initialize_model(args=args,
                                         encoder_entity=dataset["train"]._encoder_entity,
                                         encoder_attributes=dataset["train"]._encoder_attributes,
                                         vocab2ind=vocab2ind,
                                         fold=fold)
    ## Run Training Procedure
    print("[Starting Training Procedure for Primary Model]")
    training_logs, model = train(dataset=dataset,
                                 model=model,
                                 lr=args.model_lr,
                                 lr_warmup=args.model_lr_warmup,
                                 lr_step_size=args.model_lr_step_size,
                                 lr_gamma=args.model_lr_gamma,
                                 lr_adaptive_method=args.model_lr_adaptive_method,
                                 nesterov=args.model_nesterov,
                                 momentum=args.model_momentum,
                                 opt_method=args.model_opt_method,
                                 weight_decay=args.model_weight_decay,
                                 grad_clip_norm=args.model_grad_clip_norm,
                                 max_epochs=args.model_n_epochs,
                                 max_steps=args.model_n_steps,
                                 train_batch_size=args.model_train_batch_size,
                                 train_gradient_accumulation=args.model_train_gradient_accumulation,
                                 eval_batch_size=args.model_eval_batch_size,
                                 eval_frequency=args.model_eval_frequency,
                                 eval_strategy=args.model_eval_strategy,
                                 eval_train=args.eval_train,
                                 eval_test=args.eval_test,
                                 save_frequency=args.model_save_frequency,
                                 weighting_entity=args.weighting_entity,
                                 weighting_attribute=args.weighting_attribute,
                                 use_first_index=args.attribute_use_first_index,
                                 early_stopping_tol=args.early_stopping_tol,
                                 early_stopping_patience=args.early_stopping_patience,
                                 early_stopping_warmup=args.early_stopping_warmup,
                                 random_state=args.random_state,
                                 display_cumulative_batch_loss=args.display_batch_loss,
                                 model_init=model_init if not args.model_init_reset_training else None,
                                 checkpoint_dir=f"{fold_output_dir}/checkpoints/",
                                 gpu_id=args.gpu_id,
                                 no_training=args.no_training)
    ## Cache Training Logs
    print("[Caching Primary Model Training Logs]")
    _ = torch.save(training_logs, f"{fold_output_dir}/train.log.pt")
    ## Cache Model State
    if args.model_save_frequency is not None and args.model_save_frequency > 0:
        print("[Caching Primary Model]")
        _ = torch.save(model.state_dict(), f"{fold_output_dir}/model.pt")
    ## Done
    print(f"[Fold {fold} Complete]")

def run_fold_evaluate(args,
                      dataset,
                      vocab2ind,
                      fold):
    """

    """
    ## Device
    print("[Determining Device]")
    device = get_device(args.gpu_id)
    print(f">> WARNING: Using Following Device for Evaluation -- {device}")
    ## Initialize Output Directory
    print("[Initializing Output Directory]")
    fold_output_dir = f"{args.output_dir}/fold-{fold}/predictions/"
    if os.path.exists(fold_output_dir) and not args.rm_existing:
        raise FileExistsError("Include --rm_existing flag to overwrite fold-predictions.")
    if not os.path.exists(fold_output_dir):
        _ = os.makedirs(fold_output_dir)
    with open(f"{fold_output_dir}/evaluate.cfg.json","w") as the_file:
        json.dump(vars(args), the_file, indent=1)
    ## Initialize Model
    print("[Initializing Model]")
    model, model_init = initialize_model(args=args,
                                         train_dataset=dataset["train"],
                                         encoder_entity=dataset["train"]._encoder_entity,
                                         encoder_attributes=dataset["train"]._encoder_attributes,
                                         vocab2ind=vocab2ind,
                                         fold=fold)
    ## Check Init
    print("[Validating Model Initialization]")
    if model_init is None:
        raise ValueError("Model should be trained already to run evaluation.")
    ## Initialize Class Weights
    print("[Initializing Class Weights]")
    entity_weights, attribute_weights = initialize_class_weights(dataset=dataset["train"],
                                                                 weighting_entity=args.weighting_entity,
                                                                 weighting_attribute=args.weighting_attribute)
    ## Iterate Through Splits
    print("[Beginning Split Evaluation]")
    for split in ["train","dev","test"]:
        ## Ignore Test if Desired
        if split == "test" and not args.eval_test:
            continue
        ## Run Prediction
        output = eval.evaluate(model=model,
                               dataset=dataset[split],
                               batch_size=args.model_eval_batch_size,
                               entity_weights=entity_weights,
                               attribute_weights=attribute_weights,
                               desc=split.title(),
                               device=device,
                               use_first_index=args.attribute_use_first_index,
                               return_predictions=True)
        ## Defaults
        entity_predictions_fmt = None
        attribute_predictions_fmt = None
        ## Make Predictions
        if output["valid"]["entity"]:
            print("[Formatting Entity Predictions: {}]".format(split))
            entity_predictions_fmt = eval.format_entity_predictions(entity_predictions=output["entity"]["predictions"],
                                                                    dataset=dataset[split],
                                                                    vocab2ind=vocab2ind)
        if output["valid"]["attributes"]:
            print("[Formatting Attribute Predictions: {}]".format(split))
            attribute_predictions_fmt = eval.format_attribute_predictions(attribute_predictions=output["attributes"]["predictions"],
                                                                          dataset=dataset[split],
                                                                          vocab2ind=vocab2ind,
                                                                          context_size=10)
        ## Cache
        if entity_predictions_fmt is not None:
            print("[Caching Entity Predictions: {}]".format(split))
            _ = entity_predictions_fmt.to_csv(f"{fold_output_dir}/entities.{split}.csv", index=False)
        if attribute_predictions_fmt is not None:
            print("[Caching Attribute Predictions: {}]".format(split))
            _ = attribute_predictions_fmt.to_csv(f"{fold_output_dir}/attributes.{split}.csv", index=False)

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## GPU Setup
    print("[Initializing GPU and Relevant Arguments (If Appropriate)]")
    args, gpu_hold = initialize_gpu_args(args)
    ## Initialize Output Directory
    print("[Initializing Output Directory]")
    if not os.path.exists(args.output_dir):
        try:
            _ = os.makedirs(args.output_dir)
        except FileExistsError as e:
            print("[WARNING: Output directory already exists. This may be expected if running concurrent folds.]")
    ## Check Existence
    if args.eval_cv_fold is not None and not args.evaluate:
        print("[Checking for Prior Completion]")
        all_complete = True
        for fold in args.eval_cv_fold:
            fold_output_dir = f"{args.output_dir}/fold-{fold}/"
            if os.path.exists(fold_output_dir) and args.keep_existing:
                if not os.path.exists(f"{fold_output_dir}/train.log.pt"):
                    all_complete = False
                    print(f">> Found incomplete fold: {fold}")
                else:
                    pass
            else:
                all_complete = False
        ## Early Exit
        if all_complete:
            print(">> All CV folds complete. Exiting early.")
            print("[Script Complete]")
            return None
    ## Create Dataset
    print("[Creating Datasets]")
    datasets, vocab2ind = create_datasets(args)
    ## Run Procedures
    print("[Beginning Script Procedures]")
    for f, (fold, fold_dataset) in enumerate(datasets.items()):
        if args.evaluate:
            print("[Beginning Evaluation Fold {} ({}/{})]".format(fold, f+1, len(datasets)))
            _ = run_fold_evaluate(args=args,
                                  dataset=fold_dataset,
                                  vocab2ind=vocab2ind,
                                  fold=fold)
        else:
            print("[Beginning Training Fold {} ({}/{})]".format(fold, f+1, len(datasets)))
            _ = run_fold_train(args=args,
                               dataset=fold_dataset,
                               vocab2ind=vocab2ind,
                               fold=fold)
    ## Done
    print("[Script Complete]")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()