"""
Fine-tune a BERT model using the Masked Language Model objective
Original Source: https://github.com/x4nth055/pythoncode-tutorials/blob/master/machine-learning/nlp/pretraining-bert/pretrainingbert.py
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import argparse
from functools import partial

## External Libraries
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers import EarlyStoppingCallback

## Private
from cce.util.helpers import flatten
from cce.model.datasets import MLMTokenDataset, mlm_collate_batch

#######################
### Functions
#######################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--setting", type=str, choices={"tokenize","train","plot"}, default=None)
    _ = parser.add_argument("--model_dir", type=str, default=None, help="Initial model to fine-tune.")
    _ = parser.add_argument("--model_dir_rm", action="store_true", default=False, help="By default, reuse of a model directory will continue pretraining. This overwrites the original directory and starts from scratch.")
    _ = parser.add_argument("--tokenizer_init", type=str, default=None, help="Tokenizer to use.")
    _ = parser.add_argument("--model_init", type=str, default=None, help="Tokenizer/Model to use as the initialization. If None (default), assumes training a BERT base model from scratch.")
    _ = parser.add_argument("--model_resume_training", type=str, default=None, help="Last checkpoint directory to use for resuming training.")
    _ = parser.add_argument("--data_train", type=str, default=None, nargs="+", help="Data files to use for pretraining.")
    _ = parser.add_argument("--data_eval", type=str, default=None, nargs="+", help="Data files to use for evaluation.")
    _ = parser.add_argument("--data_eval_p", type=float, default=0.1, help="Proportion of training data to hold out for evaluation if no evaluation files provided.")
    _ = parser.add_argument("--sample_train_size", type=int, default=None, help="Downsample training set size if desired.")
    _ = parser.add_argument("--sample_eval_size", type=int, default=None, help="Downsample evaluation set size if desired.")
    _ = parser.add_argument("--sample_random_state", type=int, default=42, help="Random seed for sampling.")
    _ = parser.add_argument("--learn_vocabulary_size", type=int, default=50000, help="Size of the vocabulary.")
    _ = parser.add_argument("--learn_vocabulary_min_freq", type=int, default=3, help="Minimum token frequency to keep in vocabulary.")
    _ = parser.add_argument("--learn_batch", type=int, default=8, help="Batch Size")
    _ = parser.add_argument("--learn_gradient_accumulation_steps", type=int, default=1, help="How many batches to process before running a backwards pass.")
    _ = parser.add_argument("--learn_sequence_length_data", type=int, default=128, help="Maximum sequence length of training dataset.")
    _ = parser.add_argument("--learn_sequence_length_model", type=int, default=128, help="Context window size for the BERT model.")
    _ = parser.add_argument("--learn_sequence_overlap", type=int, default=0, help="How much token overlap to include between sequences from the same note (dataset level).")
    _ = parser.add_argument("--learn_lr", type=float, default=0.00005, help="Initial learning rate")
    _ = parser.add_argument("--learn_warmup_steps", type=int, default=10000, help="Warm up before decaying learning rate.")
    _ = parser.add_argument("--learn_weight_decay", type=float, default=0, help="Weight decay.")
    _ = parser.add_argument("--learn_epochs", type=int, default=1, help="Number of training epochs (passes through the data)")
    _ = parser.add_argument("--learn_max_steps", type=int, default=-1, help="If desired, maximum number of steps. Overrides --learn_epochs.")
    _ = parser.add_argument("--learn_mask_probability", type=float, default=0.15, help="Mask probability.")
    _ = parser.add_argument("--learn_mask_max_per_sequence", type=int, default=20, help="Maximum number of tokens to mask in each sequence.")
    _ = parser.add_argument("--learn_eval_strategy", type=str, choices={"epoch","steps"}, default="epoch", help="Whether to evaluate based on epoch or step frequency.")
    _ = parser.add_argument("--learn_eval_frequency", type=int, default=1, help="After how many updates to re-run evaluation during training.")
    _ = parser.add_argument("--learn_save_total_limit", type=int, default=None, help="Maximum number of checkpoints to keep. Default is all.")
    _ = parser.add_argument("--learn_random_state", type=int, default=42, help="Random seed.")
    _ = parser.add_argument("--learn_early_stopping", action="store_true", default=False, help="Whether to use early stopping.")
    _ = parser.add_argument("--learn_early_stopping_patience", type=int, default=3, help="Early Stopping Patience")
    _ = parser.add_argument("--learn_early_stopping_tol", type=float, default=0.0, help="Early Stopping Tolerance")
    _ = parser.add_argument("--ignore_data_skip", action="store_true", default=False)
    _ = parser.add_argument("--plot_prop", type=float, default=None, help="If desired, proportion of training curve to plot.")
    args = parser.parse_args()
    ## Check Arguments
    if args.model_dir is None:
        raise FileNotFoundError("A --model_dir was not included.")
    if args.setting is None:
        raise ValueError("Must specify a setting (--setting)")
    if args.setting == "tokenize" and args.model_init is not None:
        raise ValueError("No need to fit a tokenizer for an existing model specified via --model_init.")
    if args.model_init is not None and os.path.exists(args.model_init) and os.path.abspath(args.model_init) == os.path.abspath(args.model_dir):
        raise FileExistsError("Please a new output directory (--model_dir) when loading an existing pretrained model.")
    if args.setting in ["tokenize","train"] and args.data_train is None:
        raise ValueError("Must provide training data if running training.")
    return args

def _load_data_file(file):
    """
    
    """
    ## Check for File
    if not os.path.exists(file):
        raise FileNotFoundError(f"Data file not found: '{file}'")
    ## Initialize Cache
    file_data = []
    ## Load Based on Type
    if file.endswith(".txt"):
        with open(file,"r") as the_file:
            for line in the_file:
                file_data.append(line.strip())
    elif file.endswith(".json"):
        with open(file, "r") as the_file:
            for line in the_file:
                line_data = json.loads(line)
                if "text" not in line_data:
                    raise KeyError("Missing 'text' field in dictionary.")
                file_data.append(line_data["text"])
    elif file.endswith(".csv"):
        file_data = pd.read_csv(file)
        if "text" not in file_data.columns:
            raise KeyError("Missing 'text' field in dataframe.")
        file_data = file_data["text"].tolist()
    else:
        raise NotImplementedError("Data file type not recognized.")
    ## Return
    return file_data

def _load_data_files(filenames):
    """
    
    """
    ## Check
    if filenames is None or len(filenames) == 0:
        raise FileNotFoundError("Must provide at least one filename.")
    if isinstance(filenames, str):
        raise TypeError("'filenames' should be a list of filename strings.")
    ## Initialize
    data = []
    for file in filenames:
        data.extend(_load_data_file(file))
    ## Return
    return data

def load_dataset(data_train,
                 data_eval,
                 data_eval_p=0.1,
                 train_size=None,
                 eval_size=None,
                 random_state=42):
    """
    
    """
    ## Initialize Dataset
    dataset = {}
    ## Load Training
    print("[Loading Training Data Files]")
    dataset["train"] = _load_data_files(data_train)
    ## Load or Create Evaluation
    if data_eval is not None:
        print("[Loading Evaluation Data Files]")
        dataset["eval"] = _load_data_files(data_eval)
    else:
        assert data_eval_p < 1
        eval_p_seed = np.random.RandomState(random_state)
        eval_p_size = int(len(dataset["train"]) * data_eval_p)
        eval_p_inds_shuffle = sorted(list(range(len(dataset["train"]))), key=lambda x: eval_p_seed.rand())
        eval_p_inds_train = sorted(eval_p_inds_shuffle[:-eval_p_size])
        eval_p_inds_eval = sorted(eval_p_inds_shuffle[-eval_p_size:])
        assert len(set(eval_p_inds_train) & set(eval_p_inds_eval)) == 0
        dataset["eval"] = [dataset["train"][ind] for ind in eval_p_inds_eval]
        dataset["train"] = [dataset["train"][ind] for ind in eval_p_inds_train]
    ## Downsample if Desired
    if train_size is not None and train_size < len(dataset["train"]):
        print("[Downsampling Training Data: {:,d} to {:,d}]".format(len(dataset["train"]), train_size))
        train_seed = np.random.RandomState(random_state)
        train_ind = sorted(train_seed.choice(len(dataset["train"]), train_size, replace=False))
        dataset["train"] = [dataset["train"][ind] for ind in train_ind]
    if eval_size is not None and eval_size < len(dataset["eval"]):
        print("[Downsampling Eval Data: {:,d} to {:,d}]".format(len(dataset["eval"]), eval_size))
        eval_seed = np.random.RandomState(random_state)
        eval_ind = sorted(eval_seed.choice(len(dataset["eval"]), eval_size, replace=False))
        dataset["eval"] = [dataset["eval"][ind] for ind in eval_ind]
    ## Return
    return dataset

def split_sequence(tokens,
                   max_sequence_length=512,
                   overlap=0,
                   special_start_id=101,
                   special_end_id=102):
    """
    
    """
    ## Check Length
    if len(tokens) <= max_sequence_length:
        return [tokens]
    ## Remove Special Characters
    tokens = tokens[1:-1]
    ## Size
    max_sequence_length_no_special = max_sequence_length - 2
    ## Generate Splits
    tokens_split = []
    cur_start = 0
    while True:
        ## Add to Split
        tokens_split.append([special_start_id] + tokens[cur_start:cur_start+max_sequence_length_no_special] + [special_end_id])
        ## Check Completion
        if cur_start+max_sequence_length_no_special >= len(tokens):
            break
        ## Move Forward
        cur_start = cur_start + max_sequence_length_no_special - overlap
    ## Return
    return tokens_split

def fit_tokenizer(dataset,
                  model_dir,
                  vocab_size=32_500,
                  min_frequency=3,
                  max_length=512):
    """

    """
    ## Initialize New Tokenizer
    tokenizer = BertWordPieceTokenizer()
    ## Train Tokenizer Using Training Data
    print("[Training Tokenizer]")
    _ = tokenizer.train_from_iterator(dataset,
                                      vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      show_progress=True,
                                      wordpieces_prefix="##")
    tokenizer.enable_truncation(max_length=max_length)
    ## Save Tokenizer
    print("[Caching Tokenizer]")
    tokenizer.save_model(model_dir)
    ## Cache Config
    print("[Caching Tokenizer Configuration]")
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)

def initialize_torch_dataset(dataset,
                             tokenizer_path,
                             max_length=512,
                             random_state=42):
    """

    """
    ## Initialize Trained Tokenizer
    print("[Initializing Fast Tokenizer]")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    ## Tokenize
    tokens_train = list(map(tokenizer.encode, tqdm(dataset["train"], desc="[Tokenizing Training Data]", file=sys.stdout)))
    tokens_eval = list(map(tokenizer.encode, tqdm(dataset["eval"], desc="[Tokenizing Eval Data]", file=sys.stdout)))
    print(">> # Training Tokens: {:,d}".format(sum(list(map(len, tokens_train)))))
    print(">> # Eval Tokens: {:,d}".format(sum(list(map(len, tokens_eval)))))
    ## Split into Max Length
    tokens_train = flatten(list(map(lambda tokens: split_sequence(tokens, max_sequence_length=max_length, overlap=0, special_start_id=tokenizer.cls_token_id, special_end_id=tokenizer.sep_token_id), tqdm(tokens_train, desc="[Segmenting Training Dataset]", file=sys.stdout))))
    tokens_eval = flatten(list(map(lambda tokens: split_sequence(tokens, max_sequence_length=max_length, overlap=0, special_start_id=tokenizer.cls_token_id, special_end_id=tokenizer.sep_token_id), tqdm(tokens_eval, desc="[Segmenting Eval Dataset]", file=sys.stdout))))
    ## Translate Tokens into Dataset
    print("[Initializing Token Datasets]")
    ds_train = MLMTokenDataset(tokens_train)
    ds_eval = MLMTokenDataset(tokens_eval)
    ## Return
    return tokenizer, ds_train, ds_eval

def run_tokenize(args):
    """

    """
    ## Load Training Examples (Format {"train":[str, str, ...], "eval":[str, str, ...]})
    print("[Loading Dataset]")
    dataset = load_dataset(data_train=args.data_train,
                           data_eval=args.data_eval,
                           data_eval_p=args.data_eval_p,
                           train_size=args.sample_train_size,
                           eval_size=args.sample_eval_size,
                           random_state=args.sample_random_state)
    ## Fit Tokenizer
    print("[Fitting Tokenizer]")
    _ = fit_tokenizer(dataset=dataset["train"],
                      model_dir=args.model_dir,
                      vocab_size=args.learn_vocabulary_size,
                      min_frequency=args.learn_vocabulary_min_freq,
                      max_length=args.learn_sequence_length_data)


def _load_mlm_dataset_and_tokenizer(args):
    """

    """
    ## Load Training Examples (Format {"train":[str, str, ...], "eval":[str, str, ...]})
    print("[Loading Dataset]")
    dataset = load_dataset(data_train=args.data_train,
                           data_eval=args.data_eval,
                           data_eval_p=args.data_eval_p,
                           train_size=args.sample_train_size,
                           eval_size=args.sample_eval_size,
                           random_state=args.sample_random_state)
    ## Tokenizer
    print("[Determining Tokenizer]")
    if args.tokenizer_init is not None:
        tokenizer_path = args.tokenizer_init
    else:
        if args.model_init is None:
            tokenizer_path = args.model_dir
        else:
            tokenizer_path = args.model_init
    ## Initialize Datasets
    print("[Initializing Tokenized PyTorch Datasets]")
    tokenizer, ds_train, ds_eval = initialize_torch_dataset(dataset=dataset,
                                                            tokenizer_path=tokenizer_path,
                                                            max_length=args.learn_sequence_length_data)
    ## Return
    return tokenizer, ds_train, ds_eval

def _load_bert_model(model_init=None,
                     vocabulary_size=None,
                     max_length=None):
    """

    """
    if model_init is None:
        ## Check
        if vocabulary_size is None or max_length is None:
            raise ValueError("Unable to initialize a BERT model from scratch without specifying --learn_vocabulary_size or --learn_sequence_length.")
        ## Initialize
        print("[Initializing BERT Model from Scratch]")
        model_config = BertConfig(vocab_size=vocabulary_size,
                                  max_position_embeddings=max_length)
        model = BertForMaskedLM(config=model_config)
    else:
        ## Initialize
        print(f"[Initializing BERT Model from Pretrained: {model_init}]")
        model_config = BertConfig.from_pretrained(model_init)
        if max_length != model_config.max_position_embeddings:
            print("[WARNING: Initializing BERT Model with Different max_position_embeddings Than Currently Exists]")
        model = BertForMaskedLM.from_pretrained(model_init,
                                                max_position_embeddings=max_length,
                                                ignore_mismatched_sizes=True)
    ## Return
    return model

def run_train(args):
    """

    """
    ## Load
    print("[Loading MLM Dataset and Tokenizer]")
    tokenizer, ds_train, ds_eval = _load_mlm_dataset_and_tokenizer(args)
    ## Initialize Model
    print("[Initializing BERT Model]")
    model = _load_bert_model(model_init=args.model_init,
                             vocabulary_size=len(tokenizer.vocab),
                             max_length=args.learn_sequence_length_model)
    ## Check Vocabulary Alignment
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Mismatch between tokenizer vocab_size and model vocab_size")
    ## Set Rank (For Parallel Training)
    try:
        _ = torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    except Exception as e:
        print("[WARNING: Error occured while setting local device rank]")
        print("[WARNING: {}]".format(e))
    ## Early Stopping
    callbacks = []
    if args.learn_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.learn_early_stopping_patience,
                                  early_stopping_threshold=args.learn_early_stopping_tol)
        )
    ## Training Arguments
    print("[Initializing Training Arguments]")
    training_args = TrainingArguments(output_dir=args.model_dir,
                                      logging_dir=f"{args.model_dir}/logs/",
                                      overwrite_output_dir=False,
                                      per_device_train_batch_size=args.learn_batch,
                                      per_device_eval_batch_size=args.learn_batch,
                                      gradient_accumulation_steps=args.learn_gradient_accumulation_steps,
                                      eval_accumulation_steps=args.learn_gradient_accumulation_steps,
                                      warmup_steps=args.learn_warmup_steps,
                                      weight_decay=args.learn_weight_decay,
                                      learning_rate=args.learn_lr,
                                      num_train_epochs=args.learn_epochs,
                                      max_steps=args.learn_max_steps,
                                      seed=args.learn_random_state,
                                      data_seed=args.learn_random_state,
                                      push_to_hub=False,
                                      no_cuda=False,
                                      evaluation_strategy=args.learn_eval_strategy,
                                      eval_steps=args.learn_eval_frequency if args.learn_eval_strategy == "steps" else None,
                                      save_strategy=args.learn_eval_strategy,
                                      save_steps=args.learn_eval_frequency if args.learn_eval_strategy == "steps" else None,
                                      save_total_limit=args.learn_save_total_limit,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="eval_loss",
                                      greater_is_better=False,
                                      logging_strategy=args.learn_eval_strategy,
                                      logging_steps=args.learn_eval_frequency if args.learn_eval_strategy == "steps" else None,
                                      logging_first_step=True,
                                      optim="adamw_torch",
                                      ddp_find_unused_parameters=False,
                                      ignore_data_skip=args.ignore_data_skip)
    print("[Using Device: {}]".format(training_args.device))
    ## Parameterize The Collator
    print("[Initializing Data Collator]")
    collator = partial(mlm_collate_batch,
                       mlm_probability=args.learn_mask_probability,
                       mlm_max_per_sequence=args.learn_mask_max_per_sequence,
                       special_mask_id=tokenizer.mask_token_id)
    ## Initialize Trainer
    print("[Initializing Trainer]")
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=ds_train,              # training dataset
        eval_dataset=ds_eval,                # evaluation dataset
        data_collator=collator,              # function used for collating and random masking
        callbacks=callbacks
    )
    ## Run Training
    print("[Beginning Training Procedure]")
    trainer.train(resume_from_checkpoint=False if args.model_resume_training is None else args.model_resume_training)
    ## Save Final Model
    print("[Caching Final Model]")
    _ = trainer.save_model()
    _ = trainer.save_state()

def run_plot(args):
    """

    """
    ## Load
    print("[Loading State]")
    with open(f"{args.model_dir}/trainer_state.json","r") as the_file:
        state = json.load(the_file)
    ## Format History
    logs = pd.DataFrame(state["log_history"])
    ## Isolate Train/Eval
    train_hist = logs.dropna(subset=["loss"])
    eval_hist = logs.dropna(subset=["eval_loss"])
    ## Isolate Recent
    if args.plot_prop is not None:
        min_step = train_hist.step.max() * (1 - args.plot_prop)
        train_hist = train_hist.loc[train_hist["step"] >= min_step]
        eval_hist = eval_hist.loc[eval_hist["step"] >= min_step]
    ## Plot
    fig, ax = plt.subplots()
    ax.plot(train_hist.step, train_hist.loss, label="Training")
    ax.plot(eval_hist.step, eval_hist.eval_loss, label="Evaluation")
    ax.legend(loc="upper right")
    ax.set_xlabel("Steps", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title("Loss (Last {}%)".format(args.plot_prop * 100) if args.plot_prop is not None else "Loss")
    fig.tight_layout()
    plt.savefig(f"{args.model_dir}/log_history.png",dpi=300)
    plt.close(fig)

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Initialize Output Directory
    print("[Initializing Model Directory]")
    if os.path.exists(args.model_dir) and args.model_dir_rm:
        if args.setting == "train" and args.model_init is None:
            raise ValueError("Removing the model directory will remove the tokenizer that has already been trained.")
        _ = os.system(f"rm -rf {args.model_dir}")
    if not os.path.exists(args.model_dir):
        if args.setting == "train" and (args.model_init is None and args.tokenizer_init is None):
            raise ValueError("Fit the tokenizer or provide --tokenizer_init before running the training script.")
        try:
            _ = os.makedirs(args.model_dir)
        except FileExistsError: # Race condition when running multi-gpu training
            pass
    ## Run Based on Setting
    print(f"[Running Setting: '{args.setting}']")
    if args.setting == "tokenize":
        _ = run_tokenize(args)
    elif args.setting == "train":
        _ = run_train(args)
    elif args.setting == "plot":
        _ = run_plot(args)
    else:
        raise ValueError("Setting not recognized.")
    ## Done
    print("[Script Complete]")

########################
### Execute
########################

if __name__ == "__main__":
    _ = main()