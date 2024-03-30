
## TODO: Apply Trained Model to Dataset

#######################
### Globals
#######################

## Configuration Changes That Prevent Model Warm Starts
WARM_START_BREAKING_KEYS = [
    "encoder",
    "tokenizer",
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
        "tokenizer"
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

def parse_command_line()
    """

    """
    _ = parser.add_argument("--model_init", type=str, default=None, help="If desired, can provide a cached model to use as the starting point of training.")
    _ = parser.add_argument("--model_init_ignore_fold", action="store_true", default=False, help="By default, we assume the model_init folds align with what is being trained.")
    _ = parser.add_argument("--model_init_reset_training", action="store_true", default=False, help="If including a model_init to initialize weights but you don't want to continue training.")
    _ = parser.add_argument("--no_training", action='store_true', default=False, help="Load an existing checkpoint / training logs and generate plots without additional training.")

    if args.model_init is not None and not os.path.exists(args.model_init):
        raise FileNotFoundError(f"Missing required file to perform model initialization: {args.model_init}")
    if args.no_training:
        if args.model_init is None:
            print(">> WARNING - No model_init provided alongside --no_training flag. This may be okay depending on purpose.")
 

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
                                                                 weighting_attribute=args.weighting_attribute,
                                                                 gamma_entity=args.weighting_entity_gamma,
                                                                 gamma_attribute=args.weighting_attribute_gamma)
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