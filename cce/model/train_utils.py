
######################
### Imports
######################

## Standard Libraries
import os
from glob import glob
from datetime import datetime
from collections import Counter

## External Libraries
import torch
import numpy as np
import pandas as pd

## Local
from ..util.helpers import chunks
from .datasets import collate_entity_attribute, move_batch_to_device
from .loss import compute_entity_loss, compute_attribute_loss
from .eval import evaluate, display_evaluation
from .architectures import BaselineTaggerModel
## NOTE: ADDED START
from .eval import format_entity_predictions, format_attribute_predictions
## NOTE: ADDED END

######################
### Functions
######################

def show_cuda_memory_info(device):
    """

    """
    ## Ignore CPU
    if device == "cpu":
        return 
    ## Computations
    gib_converter = 1 / (1024 ** 3)
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    print("Total CUDA Memory: {:.3f} GiB".format(t * gib_converter))
    print("Reserved CUDA Memory: {:.3f} GiB".format(r * gib_converter))
    print("Allocated CUDA Memory: {:.3f} GiB".format(a * gib_converter))
    print("Free CUDA Memory: {:.3f} GiB".format(f * gib_converter))

def get_device(gpu_id=None):
    """

    """
    if gpu_id is None or not torch.cuda.is_available():
        training_device = "cpu"
    else:
        if isinstance(gpu_id, str) or gpu_id != -1:
            training_device = f"cuda:{gpu_id}"
        else:
            training_device = "cuda"
    return training_device

def initialize_class_weights(dataset,
                             weighting_entity=None,
                             weighting_attribute=None,
                             alpha=1,
                             gamma_entity=1,
                             gamma_attribute=1):
    """
    
    """
    ## Check Parameters
    assert alpha >= 0
    for gamma in [gamma_entity, gamma_entity]:
        assert gamma >= 0
        if gamma > 1:
            print(">> WARNING - Setting gamma_* > 1 may cause unexpected behavior.")
    ## Counts
    entity_counts = {}
    attribute_counts = {}
    ## Get Counts
    for x in dataset:
        if dataset._encoder_entity is not None:
            for c, (concept, task) in enumerate(zip(x["entity_labels"], dataset._encoder_entity._id2task)):
                if task not in entity_counts:
                    entity_counts[task] = Counter()
                for ind, ind_count in zip(*torch.unique(concept, return_counts=True)):
                    entity_counts[task][ind.item()] += ind_count.item()
        if dataset._encoder_attributes is not None:
            for task, task_concepts in x["attribute_spans"].items():
                if task not in attribute_counts:
                    attribute_counts[task] = Counter()            
                for concept in task_concepts:
                    for lbl, _ in concept:
                        attribute_counts[task][lbl] += 1
    ## Format Counts
    if dataset._encoder_entity is not None:
        entity_counts = {x:np.array([y[i] + alpha for i in range(3)]) for x, y in entity_counts.items()}
    if dataset._encoder_attributes is not None:
        for task, task_counts in attribute_counts.items():
            task_classes = list(dataset._encoder_attributes[task]._id2class_filt.values())[0]
            attribute_counts[task] = np.array([task_counts[i] for i in range(len(task_classes))])
        for task, counts in attribute_counts.items():
            task_alpha = np.array([alpha for _ in attribute_counts[task]])
            attribute_counts[task] = counts + task_alpha
    ## Gamma Factor (Smoothing - Gamma of 0 means everything has equal weight. Gamma of 1 means complete inverse weight)
    if dataset._encoder_entity:
        entity_weights = {x:np.power(y, gamma_entity) for x, y in entity_counts.items()}
    if dataset._encoder_attributes is not None:
        attribute_counts = {x:np.power(y, gamma_attribute) for x, y in attribute_counts.items()}
    ## Compute Balanced Weights
    entity_weights, attribute_weights = None, None
    if dataset._encoder_entity is not None:
        entity_weights = {x:torch.tensor(1 - (y / y.sum())).float() for x, y in entity_counts.items()}
    if dataset._encoder_attributes is not None:
        attribute_weights = {x:torch.tensor(1 - (y / y.sum())).float() for x, y in attribute_counts.items()}
    ## If No Weighting, Set as Ones
    if entity_weights is not None:
        if weighting_entity is None:
            entity_weights = {x:torch.ones_like(y) for x, y in entity_weights.items()}
        else:
            if weighting_entity != "balanced":
                raise NotImplementedError("Weighting not supported.")
    if attribute_weights is not None:
        if weighting_attribute is None:
            attribute_weights = {x:torch.ones_like(y) for x, y in attribute_weights.items()}
        else:
            if weighting_attribute != "balanced":
                raise NotImplementedError("Weighting not supported.")
    ## Return
    return entity_weights, attribute_weights

def initialize_transition_probabilities(dataset,
                                        model,
                                        alpha=1,
                                        entity_weights=None,
                                        informed_prior=True):
    """

    """
    ## Base Cases
    if dataset._encoder_entity is None:
        return model
    if not model._use_crf:
        return model
    elif model._use_crf and not informed_prior:
        return model
    ## Gather Start/End/Transition Counts
    task_counts = [[np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros((3, 3), dtype=int)] for task in dataset._encoder_entity.get_tasks()]
    for x in dataset:
        for t, tlbls in enumerate(x["entity_labels"]):
            task_counts[t][0][tlbls[0].item()] += 1
            task_counts[t][1][tlbls[-1].item()] += 1
            for i_1, i in zip(tlbls[:-1], tlbls[1:]):
                task_counts[t][2][i_1,i] += 1
    ## Reweighting of Transition Counts
    if entity_weights is not None:
        for i, (s, e, t) in enumerate(task_counts):
             task_counts[i][2] = t * entity_weights[dataset._encoder_entity.get_tasks()[i]].numpy()
    ## Initialization
    for i, (s, e, t) in enumerate(task_counts):
        s_norm =  torch.tensor(np.log((s + alpha) / (s + alpha).sum()))
        e_norm = torch.tensor(np.log((e + alpha) / (e + alpha).sum()))
        t_norm = torch.tensor(np.log((t + alpha) / (t + alpha).sum(axis=1, keepdims=True)))
        for j in range(3):
            model.entity_crf[i].start_transitions[j].data.fill_(s_norm[j])
            model.entity_crf[i].end_transitions[j].data.fill_(e_norm[j])
            for k in range(3):
                model.entity_crf[i].transitions[j,k].data.fill_(t_norm[j,k])
    return model

def initialize_optimizer(model,
                         lr=0.01,
                         weight_decay=1e-3,
                         lr_gamma=0.9,
                         lr_step_size=10,
                         momentum=0.9,
                         nesterov=False,
                         method="adam",
                         lr_adaptive_method="exponential"):
    """

    """
    ## Optimizer
    if method == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
    elif method == "adamw":
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
    elif method == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov,
                                    weight_decay=weight_decay)
    else:
        raise NotImplementedError("Provided an optimization method that isn't supported.")
    ## Adapative Learning Rate
    scheduler = None
    if lr_adaptive_method is not None:
        if lr_adaptive_method == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        elif lr_adaptive_method == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_gamma, step_size=lr_step_size)
        elif lr_adaptive_method == "cosine_annealing_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=lr_step_size, eta_min=0.0001*lr)
        else:
            raise NotImplementedError("Learning rate scheduler not supported: {}".format(lr_adaptive_method))
    ## Return
    return optimizer, scheduler

def move_class_weights_to_device(entity_weights,
                                 attribute_weights,
                                 device):
    """

    """
    if entity_weights is not None:
        entity_weights = {x:y.to(device) for x, y in entity_weights.items()}
    if attribute_weights is not None:
        attribute_weights = {x:y.to(device) for x, y in attribute_weights.items()}
    return entity_weights, attribute_weights

def _compute_training_loss(batch_inds,
                           dataset,
                           model,
                           entity_weights,
                           attribute_weights,
                           training_device,
                           use_first_index=False,
                           accumulated_loss=None,
                           take_step=True):
    """

    """
    ## Check
    if model._encoder_entity is None and model._encoder_attributes is None:
        raise ValueError("Cannot have a null entity encoder and attribute encoder.")
    ## Collation
    batch = collate_entity_attribute([dataset["train"][idx] for idx in batch_inds],
                                     use_first_index=use_first_index)
    ## Update Device
    batch = move_batch_to_device(batch, training_device)
    ## Forward Pass
    entity_logits, attribute_logits = model(batch)
    ## Initialize Loss
    total_loss = 0
    ## Loss Computation
    if model._encoder_entity is not None:
        total_loss += compute_entity_loss(model=model,
                                          entity_weights=entity_weights,
                                          entity_logits=entity_logits,
                                          inputs=batch,
                                          reduce=True)
    if model._encoder_attributes is not None:
        total_loss += compute_attribute_loss(model=model,
                                             attribute_weights=attribute_weights,
                                             attribute_logits=attribute_logits,
                                             inputs=batch,
                                             reduce=True)
    ## Return
    return total_loss

def train(dataset,
          model,
          lr=0.00005,
          lr_gamma=0.9,
          lr_warmup=100,
          lr_step_size=250,
          lr_adaptive_method=None,
          opt_method="adamw",
          nesterov=False,
          momentum=0.9,
          weight_decay=0.00001,
          grad_clip_norm=None,
          max_epochs=10,
          max_steps=None,
          train_batch_size=16,
          train_gradient_accumulation=None,
          eval_batch_size=16,
          eval_frequency=25,
          eval_strategy="steps",
          eval_train=False,
          eval_test=False,
          save_criteria=None,
          save_models=False,
          save_predictions=False,
          weighting_entity=None,
          weighting_attribute=None,
          weighting_entity_gamma=1,
          weighting_attribute_gamma=1,
          use_crf_informed_prior=True,
          use_first_index=False,
          random_state=42,
          early_stopping_tol=0.001,
          early_stopping_patience=5,
          early_stopping_warmup=None,
          early_stopping_criteria="loss",
          display_cumulative_batch_loss=False,
          model_init=None,
          checkpoint_dir=None,
          gpu_id=None,
          no_training=False,
          eval_first_update=True):
    """

    """
    ## Check
    if save_criteria is not None and not (save_models or save_predictions):
        raise ValueError("Must specify save_models=True or save_predictions=True if save_criteria is not None.")
    if early_stopping_criteria not in ["f1","loss"]:
        raise ValueError("Must specify early_stopping_criteria in ['f1', 'loss']")
    ## Device Formatting
    training_device = get_device(gpu_id)
    print(f">> WARNING: Using Following Device for Training -- {training_device}")
    ## Validate
    if eval_strategy not in ["steps","epochs"]:
        raise KeyError("Expected eval_strategy to be either 'steps' or 'epochs'")
    ## Initialize Sampler Seed
    seed = np.random.RandomState(random_state)
    ## Initialize Checkpoint Directory
    if checkpoint_dir is not None:
        ## Remove Existing
        existing_checkpoints = glob(f"{checkpoint_dir}/checkpoint-*")
        if len(existing_checkpoints) > 0:
            print(">> WARNING - Removing {:,d} Existing Checkpoints".format(len(existing_checkpoints)))
            for ec in existing_checkpoints:
                _ = os.system(f"rm -rf {ec}")
        if not os.path.exists(checkpoint_dir):
            _ = os.makedirs(checkpoint_dir)
    ## Training Limits
    if max_steps is None and max_epochs is None:
        raise ValueError("Must specify either max_steps or max_epochs.")
    elif max_steps is None:
        max_steps = int(1e6)
        limiter = "epochs"
    elif max_epochs is None:
        max_epochs = int(1e6)
        limiter = "steps"
    else:
        limiter = "min"
    ## Initialize Training Loop Items
    if model_init is None:
        ## Number of Steps / Epochs / Training Cache
        training_log = []
        n_steps = 0
        epoch_loop = list(range(max_epochs))
    else:
        ## Load Existing Log
        training_log = torch.load(f"{model_init}/train.log.pt")
        ## Identify Number of Steps Run
        n_steps = max([i["steps"] for i in training_log])
        ## Epochs Completed And Remaining
        last_epoch = max([i["epoch"] for i in training_log]) + 1
        max_epochs = max_epochs + last_epoch
        epoch_loop = list(range(last_epoch, max_epochs))
    ## Early Exit
    if no_training:
        return training_log, model
    ## Initialize Class Weights
    entity_weights, attribute_weights = initialize_class_weights(dataset=dataset["train"],
                                                                 weighting_entity=weighting_entity,
                                                                 weighting_attribute=weighting_attribute,
                                                                 gamma_entity=weighting_entity_gamma,
                                                                 gamma_attribute=weighting_attribute_gamma)
    ## Initialize Transition Probabilities
    if model_init is None:
        model = initialize_transition_probabilities(dataset=dataset["train"],
                                                    model=model,
                                                    entity_weights=entity_weights,
                                                    alpha=1,
                                                    informed_prior=use_crf_informed_prior) ## False -> Just use random initialization
    ## Device Setup
    model = model.to(training_device)
    entity_weights, attribute_weights = move_class_weights_to_device(entity_weights,
                                                                     attribute_weights,
                                                                     training_device)
    ## Initialize Optimizer/Sheduler
    optimizer, scheduler = initialize_optimizer(model=model,
                                                lr=lr,
                                                weight_decay=weight_decay,
                                                lr_gamma=lr_gamma,
                                                lr_step_size=lr_step_size,
                                                lr_adaptive_method=lr_adaptive_method,
                                                method=opt_method,
                                                nesterov=nesterov,
                                                momentum=momentum)
    ## Reset Optimizer
    optimizer.zero_grad()
    ## Initialize Tracking Variables
    accum_steps = 0
    cache_steps = 0
    last_update_reached = False
    ## Best Model Information
    best_checkpoint_dir = None
    best_checkpoint_loss_new = False
    best_checkpoint_f1_new = False
    ## Early Stopping Information
    best_dev_loss = None ## Best Loss Seen
    best_dev_loss_steps = 0 ## Number of Evaluations With Best Loss
    best_dev_macro_f1 = None
    best_dev_macro_f1_steps = 0
    early_stopping_triggered = None
    ## Check Effective Batch Size
    if train_gradient_accumulation is not None and train_batch_size * train_gradient_accumulation > len(dataset["train"]):
        raise ValueError("You cannot specify an effective batch size larger than training dataset. Consider lowering batch size or train_gradient_accumulation.")
    ## Training Loop
    lr_val = lr
    training_index = list(range(len(dataset["train"])))
    for epoch in epoch_loop:
        ## Check for Last Update Reached
        if last_update_reached:
            break
        ## Check Early Stopping
        if early_stopping_triggered is not None:
            break
        ## Shuffle and Create Mini-Batches
        _ = seed.shuffle(training_index)
        train_batches = list(chunks(training_index, train_batch_size))
        ## Initialize Batch-Specific Loss
        batch_train_loss = 0
        batch_instances_seen = 0
        for b, batch_inds in enumerate(train_batches):
            ## Update Accumulation Steps
            accum_steps += 1
            ## Update Info (First)
            is_first_update = (epoch == 0) and (b == 0)
            ## Update Info (Last)
            is_last_update = False
            if (limiter == "epochs" or limiter == "min") and ((epoch == epoch_loop[-1]) and (b == len(train_batches) - 1)):
                is_last_update = True
            elif (limiter == "steps" or limiter == "min") and ((n_steps == (max_steps - 1)) and (train_gradient_accumulation is None or accum_steps == train_gradient_accumulation)):
                is_last_update = True
            ## Update Accumulation Steps
            accum_take_step = (train_gradient_accumulation is None) or (accum_steps == train_gradient_accumulation) or (is_last_update)
            ## Verbose Training
            print("Epoch {} | Batch {}/{}".format(epoch+1, b+1, len(train_batches)))
            ## Try Full Batch
            try:
                this_batch_loss = _compute_training_loss(batch_inds=batch_inds,
                                                         dataset=dataset,
                                                         model=model,
                                                         entity_weights=entity_weights,
                                                         attribute_weights=attribute_weights,
                                                         training_device=training_device,
                                                         use_first_index=use_first_index)
                ## Track Loss
                batch_train_loss += this_batch_loss.detach().data * len(batch_inds) if not isinstance(this_batch_loss, float) else this_batch_loss
                batch_instances_seen += len(batch_inds)
                ## Backprop
                if this_batch_loss is not None:
                    ## Normalize Loss by Number of Accumulations
                    if train_gradient_accumulation is not None:
                        this_batch_loss = this_batch_loss / train_gradient_accumulation
                    ## Backprop
                    _ = this_batch_loss.backward()
                ## Take a Step
                if accum_take_step:
                    if grad_clip_norm is not None:
                        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    ## Take step + reset gradients
                    print(">> Taking Step")
                    optimizer.step()
                    optimizer.zero_grad()
                    ## Update Tracking
                    accum_steps = 0
                    n_steps += 1
            except RuntimeError as e:
                raise e
            except Exception as e:
                raise e
            ## Progress
            if display_cumulative_batch_loss and accum_take_step:
                print(">> Average Batch Training Loss ({:,d} Steps): {:.4f}".format(n_steps, batch_train_loss / batch_instances_seen))
            ## Evaluation
            if ((is_first_update and eval_first_update) or is_last_update or (accum_take_step and eval_strategy == "steps" and n_steps % eval_frequency == 0) or (eval_strategy == "epochs" and b == len(train_batches) - 1 and (epoch + 1) % eval_frequency == 0)):
                ## Time of Evaluation
                eval_time = datetime.now().isoformat()
                ## Prediction Cache
                all_split_entity_predictions = []
                all_split_attribute_predictions = []
                all_split_performance_summary = {}
                ## Run Evaluation
                for split in ["train","dev","test"]:
                    ## Skip Testing If Not Appropriate
                    if split == "test" and not eval_test:
                        continue
                    ## Skip Training if not Desired
                    if split == "train" and not eval_train:
                        continue
                    ## Get Performance
                    split_performance = evaluate(model=model,
                                                 dataset=dataset[split],
                                                 batch_size=eval_batch_size,
                                                 entity_weights=entity_weights,
                                                 attribute_weights=attribute_weights,
                                                 desc=split.title(),
                                                 device=training_device,
                                                 use_first_index=use_first_index,
                                                 ## NOTE: ADDED START
                                                 return_predictions=save_predictions
                                                 ## NOTE: ADDED END
                                                 )
                    ## Prediction Formatted
                    if save_predictions:
                        if split_performance["valid"]["entity"]:
                            print("[Formatting Entity Predictions: {}]".format(split))
                            split_entity_predictions = format_entity_predictions(entity_predictions=split_performance["entity"]["predictions"],
                                                                                 dataset=dataset[split],
                                                                                 vocab2ind=model._token_encoder_vocab)
                            split_entity_predictions["split"] = split
                            all_split_entity_predictions.append(split_entity_predictions)
                            ## Drop Predictions from Performance Dict
                            _ = split_performance["entity"]["predictions"] = None
                        if split_performance["valid"]["attributes"]:
                            print("[Formatting Attribute Predictions: {}]".format(split))
                            split_attribute_predictions = format_attribute_predictions(attribute_predictions=split_performance["attributes"]["predictions"],
                                                                                       dataset=dataset[split],
                                                                                       vocab2ind=model._token_encoder_vocab)
                            split_attribute_predictions["split"] = split
                            all_split_attribute_predictions.append(split_attribute_predictions)
                            ## Drop Predictions
                            split_performance["attributes"]["predictions"] = None
                    ## Show User Performance
                    print("*"*50 + f" {split.title()} Performance " + "*"*50)
                    all_split_performance_summary[split] = display_evaluation(split_performance)
                    ## Append Metadata
                    split_performance["epoch"] = epoch
                    split_performance["steps"] = n_steps
                    split_performance["split"] = split
                    split_performance["eval_time"] = eval_time
                    split_performance["learning_rate"] = lr_val
                    ## Cache
                    training_log.append(split_performance)
                    ## Early Stopping Check (Increasing Validation Loss)
                    if split == "dev":
                        ## Calculate Overall Dev Loss Over Tasks
                        overall_dev_loss = []
                        overall_dev_macro_f1 = []
                        if split_performance["valid"]["entity"]:
                            overall_dev_loss.append(split_performance["entity"]["loss"])
                            overall_dev_macro_f1.append(split_performance["entity"]["scores_entity"]["strict"]["f1-score"])
                        if split_performance["valid"]["attributes"]:
                            for task, task_loss in split_performance["attributes"]["loss"].items():
                                overall_dev_loss.append(task_loss)
                            for task, task_scores in split_performance["attributes"]["scores"].items():
                                overall_dev_macro_f1.append(task_scores["macro avg"]["f1-score"])
                        ## Type Validation
                        if best_dev_loss is not None and len(best_dev_loss) != len(overall_dev_loss):
                            raise ValueError("This needs to be fixed.")
                        if best_dev_macro_f1 is not None and len(best_dev_macro_f1) != len(overall_dev_macro_f1):
                            raise ValueError("This needs to be fixed.")
                        ## Comparisons
                        if best_dev_loss is None or any(ov < bdl * (1 - early_stopping_tol) for ov, bdl in zip(overall_dev_loss, best_dev_loss)):
                            best_dev_loss = overall_dev_loss
                            best_dev_loss_steps = 0
                            best_checkpoint_loss_new = True
                        else:
                            best_dev_loss_steps += 1
                            best_checkpoint_loss_new = False
                        if best_dev_macro_f1 is None or any(ov > bdl * (1 + early_stopping_tol) for ov, bdl in zip(overall_dev_macro_f1, best_dev_macro_f1)):
                            best_dev_macro_f1 = overall_dev_macro_f1
                            best_dev_macro_f1_steps = 0
                            best_checkpoint_f1_new = True
                        else:
                            best_dev_macro_f1_steps += 1
                            best_checkpoint_f1_new = False
                        ## Early Stopping Check
                        if (early_stopping_warmup is None or n_steps >= early_stopping_warmup):
                            if best_dev_loss_steps >= early_stopping_patience and early_stopping_criteria == "loss":
                                early_stopping_triggered = "dev loss"
                            elif best_dev_macro_f1_steps >= early_stopping_patience and early_stopping_criteria == "f1":
                                early_stopping_triggered = "dev f1"
                ## Model Checkpoint
                if checkpoint_dir is not None and save_criteria is not None:
                    ## Save
                    if save_criteria == "all" or (save_criteria == "f1" and best_checkpoint_f1_new) or (save_criteria == "loss" and best_checkpoint_loss_new):
                        ## Remove Old Best Checkpoint
                        if save_criteria != "all" and best_checkpoint_dir is not None:
                            print(f">> Removing previous best checkpoint: '{best_checkpoint_dir}")
                            _ = os.system(f"rm -rf {best_checkpoint_dir}")
                        ## Update Best Checkpoint
                        best_checkpoint_dir = f"{checkpoint_dir}/checkpoint-{n_steps}/"
                        print(f">> Saving new checkpoint: '{best_checkpoint_dir}'")
                        if os.path.exists(best_checkpoint_dir):
                            _ = os.system(f"rm -rf {best_checkpoint_dir}")
                        _ = os.makedirs(best_checkpoint_dir)
                        if save_models:
                            _ = torch.save(model.state_dict(), f"{best_checkpoint_dir}/model.pt")
                            _ = torch.save(training_log, f"{best_checkpoint_dir}/train.log.pt")                    
                        if save_predictions:
                            if len(all_split_entity_predictions) > 0:
                                all_split_entity_predictions = pd.concat(all_split_entity_predictions, axis=0, ignore_index=True)
                                _ = all_split_entity_predictions.to_json(f"{best_checkpoint_dir}/predictions.entity.json", index=False, orient="records", indent=5)
                            if len(all_split_attribute_predictions) > 0 :
                                all_split_attribute_predictions = pd.concat(all_split_attribute_predictions, axis=0, ignore_index=True)
                                _ = all_split_attribute_predictions.to_json(f"{best_checkpoint_dir}/predictions.attributes.json", index=False, orient="records", indent=5)
                        ## Save Summary
                        for split, split_summary in all_split_performance_summary.items():
                            with open(f"{best_checkpoint_dir}/summary.{split}.txt","w") as the_file:
                                the_file.write(split_summary)
            ## Scheduler Step
            if scheduler is not None and (lr_warmup is None or n_steps >= lr_warmup):
                scheduler.step()
            ## Learning Rate
            cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
            if lr_val != cur_lr:
                print(">> Updating Learning Rate: {} to {}".format(lr_val, cur_lr))
                lr_val = cur_lr
            ## Last Update
            if is_last_update:
                print(">> Reached maximum update. Exiting training loop.")
                last_update_reached = True
                break
            ## Early Stopping
            if early_stopping_triggered is not None:
                print(f">> Early stopping event triggered ({early_stopping_triggered} stopped improving). Exiting training loop.")
                break
    ## Return
    return training_log, model

def train_baseline(dataset,
                   eval_train=False,
                   eval_test=False,
                   weighting_entity=None,
                   weighting_attribute=None,
                   use_char=False):
    """

    """
    ## Initialize Model
    baseline_model = BaselineTaggerModel(encoder_entity=dataset["train"]._encoder_entity,
                                         encoder_attributes=dataset["train"]._encoder_attributes,
                                         use_char=use_char)
    ## Learn Distribution
    baseline_model = baseline_model.fit(dataset["train"])
    ## Initialize Class Weights
    entity_weights, attribute_weights = initialize_class_weights(dataset=dataset["train"],
                                                                 weighting_entity=weighting_entity,
                                                                 weighting_attribute=weighting_attribute)
    ## Iterate Through Baseline Application Scnarios
    baseline_logs = {}
    for mode in ["task","entity","token"]:
        baseline_logs[mode] = []
        baseline_model = baseline_model.set_mode(mode)
        for split in ["train","dev","test"]:
            ## Check
            if split == "test" and not eval_test:
                continue
            if split == "train" and not eval_train:
                continue
            ## Compute
            baseline_split_performance = evaluate(model=baseline_model,
                                                  dataset=dataset[split],
                                                  batch_size=16,
                                                  entity_weights=entity_weights,
                                                  attribute_weights=attribute_weights,
                                                  desc=split.title(),
                                                  device="cpu",
                                                  use_first_index=False)
            ## Metadata
            baseline_split_performance["epoch"] = 0
            baseline_split_performance["steps"] = 1
            baseline_split_performance["split"] = split
            baseline_split_performance["eval_time"] = datetime.now().isoformat()
            baseline_split_performance["learning_rate"] = 0
            baseline_logs[mode].append(baseline_split_performance)
            ## Show Performance
            print("*"*50 + f" {mode.title()} -- {split.title()} Performance " + "*"*50)
            _ = display_evaluation(baseline_split_performance)
    ## Return
    return baseline_logs, baseline_model

def _sample_splits(preprocessed_data,
                   data,
                   metadata=None,
                   eval_cv=5,
                   eval_cv_groups=None,
                   random_state=42,
                   verbose=True):
    """

    """
    ## Accepted Document IDs Found in Dataset (Not Used for Splitting)
    accept_doc_ids = set(preprocessed_data["document_id"])
    ## Get Groups of Documents Which Should Be Kept Together (Splitting Criteria)
    if eval_cv_groups is None:
        document_groups = {x["document_id"]:i for i, x in enumerate(data)}
    else:
        if metadata is None:
            raise ValueError("Must provide metadata.")
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple, axis=1).to_dict()
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
        train_doc = set.union(*[document_groups_r[g] for g in train_g]) & accept_doc_ids
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g]) & accept_doc_ids
        test_doc = set.union(*[document_groups_r[g] for g in test_g]) & accept_doc_ids
        ## Check Mutual Exclusion
        assert len(train_doc & dev_doc) == 0
        assert len(train_doc & test_doc) == 0
        assert len(dev_doc & test_doc) == 0
        ## Store
        splits[k] = {"train":train_doc, "dev":dev_doc, "test":test_doc}
    ## Return
    return splits

def _sample_splits_monte_carlo(preprocessed_data,
                               data,
                               metadata=None,
                               eval_cv=10,
                               eval_cv_groups=None,
                               split_frac=[7,2,1],
                               random_state=42,
                               verbose=True):
    """

    """
    ## Accepted Document IDs
    accept_doc_ids = set(preprocessed_data["document_id"])
    ## Format Split Frac
    if not all(isinstance(i, int) for i in split_frac):
        raise ValueError("Expected integer ratios for split_frac")
    split_frac_sum = sum(split_frac)
    split_frac = [x / split_frac_sum for x in split_frac]
    ## Get Groups of Documents Which Should Be Kept Together
    if eval_cv_groups is None:
        document_groups = {x["document_id"]:i for i, x in enumerate(data)}
    else:
        if metadata is None:
            raise ValueError("Must provide metadata.")
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple, axis=1).to_dict()
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
        train_doc = set.union(*[document_groups_r[g] for g in train_g]) & accept_doc_ids
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g]) & accept_doc_ids
        test_doc = set.union(*[document_groups_r[g] for g in test_g]) & accept_doc_ids
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
    ## Accepted Document IDs
    accept_doc_ids = set(preprocessed_data["document_id"])
    ## Get Groups of Documents Which Should Be Kept Together
    if eval_cv_groups is None:
        document_groups = {x["document_id"]:i for i, x in enumerate(data)}
    else:
        if metadata is None:
            raise ValueError("Must provide metadata.")
        if any(g not in metadata.columns for g in eval_cv_groups):
            raise KeyError("Not all parameters in eval_cv_groups found in metadata.")
        document_groups = metadata.set_index("document_id")[eval_cv_groups].apply(tuple, axis=1).to_dict()
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
    group_labels_oh = group_labels_oh.sort_index(axis=0).sort_index(axis=1)
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
        train_doc = set.union(*[document_groups_r[g] for g in train_g]) & accept_doc_ids
        dev_doc = set.union(*[document_groups_r[g] for g in dev_g]) & accept_doc_ids
        test_doc = set.union(*[document_groups_r[g] for g in test_g]) & accept_doc_ids
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

def sample_splits(split_method,
                  **kwargs):
    """

    """
    ## Get Function
    func = {
        "k_fold":_sample_splits,
        "monte_carlo":_sample_splits_monte_carlo,
        "stratified_k_fold":_sample_splits_stratified,
        "stratified_monte_carlo":_sample_splits_stratified_monte_carlo
    }.get(split_method)
    if func is None:
        raise KeyError(f"Split method not recognized: '{split_method}'")
    ## Apply
    return func(**kwargs)