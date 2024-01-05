
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
                             alpha=1):
    """
    
    """
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
                                        alpha=1):
    """

    """
    if dataset._encoder_entity is None:
        return model
    if not model._use_crf:
        return model
    task_counts = [[np.zeros(3, dtype=int), np.zeros(3, dtype=int), np.zeros((3, 3), dtype=int)] for task in dataset._encoder_entity.get_tasks()]
    for x in dataset:
        for t, tlbls in enumerate(x["entity_labels"]):
            task_counts[t][0][tlbls[0].item()] += 1
            task_counts[t][1][tlbls[-1].item()] += 1
            for i_1, i in zip(tlbls[:-1], tlbls[1:]):
                task_counts[t][2][i_1,i] += 1
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
          weighting_entity=None,
          weighting_attribute=None,
          use_first_index=False,
          random_state=42,
          early_stopping_tol=0.001,
          early_stopping_patience=5,
          early_stopping_warmup=None,
          display_cumulative_batch_loss=False,
          model_init=None,
          checkpoint_dir=None,
          gpu_id=None,
          no_training=False,
          eval_first_update=True):
    """

    """
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
                                                                 weighting_attribute=weighting_attribute)
    ## Initialize Transition Probabilities
    if model_init is None:
        model = initialize_transition_probabilities(dataset=dataset["train"],
                                                    model=model)
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
            if ((is_first_update and eval_first_update) or is_last_update or (accum_take_step and eval_strategy == "steps" and n_steps % eval_frequency == 0) or (accum_take_step and eval_strategy == "epochs" and b == len(train_batches) - 1 and (epoch + 1) % eval_frequency == 0)):
                ## Time of Evaluation
                eval_time = datetime.now().isoformat()
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
                                                 use_first_index=use_first_index)
                    ## Show User Performance
                    print("*"*50 + f" {split.title()} Performance " + "*"*50)
                    _ = display_evaluation(split_performance)
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
                        overall_dev_loss = 0
                        overall_dev_macro_f1 = [0, 0]
                        if split_performance["valid"]["entity"]:
                            overall_dev_loss += split_performance["entity"]["loss"]
                            overall_dev_macro_f1[0] += 1
                            overall_dev_macro_f1[1] += split_performance["entity"]["scores_entity"]["strict"]["f1-score"]
                        if split_performance["valid"]["attributes"]:
                            for task, task_loss in split_performance["attributes"]["loss"].items():
                                overall_dev_loss += task_loss
                            for task, task_scores in split_performance["attributes"]["scores"].items():
                                task_macro_f1 = task_scores["macro avg"]["f1-score"]
                                if not pd.isnull(task_macro_f1):
                                    overall_dev_macro_f1[0] += 1
                                    overall_dev_macro_f1[1] += task_macro_f1
                        overall_dev_macro_f1 = overall_dev_macro_f1[1] / overall_dev_macro_f1[0] if overall_dev_macro_f1[0] > 0 else 0
                        ## Loss Comparison
                        if best_dev_loss is None or overall_dev_loss < best_dev_loss * (1 - early_stopping_tol):
                            best_dev_loss = overall_dev_loss
                            best_dev_loss_steps = 0
                            best_checkpoint_loss_new = True
                        else:
                            best_dev_loss_steps += 1
                            best_checkpoint_loss_new = False                  
                        ## Macro F1 Comparison
                        if best_dev_macro_f1 is None or overall_dev_macro_f1 > best_dev_macro_f1 * (1 + early_stopping_tol):
                            best_dev_macro_f1 = overall_dev_macro_f1
                            best_dev_macro_f1_steps = 0
                            best_checkpoint_f1_new = True
                        else:
                            best_dev_macro_f1_steps += 1
                            best_checkpoint_f1_new = False
                        ## Early Stopping Check
                        if (early_stopping_warmup is None or n_steps >= early_stopping_warmup):
                            if best_dev_loss_steps >= early_stopping_patience and best_dev_macro_f1_steps >= early_stopping_patience:
                                early_stopping_triggered = "dev loss and macro f1"
                            elif best_dev_loss_steps >= early_stopping_patience:
                                early_stopping_triggered = "dev loss"
                            elif best_dev_macro_f1_steps >= early_stopping_patience:
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
                        _ = os.makedirs(best_checkpoint_dir)
                        _ = torch.save(model.state_dict(), f"{best_checkpoint_dir}/model.pt")
                        _ = torch.save(training_log, f"{best_checkpoint_dir}/train.log.pt")                    
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