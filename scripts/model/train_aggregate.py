
"""
Aggregate results over cross validation folds. Allows inspection of training
before full number of epochs complete.
"""

#########################
### Imports
#########################

## Standard Library
import os
import json
import argparse
from glob import glob
from collections import Counter
from time import sleep

## External
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from torch import load as load_torch

#########################
### Functions
#########################

def parse_command_line():
    """

    """
    ## Parser
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--results_dir", type=str, default=None, help="Path to --output_dir from training run.")
    _ = parser.add_argument("--log_files", type=str, default=None, nargs="+", help="If desired, provide log files directory.")
    _ = parser.add_argument("--eval_strategy", type=str, default=None, choices={"epoch","steps"})
    _ = parser.add_argument("--model", type=str, choices={"model","baseline-task","baseline-entity","baseline-token"})
    _ = parser.add_argument("--output_dir", type=str, default=None, help="Where to store aggregate results/visualizations.")
    _ = parser.add_argument("--dpi", type=int, default=100, help="Pixel density for saving PNG files.")
    _ = parser.add_argument("--pdf", action="store_true", default=False, help="If included, save as PDF files instead of PNG files.")
    _ = parser.add_argument("--rm_existing", action="store_true", default=False, help="Must include to overwrite existing output directory.")
    _ = parser.add_argument("--plot", action="store_true", default=False, help="If included, will generate plots. Otherwise, just does stat summarization.")
    _ = parser.add_argument("--oracle", action="store_true", default=False, help="If included, will select optimal step for each fold based on test data (if available) instead of development.")
    _ = parser.add_argument("--expected_folds", type=int, default=None, help="Number of expected output files.")
    _ = parser.add_argument("--allow_checkpoints", action="store_true", default=False)
    args = parser.parse_args()
    ## Verfication
    if args.results_dir is None or not os.path.exists(args.results_dir):
        raise FileNotFoundError("Could not find --results_dir")
    if args.output_dir is None:
        raise FileNotFoundError("Must provide an --output_dir")
    if os.path.exists(args.output_dir) and not args.rm_existing:
        raise FileExistsError("Must include --rm_existing flag to overwrite an existing output directory.")
    if args.eval_strategy is None:
        raise ValueError("Must specify the expected --eval_strategy for aggregating curves.")
    ## Return
    return args

def load_label_distribution(results_dir):
    """

    """
    ## Search
    label_files = glob(f"{results_dir}/fold-*/labels.csv")
    ## Check
    if len(label_files) == 0:
        print("[WARNING: No Label distribution files.]")
        return None
    ## Load
    labels = []
    for lf in label_files:
        lf_data = pd.read_csv(lf)
        lf_data["fold"] = int(os.path.dirname(lf).split("fold-")[1])
        labels.append(lf_data)
    ## Merge
    labels = pd.concat(labels,axis=0).reset_index(drop=True)
    ## Return
    return labels

def plot_entity_validity_distribution(labels):
    """

    """
    ## Compute
    label_dist = pd.pivot_table(labels, index=["fold","label","split"],columns=["valid"],values="start",aggfunc=len).fillna(0).astype(int)
    for col in [False, True]:
        if col not in label_dist.columns:
            label_dist[col] = 0
    label_dist["format"] = label_dist.apply(lambda row: "{:,d}/{:,d}".format(row[True], row.sum()), axis=1)
    label_dist["precision"] = label_dist[True] / label_dist.iloc[:,:2].sum(axis=1)
    label_dist = label_dist.unstack()
    ## Plot
    folds = label_dist.index.levels[0]
    fig, ax = plt.subplots(1, len(folds), figsize=(max(15, len(folds) * 3),10), sharex=True, sharey=True)
    ax = ax if len(folds) > 1 else [ax]
    for f, fold in enumerate(folds):
        ax[f].imshow(label_dist.loc[fold]["precision"].fillna(0).loc[:,["train","dev","test"]],
                     cmap=plt.cm.Purples,
                     alpha=0.3,
                     interpolation="nearest",
                     aspect="auto") 
        for r, row in enumerate(label_dist.loc[fold]["format"].loc[:,["train","dev","test"]].values):
            for c, cell in enumerate(row):
                if pd.isnull(cell):
                    ax[f].text(c, r, "None", ha="center", va="center", fontsize=6)
                else:
                    ax[f].text(c, r, cell, ha="center", va="center", fontsize=6)
        ax[f].set_xticks(range(3))
        ax[f].set_xticklabels(["Train","Dev","Test"])
        if f == 0:
            ax[f].set_yticks(range(label_dist.loc[fold].shape[0]))
            ax[f].set_yticklabels(label_dist.loc[fold].index.tolist())
            ax[f].set_ylabel("Entity", fontweight="bold")
        ax[f].set_title(f"Fold: {fold}", fontweight="bold")
    fig.tight_layout()
    return fig

def plot_attribute_distribution(labels):
    """

    """
    ## Get Task Columns
    task_cols = labels.drop([
        "start",
        "end",
        "label",
        "in_header",
        "in_autolabel_postprocess",
        "valid",
        "tokens",
        "tokens_boundaries",
        "split",
        "fold",
        "document_id"
    ], axis=1).columns.tolist()
    task_cols = sorted(task_cols)
    ## Aggregate
    labels_task_dist = pd.concat([labels.groupby(["fold","split",tc]).size().to_frame(tc).unstack() for tc in task_cols], axis=1).T
    folds = labels_task_dist.columns.levels[0]
    ## Plot
    fig, ax = plt.subplots(len(task_cols), len(folds), figsize=(max(15,len(folds) * 3), len(task_cols) * 2.5), sharex=True, sharey=False)
    for t, tc in enumerate(task_cols):
        for f, fold in enumerate(folds):
            if len(task_cols) == 1 and len(folds) == 1:
                pax = ax
            elif len(task_cols) == 1:
                pax = ax[f]
            elif len(folds) == 1:
                pax = ax[t]
            else:
                pax = ax[t, f]
            tf_dist = labels_task_dist.loc[tc][fold].loc[:,["train","dev","test"]].fillna(0).astype(int)
            tf_dist_norm = tf_dist.apply(lambda x: np.nan if x.sum() == 0 else x / x.sum(), axis=0)
            pax.imshow(tf_dist_norm,
                       alpha=0.3,
                       cmap=plt.cm.Purples,
                       interpolation="nearest",
                       aspect="auto")
            for r, row in enumerate(tf_dist.values):
                for c, cell in enumerate(row):
                    if cell == 0:
                        continue
                    pax.text(c, r, "{:,d}".format(cell), ha="center", va="center")
            pax.set_yticks(range(tf_dist.shape[0]))
            if f == 0:
                pax.set_yticklabels([str(i).replace("/","/\n") for i in tf_dist.index.tolist()])
                pax.set_ylabel(tc, fontweight="bold", labelpad=20)
            else:
                pax.set_yticklabels([])
            pax.set_xticks(range(3))
            if t == len(task_cols) - 1:
                pax.set_xticklabels(["Train","Dev","Test"], rotation=45, ha="right")
            if t == 0:
                pax.set_title("Fold: {}".format(fold), fontweight="bold")
    fig.tight_layout()
    return fig

def plot_overall_label_summary(labels):
    """

    """
    ## Deduplicate
    labels_unique = labels.drop_duplicates(subset=["document_id","label","start","end"],keep="first")
    ## Tasks
    task_cols = labels.drop([
        "start",
        "end",
        "label",
        "in_header",
        "in_autolabel_postprocess",
        "valid",
        "tokens",
        "tokens_boundaries",
        "split",
        "fold",
        "document_id"
    ], axis=1).columns.tolist()
    task_cols = sorted(list(filter(lambda tc: not tc.startswith("validity_"), task_cols)))
    task_concepts = sorted(labels_unique["label"].unique())
    ## Initialize Counts
    counts = {}
    for task in ["concept_extraction"] + task_cols:
        counts[task] = {tc:Counter() for tc in task_concepts}
    ## Gather Counts
    for _, row in labels_unique.iterrows():
        ## Row Concept
        row_concept = row["label"]
        ## Validity
        row_validity = {True:"Valid",False:"Invalid"}.get(row["valid"])
        counts["concept_extraction"][row_concept][row_validity] += 1
        ## Other Tasks
        for tc in task_cols:
            row_tc = row[tc]
            if pd.isnull(row_tc):
                continue
            counts[tc][row_concept][row_tc] += 1
    ## Format Counts
    counts = pd.concat({x:pd.DataFrame(y) for x, y in counts.items()}).fillna(0).astype(int)
    counts["Total"] = counts.sum(axis=1)
    ## Plot
    fig = plt.figure(figsize=(15, max(10, len(counts.index.levels[0]) * 2.5)), layout="constrained")
    spec = gridspec.GridSpec(ncols=1, 
                             nrows=len(counts.index.levels[0]),
                             top=0.975,
                             bottom=0.1,
                             hspace=0.01, 
                             figure=fig,
                             height_ratios=[counts.loc[i].shape[0] for i in counts.index.levels[0]])
    for i, iname in enumerate(counts.index.levels[0]):
        idata = counts.loc[iname]
        ax_i = fig.add_subplot(spec[i])
        ax_i.imshow(idata,
                    cmap=plt.cm.Purples,
                    alpha=0.3,
                    interpolation="nearest",
                    aspect="auto")
        for r, row in enumerate(idata.values):
            for c, cell in enumerate(row):
                if cell == 0:
                    continue
                ax_i.text(c, r, "{:,d}".format(cell), ha="center", va="center")
        for ii in range(idata.shape[0] - 1):
            ax_i.axhline(ii + 0.5, color="black", alpha=0.2)
        for jj in range(idata.shape[1] - 1):
            ax_i.axvline(jj + 0.5, color="black", alpha=0.2)
        ax_i.set_yticks(range(idata.shape[0]))
        ax_i.set_yticklabels([str(c).replace("/","/\n") for c in idata.index.tolist()], fontsize=14)
        if i == 0 or i == counts.index.levels[0].shape[0] - 1:
            if i == 0:
                ax_i.xaxis.tick_top()
            ax_i.set_xticks(range(idata.shape[1]))
            ax_i.set_xticklabels(idata.columns.tolist(), rotation=45, ha="left" if i == 0 else "right", fontsize=14)
        else:
            ax_i.set_xticks([])
            ax_i.set_xticklabels([])
        ax_i.set_ylabel(iname.replace("_"," ").title().replace(" ","\n"), fontweight="bold", labelpad=10, fontsize=14)
    fig.align_ylabels()
    return fig

def find_log_files(results_dir,
                   model,
                   expected=None,
                   attempts=4,
                   wait_time=1,
                   allow_checkpoints=False):
    """

    """
    ## Find Fold Directories
    fold_dirs = sorted(glob(f"{results_dir}/fold-*/"))
    ## Helper
    which_checkpoint = lambda x: int(os.path.dirname(x).split("/")[-1].split("-")[1])
    ## Filename
    fname = "train.log.pt" if model == "model" else "baseline.{}.train.log.pt".format(model.split("-")[1])
    ## Find Most Complete Training Logs
    cur_attempt = 0
    cur_wait_time = wait_time
    log_files = set()
    while True and cur_attempt < attempts:
        ## Get Logs
        for fd in fold_dirs:
            file_exists = os.path.exists(f"{fd}/{fname}")
            ## Case 1: Training Complete
            if file_exists:
                log_files.add(f"{fd}/{fname}")
            ## Case 2: Last Checkpoint
            elif not file_exists and allow_checkpoints:
                ## Sort by Checkpoint
                fd_checkpoints = sorted(glob(f"{fd}/checkpoints/*/{fname}"), key=which_checkpoint)
                ## Check for Null
                if len(fd_checkpoints) == 0:
                    continue
                ## Find Most Recent, Loadable File 
                try:
                    _ = load_torch(fd_checkpoints[-1])
                    log_files.add(fd_checkpoints[-1])
                except Exception as e:
                    ## Check If No Others Exist
                    if len(fd_checkpoints) == 1:                
                        continue
                    ## Load Penultimate
                    log_files.add(fd_checkpoints[-2])   
            else:
                pass
        ## Check
        if expected is None or len(log_files) == expected:
            break
        else:
            print(">> WARNING - Unexpected File Count. Sleeping and Retrying.")
            _ = sleep(cur_wait_time)
            cur_attempt += 1
            cur_wait_time = cur_wait_time * 2
    ## Update Log File Type
    log_files = sorted(log_files)
    ## Warning
    if expected is not None and len(log_files) != expected:
        print(f">> WARNING - Unexpected File Count: {results_dir}")
    ## Return
    if len(log_files) == 0:
        return None
    else:
        print("[Identifed {:,d} Log Files]".format(len(log_files)))
    ## Return
    return log_files

def _reformat_log_data(log_data):
    """

    """
    ## Identify Metadata
    entities, attributes, folds, splits = set(), set(), set(), set()
    for ld in log_data:
        folds.add(ld["fold"])
        splits.add(ld["split"])
        if ld["valid"]["entity"]:
            entities.update(list(ld["entity"]["scores_entity_per_label"].keys()))
        if ld["valid"]["attributes"]:
            attributes.update(list(ld["attributes"]["loss"].keys()))
    ## Ordering
    entities = ["overall"] + sorted(entities) if len(entities) > 0 else None
    attributes = sorted(attributes) if len(attributes) > 0 else None
    folds = sorted(folds)
    splits = [i for i in ["train","dev","test"] if i in splits]
    ## Build Standardized Cache
    entity_cache = {ent:[] for ent in entities} if entities is not None else None
    attributes_cache = {attr:[] for attr in attributes} if attributes is not None else None
    for ld in log_data:
        if ld["valid"]["entity"]:
            entity_cache["overall"].append({
                "fold":ld["fold"],
                "split":ld["split"],
                "epoch":ld["epoch"],
                "steps":ld["steps"],
                "label":None,
                "metric":"loss",
                "score":ld["entity"]["loss"],
                "support":None
            })
            for ent in entities:
                for score_type in ["partial","strict"]:
                    for metric in ["precision","recall","f1-score"]:
                        if ent == "overall":
                            entity_cache[ent].append({
                                "fold":ld["fold"],
                                "split":ld["split"],
                                "epoch":ld["epoch"],
                                "steps":ld["steps"],
                                "label":score_type,
                                "metric":metric,
                                "score":np.nan_to_num(ld["entity"]["scores_entity"][score_type][metric]),
                                "support":ld["entity"]["scores_entity"][score_type]["support"]
                            })
                        else:
                            entity_cache[ent].append({
                                "fold":ld["fold"],
                                "split":ld["split"],
                                "epoch":ld["epoch"],
                                "steps":ld["steps"],
                                "label":score_type,
                                "metric":metric,
                                "score":np.nan_to_num(ld["entity"]["scores_entity_per_label"][ent][score_type][metric]),
                                "support":ld["entity"]["scores_entity_per_label"][ent][score_type]["support"]
                            })
        if ld["valid"]["attributes"]:
            for attr in attributes:
                attr_loss = ld["attributes"]["loss"].get(attr, np.nan)
                attr_scores = ld["attributes"]["scores"].get(attr)
                attr_scores_per_entity = ld["attributes"]["scores_per_entity"].get(attr, {})
                attr_loss_per_entity = ld["attributes"]["loss_per_entity"].get(attr, {})
                for entity in ["Overall"] + list(attr_scores_per_entity.keys()):
                    ## Determine Appropriate Score Source
                    entity_score_dict = attr_scores if entity == "Overall" else attr_scores_per_entity.get(entity)
                    if entity_score_dict is None:
                        continue
                    ## Add General Scores
                    for lbl in ["macro avg","weighted avg"] + entity_score_dict["labels"]:
                        for metric in ["precision","recall","f1-score"]:
                            attributes_cache[attr].append({
                                "fold":ld["fold"],
                                "split":ld["split"],
                                "epoch":ld["epoch"],
                                "steps":ld["steps"],
                                "label":lbl,
                                "metric":metric,
                                "entity":entity,
                                "score":np.nan_to_num(entity_score_dict[lbl][metric]),
                                "support":entity_score_dict[lbl]["support"]
                            })
                    ## Add Rank Scores (ROC/AUC)
                    for lbl in entity_score_dict["labels"]:
                        attributes_cache[attr].append({
                            "fold":ld["fold"],
                            "split":ld["split"],
                            "epoch":ld["epoch"],
                            "steps":ld["steps"],
                            "label":lbl,
                            "metric":"roc_auc",
                            "entity":entity,
                            "score":{"fpr":entity_score_dict[lbl]["roc_fpr"], "tpr":entity_score_dict[lbl]["roc_tpr"], "thresholds":entity_score_dict[lbl]["roc_thresholds"], "auc":entity_score_dict[lbl]["roc_auc"]},
                            "support":entity_score_dict[lbl]["support"]
                        })
                    ## Add Confusion Matrix
                    attributes_cache[attr].append({
                        "fold":ld["fold"],
                        "split":ld["split"],
                        "epoch":ld["epoch"],
                        "steps":ld["steps"],
                        "label":None,
                        "metric":"confusion_matrix",
                        "entity":entity,
                        "score":entity_score_dict["confusion_matrix"],
                        "support":entity_score_dict["labels"]
                    })
                    ## Add Loss
                    attributes_cache[attr].append({
                        "fold":ld["fold"],
                        "split":ld["split"],
                        "epoch":ld["epoch"],
                        "steps":ld["steps"],
                        "label":None,
                        "metric":"loss",
                        "entity":entity,
                        "score":attr_loss if entity == "Overall" else attr_loss_per_entity[entity]["loss"],
                        "support":None if entity == "Overall" else attr_loss_per_entity[entity]["support"]
                    })
    entity_cache = {ent:pd.DataFrame(data) for ent, data in entity_cache.items()} if entity_cache is not None else None
    attributes_cache = {attr:pd.DataFrame(data) for attr, data in attributes_cache.items()} if attributes_cache is not None else None
    ## Return
    return entity_cache, attributes_cache, entities, attributes, folds, splits

def _load_configuration(log_file):
    """

    """
    ## Go Back For Checkpoints
    lf_dir = os.path.dirname(log_file)
    n_back = 0
    while not os.path.exists(f"{lf_dir}/train.cfg.json"):
        ## No Luck
        if n_back == 3:
            raise FileNotFoundError("Could not find training configuration file.")
        ## Go Back a Directory
        lf_dir = lf_dir + "/../"
        n_back += 1
    ## Format Directory
    lf_dir = os.path.abspath(lf_dir)
    ## Filename
    lf_config = f"{lf_dir}/train.cfg.json"
    ## Load
    with open(lf_config,"r") as the_file:
        lf_config = json.load(the_file)
    ## Return
    return lf_config

def load_log_data(log_files):
    """

    """
    ## Iterate Through Files
    log_data = []
    log_data_configs = {}
    for lf in log_files:
        ## Load
        lf_data = load_torch(lf)
        ## Add Fold Info
        lf_fold = int(lf.split("/fold-")[1].split("/")[0])
        for datum in lf_data:
            datum["fold"] = lf_fold
        ## Cache
        log_data.extend(lf_data)
        ## Load Configuration
        lf_config = _load_configuration(log_file=lf)
        log_data_configs[lf_fold] = lf_config
    ## Reformat and Return
    return _reformat_log_data(log_data), log_data_configs

def _get_subplots(entity_cache, attributes_cache):
    """

    """
    subplots = []
    if entity_cache is not None:
        subplots.append("named_entity_recognition")
    if attributes_cache is not None:
        subplots.extend(list(attributes_cache.keys()))
    return subplots

def mean_ci(values,
            iterations=10,
            alpha=0.05,
            random_state=42):
    """

    """
    values = np.array([float(v) for v in values])
    if np.isnan(values).all():
        return (0, 0)
    stat = np.nanmean(values)
    stat_cache = np.zeros(iterations)
    stat_seed = np.random.RandomState(random_state)
    for i in range(iterations):
        i_values = stat_seed.choice(values, len(values), replace=True)
        if np.isnan(i_values).all():
            stat_cache[i] = np.nan
        else:
            stat_cache[i] = np.nanmean(i_values)
    stat_ci = np.nanpercentile(stat_cache - stat, [alpha/2 * 100, 100 - (alpha/2 * 100)])
    stat_ci = np.abs(stat_ci)
    return tuple(stat_ci)

def plot_loss_curves(entity_cache,
                     attributes_cache,
                     folds,
                     splits,
                     eval_strategy,
                     show_average=False):
    """

    """
    ## Plot
    subplots = _get_subplots(entity_cache, attributes_cache)
    ## Initialize Plot
    fig, ax = plt.subplots(len(splits),
                           len(subplots),
                           figsize=(max(15, 4.5 * len(subplots)), 4.5 * len(splits)),
                           sharex=False,
                           sharey=False)
    ## Iterate Through Subplots
    for p, subplot in enumerate(subplots):
        for s, split in enumerate(splits):
            if subplot == "named_entity_recognition":
                ## Isolate + Aggregate
                ps_df = entity_cache["overall"].loc[(entity_cache["overall"]["split"]==split)&
                                                    (entity_cache["overall"]["metric"]=="loss")]
            else:
                ps_df = attributes_cache[subplot].loc[(attributes_cache[subplot]["split"]==split)&
                                                      (attributes_cache[subplot]["metric"]=="loss")&
                                                      (attributes_cache[subplot]["entity"]=="Overall")]
            ps_df = ps_df.loc[ps_df["steps"] != 1, :]
            ps_df_agg_ind = eval_strategy
            ## Determine Subplot
            pax = ax[s, p] if len(subplots) > 1 else ax[s]
            ## Plot
            if show_average:
                ps_df_agg = ps_df.groupby([ps_df_agg_ind]).agg({"score":[np.nanmean, np.nanstd, mean_ci]})["score"]
                ps_df_agg["lower"] = ps_df_agg["nanmean"] - ps_df_agg["mean_ci"].map(lambda i: i[0])
                ps_df_agg["upper"] = ps_df_agg["nanmean"] + ps_df_agg["mean_ci"].map(lambda i: i[1])
                pax.fill_between(ps_df_agg.index,
                                 ps_df_agg["lower"],
                                 ps_df_agg["upper"],
                                 color="black",
                                 alpha=0.2)
                pax.plot(ps_df_agg.index,
                         ps_df_agg["nanmean"].values,
                         color="black",
                         alpha=0.4,
                         marker="o",
                         label="Average")
            for f, fold in enumerate(folds):
                fold_ps_df = ps_df.loc[ps_df["fold"]==fold]
                pax.plot(fold_ps_df[ps_df_agg_ind],
                         fold_ps_df["score"],
                         color=f"C{f}",
                         alpha=0.4,
                         marker="o",
                         label=fold)
            pax.spines["right"].set_visible(False)
            pax.spines["top"].set_visible(False)
            pax.legend(loc="upper right", fontsize=6)
            if s == len(splits) - 1:
                pax.set_xlabel(ps_df_agg_ind.title(), fontweight="bold")
            else:
                pax.set_xticklabels([])
            if s == 0:
                pax.set_title(subplot.replace("_"," ").title(), fontweight="bold")
            if p == 0:
                pax.set_ylabel("{} Loss".format(split.title()), fontweight="bold")
    fig.tight_layout()
    return fig

def plot_loss_curves_by_entity(attribute,
                               attributes_cache,
                               folds,
                               splits,
                               eval_strategy="steps",
                               show_average=False):
    """

    """
    ## Create a Proxy Data Object
    attributes_cache_proxy = {}
    entities = attributes_cache[attribute]["entity"].unique()
    for entity in entities:
        ent_df = attributes_cache[attribute].loc[(attributes_cache[attribute]["entity"]==entity)&(
                                                 (attributes_cache[attribute]["metric"]=="loss"))].copy()
        ent_df["entity"] = "Overall"
        ent_name = " ".join(entity) if not isinstance(entity, str) else entity
        attributes_cache_proxy[ent_name] = ent_df
    ## Plot
    return plot_loss_curves(None, attributes_cache_proxy, folds, splits, eval_strategy, show_average=show_average)

def plot_performance_curves(metric,
                            entity_cache,
                            attributes_cache,
                            folds,
                            splits,
                            eval_strategy,
                            show_average=False):
    """

    """
    ## Validate
    if entity_cache is not None and attributes_cache is not None:
        raise ValueError("Should only specify enity_cache or attributes_cache")
    elif entity_cache is None and attributes_cache is None:
        return None
    ## Subplots
    if entity_cache is not None:
        subplots = ["overall"] + [i for i in sorted(entity_cache.keys()) if i != "overall"]
        ptype = "entity"
    elif attributes_cache is not None:
        subplots = sorted(attributes_cache.keys())
        ptype = "attributes"
    ## Initialize Plot
    fig, ax = plt.subplots(len(splits),
                           len(subplots),
                           figsize=(max(15, 4.5 * len(subplots)), 4.5 * len(splits)),
                           sharex=False,
                           sharey=False)
    ## Iterate Through Subplots
    for p, subplot in enumerate(subplots):
        for s, split in enumerate(splits):
            ## Determine Subplot
            pax = ax[s, p] if len(subplots) > 1 else ax[s]
            ## Get Data
            if ptype == "entity":
                ps_df = entity_cache[subplot].loc[(entity_cache[subplot]["split"]==split)&
                                                  (entity_cache[subplot]["metric"]==metric)]
                lines = ps_df["label"].unique()
            elif ptype == "attributes":
                ps_df = attributes_cache[subplot].loc[(attributes_cache[subplot]["split"]==split)&
                                                      (attributes_cache[subplot]["metric"]==metric)*
                                                      (attributes_cache[subplot]["entity"]=="Overall")]
                lines = sorted([i for i in ps_df["label"].unique() if i not in ["macro avg","weighted avg"]]) + ["macro avg","weighted avg"]
            ## Iterate Through Lines
            l_ps_df_agg_ind = "Checkpoint"
            for l, line in enumerate(lines):
                l_ps_df = ps_df.loc[ps_df["label"]==line]
                l_ps_df = l_ps_df.loc[l_ps_df["steps"] != 1, :]
                l_ps_df_agg_ind = eval_strategy
                ## Plot
                color = f"C{l}" if line not in ["macro avg","weighted avg"] else {"macro avg":"gray","weighted avg":"black"}[line]
                if show_average:
                    l_ps_df_agg = l_ps_df.groupby([l_ps_df_agg_ind]).agg({"score":[np.nanmean, np.nanstd, mean_ci]})["score"]
                    l_ps_df_agg["lower"] = l_ps_df_agg["nanmean"] - l_ps_df_agg["mean_ci"].map(lambda i: i[0])
                    l_ps_df_agg["upper"] = l_ps_df_agg["nanmean"] + l_ps_df_agg["mean_ci"].map(lambda i: i[1])
                    pax.fill_between(l_ps_df_agg.index,
                                     l_ps_df_agg["lower"],
                                     l_ps_df_agg["upper"],
                                     color=color,
                                     alpha=0.2)
                for f, fold in enumerate(folds):
                    fold_l_ps_df = l_ps_df.loc[l_ps_df["fold"]==fold]
                    pax.plot(fold_l_ps_df[l_ps_df_agg_ind],
                             fold_l_ps_df["score"],
                             color=color,
                             marker="o",
                             alpha=0.3,
                             label=line if f == 0 else None)
            ## Format
            pax.legend(loc="lower right", fontsize=6)
            pax.spines["right"].set_visible(False)
            pax.spines["top"].set_visible(False)
            pax.set_ylim(0, 1.05)
            if s == len(splits) - 1:
                pax.set_xlabel(l_ps_df_agg_ind.title(), fontweight="bold")
            else:
                pax.set_xticklabels([])
            if s == 0:
                pax.set_title(subplot.replace("_"," ").title(), fontweight="bold")
            if p == 0:
                pax.set_ylabel("{} {}".format(split.title(), metric.title()), fontweight="bold")
    fig.tight_layout()
    return fig

def plot_performance_curves_by_entity(attribute,
                                      metric,
                                      attributes_cache,
                                      folds,
                                      splits,
                                      eval_strategy):
    """

    """
    ## Format Attributes to Reuse Performance Curve Plotting
    proxy_attributes_cache = {}
    attribute_entities = attributes_cache[attribute]["entity"].unique()
    for entity in attribute_entities:
        ent_df = attributes_cache[attribute].loc[attributes_cache[attribute]["entity"] == entity].copy()
        ent_df["entity"] = "Overall"
        ent_df_name = "Overall" if entity == "Overall" else " ".join(entity)
        proxy_attributes_cache[ent_df_name] = ent_df
    ## Return
    return plot_performance_curves(metric, None, proxy_attributes_cache, folds, splits, eval_strategy)

def plot_performance_snapshot(criteria,
                              split_plot,
                              split_criteria,
                              entity_cache,
                              attributes_cache,
                              folds,
                              eval_strategy,
                              plot=True):
    """

    """
    ## Subplots
    subplots = _get_subplots(entity_cache, attributes_cache)
    ## Initialize Plot
    if plot:
        fig, ax = plt.subplots(2 + int(attributes_cache is not None), len(subplots), figsize=(max(15, len(subplots) * 4.5), 9), sharex=False, sharey=False)
    else:
        fig, ax = None, None
    ## Initialize Score Cache
    all_scores = []
    all_scores_checkpoints = []
    all_scores_checkpoints_scores = []
    ## Iterate Through Subplots
    for s, sp in enumerate(subplots):
        ## Identify Appropriate Measures Based on Criteria
        checkpoints_cm = []
        checkpoints_else = []
        checkpoints_bo = []
        checkpoints_roc = []
        for fold in folds:
            if sp == "named_entity_recognition":
                ## Case 1: Last
                if criteria == "last":
                    step_crit = entity_cache["overall"].loc[entity_cache["overall"]["fold"]==fold]["steps"].max()
                    for ent, ent_df in entity_cache.items():
                        ent_df_c = ent_df.loc[(ent_df["fold"]==fold)&
                                              (ent_df["split"]==split_plot)&
                                              (ent_df["steps"]==step_crit)&
                                              (ent_df["metric"]!="loss")].copy()
                        ent_df_c["entity"] = ent
                        if ent == "overall":
                            checkpoints_else.append(ent_df_c)
                        else:
                            checkpoints_bo.append(ent_df_c)
                elif criteria.startswith("best-"):
                    lbl_crit, met_crit = criteria[5:].split("-",1)
                    if lbl_crit == "weighted avg":
                        fold_df = entity_cache["overall"].loc[(entity_cache["overall"]["fold"]==fold)]
                        fold_df = fold_df.loc[(fold_df["split"]==split_criteria)&
                                              (fold_df["metric"]==met_crit)&
                                              (fold_df["label"]=="strict")]
                        step_crit = fold_df.sort_values("score",ascending=False).iloc[0]["steps"]
                        for ent, ent_df in entity_cache.items():
                            ent_df_c = ent_df.loc[(ent_df["fold"]==fold)&
                                                  (ent_df["split"]==split_plot)&
                                                  (ent_df["steps"]==step_crit)&
                                                  (ent_df["metric"]!="loss")].copy()
                            ent_df_c["entity"] = ent
                            if ent == "overall":
                                checkpoints_else.append(ent_df_c)
                            else:
                                checkpoints_bo.append(ent_df_c)
                    elif lbl_crit == "macro avg":
                        for ent, ent_df in entity_cache.items():
                            fold_df = ent_df.loc[(ent_df["fold"]==fold)&
                                                 (ent_df["split"]==split_criteria)&
                                                 (ent_df["label"]=="strict")&
                                                 (ent_df["metric"]==met_crit)]
                            step_crit = fold_df.sort_values("score",ascending=False).iloc[0]["steps"]
                            ent_df_c = ent_df.loc[(ent_df["fold"]==fold)&
                                                  (ent_df["split"]==split_plot)&
                                                  (ent_df["steps"]==step_crit)&
                                                  (ent_df["metric"]!="loss")].copy()
                            ent_df_c["entity"] = ent
                            if ent == "overall":
                                checkpoints_else.append(ent_df_c)
                            else:
                                checkpoints_bo.append(ent_df_c)
                else:
                    raise ValueError("Criteria not recognized.")
            else:
                fold_df = attributes_cache[sp].loc[(attributes_cache[sp]["fold"]==fold)&
                                                   (attributes_cache[sp]["entity"]=="Overall")]
                if criteria == "last":
                    step_crit = fold_df["steps"].max()
                elif criteria.startswith("best-"):
                    lbl_crit, met_crit = criteria[5:].split("-",1)
                    fold_df_crit = fold_df.loc[(fold_df["split"]==split_criteria)&(fold_df["label"]==lbl_crit)&(fold_df["metric"]==met_crit)]
                    step_crit = fold_df_crit.loc[fold_df_crit["score"].astype(float).idxmax(),"steps"]
                else:
                    raise ValueError("Criteria not recognized.")
                fold_df_plot = fold_df.loc[(fold_df["split"]==split_plot)&(fold_df["steps"]==step_crit)].copy()
                checkpoints_cm.append(fold_df_plot.loc[fold_df_plot["metric"]=="confusion_matrix"])
                checkpoints_else.append(fold_df_plot.loc[~fold_df_plot["metric"].isin(["loss","confusion_matrix","roc_auc"])])
                checkpoints_roc.append(fold_df_plot.loc[fold_df_plot["metric"]=="roc_auc"])
                all_scores_checkpoints.append({"attribute":sp, "fold":fold, "steps":step_crit})
                fold_df_plot["task"] = sp
                all_scores_checkpoints_scores.append(fold_df_plot)
        checkpoints_cm = pd.concat(checkpoints_cm) if len(checkpoints_cm) > 0 else None
        checkpoints_bo = pd.concat(checkpoints_bo) if len(checkpoints_bo) > 0 else None
        checkpoints_roc = pd.concat(checkpoints_roc) if len(checkpoints_roc) > 0 else None
        checkpoints_else = pd.concat(checkpoints_else)
        ## Aggregate
        checkpoints_cm_agg = sum([pd.DataFrame(row["score"], index=row["support"], columns=row["support"]) for _, row in checkpoints_cm.iterrows()]) if checkpoints_cm is not None else None
        checkpoints_bo_agg = checkpoints_bo.groupby(["entity","metric","label"]).agg({"score":[np.nanmean, np.nanstd, mean_ci, list]})["score"] if checkpoints_bo is not None else None
        checkpoints_else_agg = checkpoints_else.groupby(["metric","label"]).agg({"score":[np.nanmean, np.nanstd, mean_ci, list]})["score"]
        ## Plot Standard Measures
        if plot:
            m_ax = ax[0] if len(subplots) == 1 else ax[0,s]
            for m, measure in enumerate(["precision","recall","f1-score"]):
                measure_df = checkpoints_else_agg.loc[measure]
                if sp == "named_entity_recognition":
                    measure_df_ind = ["partial","strict"]
                else:
                    measure_df_ind = ["macro avg","weighted avg"] + [i for i in sorted(measure_df.index) if i not in ["macro avg","weighted avg"]]
                bar_width = 0.95 / len(measure_df_ind)
                for i, ind in enumerate(measure_df_ind):
                    if sp == "named_entity_recognition":
                        bcolor = ["gray","black"][i]
                    else:
                        bcolor = f"C{i-2}" if ind not in ["macro avg","weighted avg"] else ["gray","black"][i]
                    m_ax.bar(m + 0.025 + bar_width * i,
                            measure_df.loc[ind]["nanmean"],
                            width=bar_width,
                            align="edge",
                            color=bcolor,
                            label=ind.title() if m == 0 else None,
                            edgecolor="white",
                            alpha=0.4)
                    m_ax.errorbar(m + 0.025 + bar_width * i + bar_width / 2,
                                measure_df.loc[ind]["nanmean"],
                                yerr=[[c] for c in list(measure_df.loc[ind]["mean_ci"])],
                                color=bcolor,
                                alpha=0.8,
                                capsize=2)
                    m_ax.scatter([m + 0.025 + bar_width * i + bar_width / 2 for _ in measure_df.loc[ind]["list"]],
                                measure_df.loc[ind]["list"],
                                color=bcolor,
                                alpha=0.6,
                                marker="o",
                                s=5)
            m_ax.spines["right"].set_visible(False)
            m_ax.spines["top"].set_visible(False)
            m_ax.set_xticks(np.arange(3) + .5)
            m_ax.set_xticklabels(["Precision","Recall","F1"], fontsize=8)
            m_ax.legend(loc="lower left", frameon=True, fontsize=5)
            if sp == "named_entity_recognition":
                m_ax.set_title("Named Entity Recognition\nWeighted Partial F1: {:.2f} $\\pm$ {:.2f}\nWeighted Strict F1: {:.2f} $\\pm$ {:.2f}".format(
                                checkpoints_else_agg.loc["f1-score"].loc["partial","nanmean"],
                                checkpoints_else_agg.loc["f1-score"].loc["partial","nanstd"],
                                checkpoints_else_agg.loc["f1-score"].loc["strict","nanmean"],
                                checkpoints_else_agg.loc["f1-score"].loc["strict","nanstd"]
                ), fontweight="bold")
            else:
                m_ax.set_title("{}\nMacro F1: {:.2f} $\\pm$ {:.2f}\nWeighted F1: {:.2f} $\\pm$ {:.2f}".format(
                                sp.title(),
                                checkpoints_else_agg.loc["f1-score"].loc["macro avg","nanmean"],
                                checkpoints_else_agg.loc["f1-score"].loc["macro avg","nanstd"],
                                checkpoints_else_agg.loc["f1-score"].loc["weighted avg","nanmean"],
                                checkpoints_else_agg.loc["f1-score"].loc["weighted avg","nanstd"]),fontweight="bold")
            m_ax.set_ylim(0, 1.05)
            m_ax.set_xlim(0, 3)
            if s == 0:
                m_ax.set_ylabel("Score", fontweight="bold")
            ## Plot Confusion Matrix or Class Breakouts
            c_ax = ax[1] if len(subplots) == 1 else ax[1,s]
            if checkpoints_cm_agg is None:
                entities_sorted = sorted(checkpoints_bo_agg.index.levels[0])
                bar_width = 0.95 / 3
                for e, entity in enumerate(entities_sorted):
                    for m, metric in enumerate(["precision","recall","f1-score"]):
                        for st, score_type in enumerate(["partial","strict"]):
                            mscore = checkpoints_bo_agg.loc[entity,metric,score_type]
                            c_ax.bar(e + 0.025 + m * bar_width,
                                    mscore["nanmean"],
                                    align="edge",
                                    color=f"C{m}",
                                    alpha=[0.3, 0.5][st],
                                    width=bar_width,
                                    label=f"{score_type.title()} {metric.title()}" if e == 0 else None,
                                    edgecolor="white",
                                    capsize=2)
                            c_ax.errorbar(e + 0.025 + m * bar_width + bar_width / 4 * [1,3][st],
                                        mscore["nanmean"],
                                        yerr=[[c] for c in list(mscore["mean_ci"])],
                                        color=f"C{m}",
                                        capsize=2,
                                        alpha=[0.5, 0.8][st])
                            c_ax.scatter([e + 0.025 + m * bar_width + bar_width / 4 * [1,3][st] for _ in mscore["list"]],
                                        mscore["list"],
                                        color=f"C{m}",
                                        s=5,
                                        marker="o",
                                        alpha=[0.5,0.8][st])
                            c_ax.text(e + 0.025 + m * bar_width + bar_width / 4 * [1,3][st],
                                    mscore["nanmean"] * 1.05,
                                    "{:.2f}".format(mscore["nanmean"]),
                                    ha="center",
                                    rotation=90,
                                    fontsize=4)
                c_ax.set_xticks(np.arange(len(entities_sorted)) + 0.5)
                c_ax.set_xticklabels(entities_sorted, rotation=45, ha="right", fontsize=8)
                c_ax.legend(loc="lower left", fontsize=5)
                c_ax.set_ylabel("Score", fontweight="bold")
                c_ax.set_xlabel("Entity Type", fontweight="bold")
                c_ax.set_xlim(0, len(entities_sorted))
                c_ax.set_ylim(0, 1.05)
                c_ax.spines["right"].set_visible(False)
                c_ax.spines["top"].set_visible(False)
                ## Macro Scores
                macro_scores = checkpoints_bo_agg.reset_index().groupby(["metric","label"]).agg({"nanmean":[np.nanmean, np.std]})["nanmean"]
                c_ax.set_title("Macro Partial F1: {:.2f} $\\pm$ {:.2f}\nMacro Strict F1: {:.2f} $\\pm$ {:.2f}".format(
                    *macro_scores.loc[("f1-score","partial")][["nanmean","std"]].values,
                    *macro_scores.loc[("f1-score","strict")][["nanmean","std"]].values
                ), fontweight="bold")
            else:
                cm_agg_normed = checkpoints_cm_agg.apply(lambda row: row / row.sum(), axis=1)
                c_ax.imshow(cm_agg_normed,
                            cmap=plt.cm.Purples,
                            alpha=0.3,
                            interpolation="nearest",
                            aspect="auto",
                            vmin=0,
                            vmax=1)
                for r, row in enumerate(checkpoints_cm_agg.values):
                    for c, cell in enumerate(row):
                        if cell == 0 or pd.isnull(cell):
                            continue
                        c_ax.text(c, r, "{:,d}".format(int(cell)), ha="center", va="center", fontsize=8)
                c_ax.set_xticks(range(checkpoints_cm_agg.shape[1]))
                c_ax.set_yticks(range(checkpoints_cm_agg.shape[0]))
                c_lbls = [i.lstrip("B-").replace("/","/\n") for i in checkpoints_cm_agg.columns.tolist()]
                c_ax.set_xticklabels(c_lbls, rotation=45, ha="right", fontsize=8)
                c_ax.set_yticklabels(c_lbls, ha="right", fontsize=8)
                if s == 0 or (s == 1 and entity_cache is not None):
                    c_ax.set_ylabel("True Label", fontweight="bold")
                c_ax.set_xlabel("Predicted Label", fontweight="bold")
            ## Plot ROC/AUC
            if checkpoints_roc is not None:
                r_ax = ax[2] if len(subplots) == 1 else ax[2,s]
                l_seen = set()
                for l, lbl in enumerate(sorted(checkpoints_roc["label"].unique())):
                    for fold in folds:
                        l_fold_roc = checkpoints_roc.loc[(checkpoints_roc["label"]==lbl)&(checkpoints_roc["fold"]==fold)]
                        l_fpr, l_tpr, l_auc = l_fold_roc["score"].item()["fpr"], l_fold_roc["score"].item()["tpr"], l_fold_roc["score"].item()["auc"]
                        if not isinstance(l_fpr, list) or not isinstance(l_tpr, list) or np.isnan(l_auc):
                            continue
                        r_ax.plot(l_fpr, l_tpr, color=f"C{l}", alpha=0.3, label=lbl if lbl not in l_seen else None)
                        l_seen.add(lbl)
                if len(l_seen) == 0:
                    r_ax.axis("off")
                else:
                    r_ax.plot([0,1],[0,1],color="black",linestyle="--",alpha=0.3)
                    r_ax.set_xlim(0,1)
                    r_ax.set_ylim(0,1)
                    r_ax.legend(loc="lower right", fontsize=5)
                    r_ax.set_xlabel("False Positive Rate", fontweight="bold")
                    r_ax.set_ylabel("True Positive Rate", fontweight="bold")
                    r_ax.spines["right"].set_visible(False)
                    r_ax.spines["top"].set_visible(False)
        ## Cache Scores
        checkpoints_else_agg["task"] = sp
        checkpoints_else_agg["lower"] = checkpoints_else_agg["nanmean"] - checkpoints_else_agg["mean_ci"].map(lambda i: i[0])
        checkpoints_else_agg["upper"] = checkpoints_else_agg["nanmean"] + checkpoints_else_agg["mean_ci"].map(lambda i: i[1])
        checkpoints_else_agg = checkpoints_else_agg.reset_index()[["task","label","metric","nanmean","nanstd","lower","upper"]]
        if checkpoints_bo_agg is not None:
            checkpoints_bo_agg["task"] = sp
            checkpoints_bo_agg["lower"] = checkpoints_bo_agg["nanmean"] - checkpoints_bo_agg["mean_ci"].map(lambda i: i[0])
            checkpoints_bo_agg["upper"] = checkpoints_bo_agg["nanmean"] + checkpoints_bo_agg["mean_ci"].map(lambda i: i[1])
            checkpoints_bo_agg = checkpoints_bo_agg.reset_index()[["task","entity","label","metric","nanmean","nanstd","lower","upper"]]
            checkpoints_else_agg["entity"] = "Overall"
        else:
            checkpoints_else_agg["entity"] = None
        all_scores.append(checkpoints_else_agg)
        if checkpoints_bo_agg is not None:
            all_scores.append(checkpoints_bo_agg)
    ## Format Plot
    if plot:
        fig.tight_layout()
    ## Format Scores
    all_scores = pd.concat(all_scores, axis=0, sort=True)
    all_scores = all_scores.loc[:, ["task","entity","label","metric","nanmean","nanstd","lower","upper"]].rename(columns={"nanmean":"mean","nanstd":"std"})
    ## Checkpoints
    all_scores_checkpoints = pd.DataFrame(all_scores_checkpoints) if len(all_scores_checkpoints) > 0 else None
    ## Checkpoints Scores
    all_scores_checkpoints_scores = pd.concat(all_scores_checkpoints_scores).reset_index(drop=True) if len(all_scores_checkpoints_scores) > 0 else None
    return fig, all_scores, all_scores_checkpoints, all_scores_checkpoints_scores

def plot_performance_snapshot_by_entity(attribute,
                                        checkpoints,
                                        split_plot,
                                        split_criteria,
                                        attributes_cache,
                                        folds,
                                        eval_strategy,
                                        plot=True):
    """

    """
    ## Checkpoint Parsing
    attr_checkpoints = checkpoints.loc[checkpoints["attribute"]==attribute].set_index("fold")["steps"].to_dict()
    ## Format Proxy Attributes
    proxy_attributes_cache = {}
    attribute_entities = attributes_cache[attribute]["entity"].unique()
    for ent in attribute_entities:
        ent_df = attributes_cache[attribute].loc[attributes_cache[attribute]["entity"] == ent].copy()
        ent_df = pd.concat([ent_df.loc[(ent_df["fold"]==x)&(ent_df["steps"]==y)] for x, y in attr_checkpoints.items()])
        ent_df["entity"] = "Overall"
        ent_df_name = "Overall" if ent == "Overall" else " ".join(ent)
        proxy_attributes_cache[ent_df_name] = ent_df
    ## Generate Plot with Proxy Data
    return plot_performance_snapshot("last", split_plot, split_criteria, None, proxy_attributes_cache, folds, eval_strategy, plot)

def main():
    """

    """
    ## Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Directory Initialization
    print("[Initializing Output Directory]")
    if os.path.exists(args.output_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.output_dir}")
    elif os.path.exists(args.output_dir) and not args.rm_existing:
        raise FileExistsError("Must include --rm_existing flag to overwrite an existing output directory.")
    _ = os.makedirs(args.output_dir)
    ## Load Label Distributions
    print("[Loading Label Distribution]")
    label_distribution = load_label_distribution(args.results_dir)
    ## Plot Label Distribution
    if label_distribution is not None and args.plot:
        ## Overall Summary
        print("[Plotting Label Summary]")
        fig = plot_overall_label_summary(label_distribution)
        fig.savefig(f"{args.output_dir}/labels.summary.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/labels.summary.pdf")
        plt.close(fig)
        ## Entity Validity
        print("[Plotting Entity Distribution]")
        fig = plot_entity_validity_distribution(label_distribution)    
        fig.savefig(f"{args.output_dir}/labels.entities.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/labels.entities.pdf")
        plt.close(fig)
        ## Attribute Distribution
        print("[Plotting Attribute Distribution]")
        fig = plot_attribute_distribution(label_distribution)
        fig.savefig(f"{args.output_dir}/labels.attributes.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/labels.attributes.pdf")
        plt.close(fig)
    ## Identify Training Logs
    print("[Identifying Training Logs]")
    if args.log_files is None:
        log_files = find_log_files(args.results_dir,
                                   args.model,
                                   expected=args.expected_folds,
                                   allow_checkpoints=args.allow_checkpoints)
    else:
        print(f">> NOTE - Using Supplied Log Files instead of: {args.results_dir}")
        log_files = args.log_files
    lf_len = 0 if log_files is None else len(log_files)
    with open(f"{args.output_dir}/filecount.txt","w") as the_file:
        the_file.write(str(lf_len))
    if log_files is None:
        print("[No Valid Training Log Files. Exiting]")
        return None
    ## Load Log Data
    print("[Loading Training Log Data]")
    (entity_cache, attributes_cache, entities, attributes, folds, splits), configs = load_log_data(log_files)
    ## Save Config
    print("[Caching Training Run Config]")
    with open(f"{args.output_dir}/config.json","w") as the_file:
        json.dump(list(configs.values())[0], the_file, indent=1)
    ## Plot Loss Curves
    if args.plot:
        print("[Plotting Loss Curves]")
        fig = plot_loss_curves(entity_cache, attributes_cache, folds, splits, args.eval_strategy)
        fig.savefig(f"{args.output_dir}/loss.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/loss.pdf")
        plt.close(fig)
        ## Loss Curves by entity
        if attributes_cache is not None:
            for attr in attributes_cache.keys():
                if len(attributes_cache[attr]["entity"].unique()) == 2:
                    continue
                fig = plot_loss_curves_by_entity(attr, attributes_cache, folds, splits, args.eval_strategy)
                fig.savefig(f"{args.output_dir}/loss-by-entity.{attr}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/loss-by-entity.{attr}.pdf")
                plt.close(fig)
    ## Plot Performance Curves
    if args.plot:
        print("[Plotting Performance Curves]")
        for metric in ["precision","recall","f1-score"]:
            ## Plot Entity Performance
            fig = plot_performance_curves(metric, entity_cache, None, folds, splits, args.eval_strategy)
            if fig is not None:
                fig.savefig(f"{args.output_dir}/ner.{metric}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/ner.{metric}.pdf")
                plt.close(fig)
            ## Plot Attribute Performance
            fig = plot_performance_curves(metric, None, attributes_cache, folds, splits, args.eval_strategy)
            if fig is not None:
                fig.savefig(f"{args.output_dir}/attributes.{metric}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/attributes.{metric}.pdf")
                plt.close(fig)
            ## Plot Attribute Performance By Entity
            if attributes_cache is not None:
                for attr in attributes_cache.keys():
                    ## Skip if No Class Breakdowns
                    if len(attributes_cache[attr]["entity"].unique()) == 2:
                        continue
                    fig = plot_performance_curves_by_entity(attr, metric, attributes_cache, folds, splits, args.eval_strategy)
                    fig.savefig(f"{args.output_dir}/attributes-by-entity.{metric}.{attr}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/attributes-by-entity.{metric}.{attr}.pdf")
                    plt.close(fig)
    ## Plot Performance Snapshots
    print("[Plotting Performance Snapshots]")
    snapshot_indices =  ["best-macro avg-f1-score","best-weighted avg-f1-score","last"] if args.model == "model" else ["last"]
    for index in snapshot_indices:
        index_str = index.replace(" ","_")
        for split in splits:
            ## Get Snapshot
            fig, scores, checkpoints, checkpoints_with_scores = plot_performance_snapshot(criteria=index,
                                                                                          split_plot=split,
                                                                                          split_criteria="test" if args.oracle else "dev",
                                                                                          entity_cache=entity_cache,
                                                                                          attributes_cache=attributes_cache,
                                                                                          folds=folds,
                                                                                          eval_strategy=args.eval_strategy,
                                                                                          plot=args.plot)
            ## Save Plot
            if args.plot:
                fig.savefig(f"{args.output_dir}/snapshot.{split}.{index_str}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/snapshot.{split}.{index_str}.pdf")
                plt.close(fig)
            ## Cache Scores
            _ = scores.to_csv(f"{args.output_dir}/snapshot-stats.{split}.{index_str}.csv",index=False)
            ## Cache Checkpoint Scores (Raw)
            if checkpoints_with_scores is not None:
                checkpoints_with_scores = checkpoints_with_scores.dropna(subset=["score"])
                with open(f"{args.output_dir}/optimal-scores.{split}.{index_str}.json","w") as the_file:
                    the_file.write(checkpoints_with_scores.to_json(orient="index"))
            ## Visualize Snapshot Entity Breakdown
            if attributes_cache is not None:
                ## Iterate Through Attributes
                for attr in attributes_cache.keys():
                    ## Skip if No Class Breakdowns
                    if len(attributes_cache[attr]["entity"].unique()) == 2:
                        continue
                    ## Plot Snapshot Breakdown
                    fig, scores_by_entity, checkpoints_by_entity, checkpoints_with_scores_by_entity = plot_performance_snapshot_by_entity(attribute=attr,
                                                                                                            checkpoints=checkpoints,
                                                                                                            split_plot=split,
                                                                                                            split_criteria="test" if args.oracle else "dev",
                                                                                                            attributes_cache=attributes_cache,
                                                                                                            folds=folds,
                                                                                                            eval_strategy=args.eval_strategy,
                                                                                                            plot=args.plot)
                    ## Save Figure
                    if args.plot:
                        fig.savefig(f"{args.output_dir}/snapshot-by-entity.{split}.{index_str}.{attr}.png",dpi=args.dpi) if not args.pdf else fig.savefig(f"{args.output_dir}/snapshot-by-entity.{split}.{index_str}.{attr}.pdf")
                        plt.close(fig)
                    ## Save Scores
                    _ = scores_by_entity.to_csv(f"{args.output_dir}/snapshot-stats-by-entity.{split}.{index_str}.{attr}.csv",index=False)
                    ## Score Breakdown
                    if checkpoints_with_scores_by_entity is not None:
                        checkpoints_with_scores_by_entity = checkpoints_with_scores_by_entity.dropna(subset=["score"])
                        with open(f"{args.output_dir}/optimal-scores-by-entity.{split}.{index_str}.{attr}.json","w") as the_file:
                            the_file.write(checkpoints_with_scores_by_entity.to_json(orient="index"))
                    
    ## Done
    print("[Script Complete]")

#########################
### Execution
#########################

if __name__ == "__main__":
    _ = main()