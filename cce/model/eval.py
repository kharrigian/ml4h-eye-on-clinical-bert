
## Standard Library
import sys

## External Libraries
import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn import metrics

## Local
from ..util.helpers import flatten, chunks
from .datasets import collate_entity_attribute, move_batch_to_device, _get_tag_span_mapping
from .loss import compute_entity_loss, compute_attribute_loss

######################
### Functions
######################

def _precision(x):
    """

    """
    ## Get True Positives
    num = x["true_positive"]
    ## Get Predicted Positives
    dn = (x["true_positive"] + x["false_positive"])
    ## Get Expected Positives
    possible = x["true_positive"] + x["false_negative"]
    ## Case 1: No Instances and None Predicted (Reward with 1)
    if possible == 0 and dn == 0:
        return 1
    ## Case 2: No Instances And a False Positive
    elif possible == 0 and dn > 0:
        return 0
    ## Case 3: Instances But No Predicted Positives (Precision Not Defined)
    elif possible > 0 and dn == 0:
        return np.nan
    ## Case 4: Instances and Some Predicted Positives (Must be either True or False)
    else:
        return num / dn

def _recall(x):
    """

    """
    ## Get True Positives
    num = x["true_positive"]
    ## Get Expected (True Positives + False Negatives)
    dn = x["true_positive"] + x["false_negative"]
    ## Case 1: No Instances Exist
    if dn == 0:
        return 1
    ## Case 2: Instances Exist and Were Either Found or Not
    else:
        return num / dn

def _f1_score(x):
    """

    """
    ## Compute Precision and Recall
    p = _precision(x)
    r = _recall(x)
    ## Check for Nulls
    if np.isnan(p) or np.isnan(r):
        return np.nan
    ## Compute F1
    num = 2 * p * r
    dn = p + r
    ## Case 1: Terrible Precision and Terrible Recall
    if p == 0 and r == 0:
        return 0
    ## Case 2: Harmonic Mean of Precision and Recall
    else:
        return num / dn

def evaluate_ner_entity(entity_true,
                        entity_pred,
                        entity_classes):
    """

    """
    ## Counting
    counts = {"".join(ent):{"FP":0, "TP_Exact":0, "TP_Partial":0, "FN":0} for ent in entity_classes}
    for et, ep in zip(entity_true, entity_pred):
        for ent in entity_classes:
            ent_fmt = "".join(ent)
            etrue = np.array([j[1] for j in et if j[0] == ent])
            epred = np.array([j[1] for j in ep if j[0] == ent])
            egroups = _align_entities(etrue, epred)
            for eg in egroups:
                if len(eg["pred"]) == 0:
                    counts[ent_fmt]["FN"] += len(eg["true"])
                else:
                    for eg_true in eg["true"]:
                        if eg_true in eg["pred"]:
                            counts[ent_fmt]["TP_Exact"] += 1
                        else:
                            counts[ent_fmt]["TP_Partial"] += 1
                if len(eg["true"]) == 0:
                    counts[ent_fmt]["FP"] += len(eg["pred"])
    ## Overall Counts
    counts_overall = pd.DataFrame(counts).T.sum(axis=0).to_dict()
    ## Get Scores
    results = {
        "partial":{
            "true_positive":counts_overall["TP_Partial"] + counts_overall["TP_Exact"],
            "false_positive":counts_overall["FP"],
            "false_negative":counts_overall["FN"],
            "support":counts_overall["FN"] + counts_overall["TP_Partial"] + counts_overall["TP_Exact"]
        },
        "strict":{
            "true_positive":counts_overall["TP_Exact"],
            "false_positive":counts_overall["FP"],
            "false_negative":counts_overall["FN"] + counts_overall["TP_Partial"],
            "support":counts_overall["FN"] + counts_overall["TP_Partial"] + counts_overall["TP_Exact"]
        }
    }
    results_per_tag = {}
    for entity, entity_counts in counts.items():
        results_per_tag[entity] = {
        "partial":{
            "true_positive":entity_counts["TP_Partial"] + entity_counts["TP_Exact"],
            "false_positive":entity_counts["FP"],
            "false_negative":entity_counts["FN"],
            "support":entity_counts["FN"] + entity_counts["TP_Partial"] + entity_counts["TP_Exact"]
        },
        "strict":{
            "true_positive":entity_counts["TP_Exact"],
            "false_positive":entity_counts["FP"],
            "false_negative":entity_counts["FN"] + entity_counts["TP_Partial"],
            "support":entity_counts["FN"] + entity_counts["TP_Partial"] + entity_counts["TP_Exact"]
        }
        }
    ## Compute Scores
    for score_type in ["partial","strict"]:
        for dic in [results[score_type]] + [i[score_type] for i in results_per_tag.values()]:
            dic["precision"] = _precision(dic)
            dic["recall"] = _recall(dic)
            dic["f1-score"] = _f1_score(dic)
    return results, results_per_tag

def classification_report(y_true,
                          y_pred,
                          labels):
    """

    """
    ## Helpers
    tp = lambda l, cm: cm[l,l]
    fp = lambda l, cm: cm[:,l].sum() - cm[l, l]
    fn = lambda l, cm: cm[l,:].sum() - cm[l, l]
    breakdown = lambda l, cm: [tp(l, cm), fp(l, cm), fn(l, cm)]
    ## Compute Confusion Matrix
    conf_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    ## Get Breakdown for Each Label
    conf_matrix_breakdown = np.array([breakdown(l, conf_matrix) for l in range(len(labels))])
    ## Initialize Report Dict
    report = {}
    for l, lbl in enumerate(labels):
        lbl_dict = dict(zip(["true_positive","false_positive","false_negative"],conf_matrix_breakdown[l]))
        report[lbl] = {
            "precision":_precision(lbl_dict),
            "recall":_recall(lbl_dict),
            "f1-score":_f1_score(lbl_dict),
            "support":lbl_dict["true_positive"] + lbl_dict["false_negative"],
        }
    n_total = sum([i["support"] for i in report.values()])
    report["weighted avg"] = {
        "precision":sum([np.nan_to_num(report[lbl]["precision"]) * report[lbl]["support"] / n_total for lbl in labels]),
        "recall":sum([np.nan_to_num(report[lbl]["recall"]) * report[lbl]["support"] / n_total for lbl in labels]),
        "f1-score":sum([np.nan_to_num(report[lbl]["f1-score"]) * report[lbl]["support"] / n_total for lbl in labels]),
        "support":n_total
    }
    report["macro avg"] = {
        "precision":np.mean([np.nan_to_num(report[lbl]["precision"]) for lbl in labels]),
        "recall":np.mean([np.nan_to_num(report[lbl]["recall"]) for lbl in labels]),
        "f1-score":np.mean([np.nan_to_num(report[lbl]["f1-score"]) for lbl in labels]),
        "support":n_total
    }
    return report

def _evaluate_batch(batch_inds,
                    dataset,
                    model,
                    entity_weights,
                    attribute_weights,
                    device,
                    use_first_index=False):
    """

    """
    ## Entity Classes
    entity_classes = model.get_entities() if dataset._encoder_entity is not None else None
    ## Collate
    batch = collate_entity_attribute([dataset[idx] for idx in batch_inds],
                                     use_first_index=use_first_index)
    ## Device Update
    batch = move_batch_to_device(batch, device)
    ## Forward Pass
    batch_entity_logits, batch_attribute_logits = model(batch)
    ## Decoding
    if dataset._encoder_entity is not None:
        batch_entity_tags = model.decode_entity(batch,
                                                batch_entity_logits)
    ## Update Entity Loss
    batch_entity_loss = None
    if dataset._encoder_entity is not None:
        batch_entity_loss = compute_entity_loss(model=model,
                                                entity_weights=entity_weights,
                                                entity_logits=batch_entity_logits,
                                                inputs=batch,
                                                reduce=True)
        batch_entity_loss = batch_entity_loss.data.item()
    ## Update Attribute Loss
    batch_attribute_loss = {}
    if dataset._encoder_attributes is not None:
        batch_attribute_loss = compute_attribute_loss(model=model,
                                                      attribute_weights=attribute_weights,
                                                      attribute_logits=batch_attribute_logits,
                                                      inputs=batch,
                                                      reduce=False)
        batch_attribute_loss = {x:y.data for x, y in zip(model.get_attributes(), batch_attribute_loss) if isinstance(y, torch.Tensor)}
        # batch_attribute_loss = {x:y.data.item() for x, y in zip(model.get_attributes(), batch_attribute_loss) if isinstance(y, torch.Tensor)}
    ## Move Outputs Back to CPU for Score Computations
    batch = move_batch_to_device(batch, "cpu")
    if dataset._encoder_entity is not None:
        batch_entity_tags = batch_entity_tags.to("cpu")
    if dataset._encoder_attributes is not None:
        batch_attribute_logits = list(map(lambda bal: bal.to("cpu") if isinstance(bal, torch.Tensor) else None, batch_attribute_logits))
    ## Store Masked Entity Tags/Labels (If Applicable)
    batch_entity_predictions = []
    batch_ent_true, batch_ent_pred = [], []
    if dataset._encoder_entity is not None:
        ## Flatten Entity Tags/Labels
        token_attention_mask = (batch["attention_mask"]==1).ravel()
        token_entity_labels = [batch["entity_labels"][:,i,:].ravel() for i in range(batch["entity_labels"].size(1))] if dataset._encoder_entity is not None else None
        token_entity_predictions = [batch_entity_tags[:,i,:].ravel() for i in range(batch_entity_tags.size(1))] if dataset._encoder_entity is not None else None
        ## Store with Attention Mask
        for i, (pred, lbl) in enumerate(zip(token_entity_predictions, token_entity_labels)):
            batch_entity_predictions.append(torch.stack([pred[token_attention_mask], lbl[token_attention_mask]]).T)
        ## Entity-Level
        batch_ent_true = [flatten([[(entity_classes[e], i[1]) for i in ent_tags] for e, ent_tags in enumerate(dataset._encoder_entity._extract_spans(tags[:,batch["attention_mask"][tt] == 1]))]) for tt, tags in enumerate(batch["entity_labels"])]
        batch_ent_pred = [flatten([[(entity_classes[e], i[1]) for i in ent_tags] for e, ent_tags in enumerate(dataset._encoder_entity._extract_spans(tags[:,batch["attention_mask"][tt] == 1]))]) for tt, tags in enumerate(batch_entity_tags)]
    ## Store Attribute Predictions
    batch_attribute_predictions = {}
    if dataset._encoder_attributes is not None:
        for attr, attr_logits in zip(model.get_attributes(), batch_attribute_logits):
            if attr_logits is None:
                continue
            at_pred = attr_logits.argmax(axis=1)
            at_prob = torch.nn.Softmax(dim=1)(attr_logits)
            at_true = batch["attribute_spans"][attr]["metadata"][:,-1]
            at_meta = batch["attribute_spans"][attr]["metadata"][:,[0,2,3,4]] # [concept_id, index, start token, end token]
            batch_attribute_predictions[attr] = torch.hstack([torch.stack([at_pred, at_true]).T, at_meta, at_prob])
    ## Return
    return batch_entity_loss, batch_attribute_loss, batch_entity_predictions, batch_ent_true, batch_ent_pred, batch_attribute_predictions

def _align(outer, inner):
    """

    """
    ## Overlap
    aligned = []
    ## Size
    n_outer = outer.shape[0]
    n_inner = inner.shape[0]
    ## Outer Loop
    i_outer = 0
    while i_outer < n_outer:
        i_start, i_end = outer[i_outer]
        j_inner = 0
        overlap_found = False
        while j_inner < n_inner:
            j_start, j_end = inner[j_inner]
            ij_overlap = i_start <= j_end and i_end >= j_start
            if ij_overlap:
                aligned.append([i_outer, j_inner])
                overlap_found = True
            elif not ij_overlap and overlap_found:
                break
            j_inner += 1
        if not overlap_found:
            aligned.append([i_outer, None])
        i_outer += 1
    return aligned
                
def _align_entities(ent_true,
                    ent_pred):
    """

    """ 
    ## Build Relationship Graph
    g = nx.Graph()
    for t, p in _align(ent_true, ent_pred):
        if p is not None:
            g.add_edge(f"true_{t}", f"pred_{p}")
        else:
            g.add_node(f"true_{t}")
    for p, t in _align(ent_pred, ent_true):
        if t is not None:
            g.add_edge(f"pred_{p}",f"true_{t}")
        else:
            g.add_node(f"pred_{p}")
    ## Identify Entity Groups
    entity_groups = list(nx.connected_components(g))
    ## Parse Entity Groups
    entity_group_bounds = []
    for g, ent_group in enumerate(entity_groups):
        group_ents = {"true":[],"pred":[]}
        for node in ent_group:
            node_id = int(node.split("_")[-1])
            node_bounds = (ent_true if node.startswith("true_") else ent_pred)[node_id]
            group_ents[node.split("_")[0]].append([*node_bounds])
        entity_group_bounds.append(group_ents)
    ## Return
    return entity_group_bounds

def _convert_tokens(token_ids,
                    ind2vocab=None):
    """

    """
    ## Base Case
    if ind2vocab is None:
        return token_ids
    ## Merge and Combine
    tokens = " " .join([ind2vocab[ind] for ind in token_ids])
    return tokens

def _get_text_window(text,
                     span_start,
                     span_end,
                     window_size=10):
    """

    """
    ## Indicate Entity
    text = text[:span_start] + "**{}**".format(text[span_start:span_end]) + text[span_end:]
    ## Move To White Space
    while span_start > 0 and text[span_start-1] not in ["\n"," "]:
        span_start -= 1
    while span_end < len(text) - 1 and text[span_end] not in ["\n"," "]:
        span_end += 1
    ## Get Regions
    left_text = text[:span_start]
    cent_text = text[span_start:span_end]
    right_text = text[span_end:]
    ## Get Windows
    left_text = " ".join(left_text.strip().split()[-window_size:])
    right_text = " ".join(right_text.strip().split()[:window_size])
    ## Combine
    text_window = f"{left_text} {cent_text} {right_text}"
    ## Return
    return text_window

def _format_entity_predictions_error_type(row):
    """

    """
    ## Extract from Row
    ent_type = row["entity"]
    ent_true = set([(i[0], i[1]) for i in row["entities_true"]])
    ent_pred = set([(i[0], i[1]) for i in row["entities_pred"]])
    ## Initialize Counts
    counts = {"FP":0,"FN":0,"TP_Exact":0,"TP_Partial":0}
    if len(ent_pred) == 0:
        counts["FN"] += len(ent_true)
    if len(ent_true) == 0:
        counts["FP"] += len(ent_pred)
    for et in ent_true:
        if et in ent_pred:
            counts["TP_Exact"] += 1
        else:
            counts["TP_Partial"] += 1
    ## Return
    return counts

def format_entity_predictions(entity_predictions,
                              dataset,
                              vocab2ind=None):
    """

    """
    ## Format
    ent_true = np.array(entity_predictions[0])
    ent_pred = np.array(entity_predictions[1])
    ## Reverse Vocab Index
    ind2vocab = None
    if vocab2ind is not None:
        ind2vocab = sorted(vocab2ind, key=lambda x: vocab2ind.get(x))
    ## Initialize Cache
    formatted = []
    ## Iterate Through Documents
    for d, (doc_id, doc_tokens, doc_tags) in enumerate(zip(dataset._document_ids, dataset._token_ids, dataset._tagged_sequences)):
        ## Isolate Label Set
        d_true = ent_true[ent_true[:,0]==d]
        d_pred = ent_pred[ent_pred[:,0]==d]
        ## Get Character to Token Mapping for Ground Truth
        char_map_true = _get_tag_span_mapping(doc_tags) 
        ## Iterate Through Tasks
        for t, task in enumerate(dataset._encoder_entity.get_tasks()):
            ## Bounds
            task_ent_true = d_true[d_true[:,-1]==t][:,1:3]
            task_ent_pred = d_pred[d_pred[:,-1]==t][:,1:3]
            ## Align Bounds
            task_ent_groups = _align_entities(task_ent_true, task_ent_pred)            
            ## Cache Bound Groups
            for group in task_ent_groups:
                formatted.append({
                    "document_id":doc_id,
                    "entity":task,
                    "entities_true":[(i, j, *char_map_true[task[0]][(i,j)], _convert_tokens(doc_tokens[i:j], ind2vocab)) for i, j in group["true"]],
                    "entities_pred":[(i, j, None, None, _convert_tokens(doc_tokens[i:j], ind2vocab)) for i, j in group["pred"]]
                })
    ## DataFrame Format
    formatted = pd.DataFrame(formatted)
    ## Error Types
    formatted["error_types"] = formatted.apply(_format_entity_predictions_error_type, axis=1)
    for etype in ["FP","FN","TP_Exact","TP_Partial"]:
        formatted[etype] = formatted["error_types"].map(lambda i: i.get(etype))
    formatted = formatted.drop("error_types", axis=1)
    ## Return
    return formatted

def format_attribute_predictions(attribute_predictions,
                                 dataset,
                                 vocab2ind=None,
                                 context_size=10):
    """

    """
    ## Reverse Vocab Index
    ind2vocab = None
    if vocab2ind is not None:
        ind2vocab = sorted(vocab2ind, key=lambda x: vocab2ind.get(x))
    ## Initialize Cache
    formatted = []
    ## Iterate Through Attributes
    for attribute, predictions in attribute_predictions.items():
        ## Iterate Through Prediction Rows
        for row in predictions:
            ## Ignore Predicted Probabilities
            row = [int(i) for i in row[:6]]
            ## Identify Tokens and Surrounding Context
            row_tokens = dataset._token_ids[row[3]][row[-2]:row[-1]]
            row_context_tokens = dataset._token_ids[row[3]][max(0, row[-2]-context_size):row[-1]+context_size]
            ## Identify Associated Concept
            row_concept = dataset._encoder_attributes[attribute]._id2task[row[2]][0]
            ## Get Tag Metadata
            row_tag = [i for i, j, k in dataset._tagged_sequences[row[3]][row[-2]][1] if i["label"]==row_concept][0]
            ## Text
            row_text_window = _get_text_window(dataset._text[row[3]], row_tag["start"], row_tag["end"], window_size=max(context_size * 3, 30))
            ## Cache
            formatted.append({
                "document_id":dataset._document_ids[row[3]],
                "attribute":attribute,
                "concept":row_concept,
                "char_start":row_tag["start"],
                "char_end":row_tag["end"],
                "in_header":row_tag.get("in_header",False),
                "in_autolabel_postprocess":row_tag.get("in_autolabel_postprocess",False),
                "ind_start":row[-2],
                "ind_end":row[-1],
                "entity_span":_convert_tokens(row_tokens, ind2vocab),
                "entity_span_context":_convert_tokens(row_context_tokens, ind2vocab),
                "entity_span_context_raw":row_text_window,
                "lbl_true":dataset._encoder_attributes[attribute]._classes[row[1]],
                "lbl_pred":dataset._encoder_attributes[attribute]._classes[row[0]]
            })
    ## DataFrame
    formatted = pd.DataFrame(formatted)
    ## Correct Label Attribute
    formatted["correct"] = formatted["lbl_true"] == formatted["lbl_pred"]
    return formatted

def evaluate(model,
             dataset,
             batch_size=16,
             entity_weights=None,
             attribute_weights=None,
             desc=None,
             device="cpu",
             use_first_index=False,
             return_predictions=False):
    """
    
    """
    ## Meta
    entity_classes = model.get_entities() if dataset._encoder_entity is not None else None
    n_entity_classes = len(entity_classes) if dataset._encoder_entity is not None else None
    ## Prediction Cache
    entity_loss = 0 if dataset._encoder_entity is not None else None
    entity_predictions = [[] for _ in range(n_entity_classes)] if dataset._encoder_entity is not None else None
    attribute_loss = {at:0 for at in model.get_attributes()} if dataset._encoder_attributes is not None else None
    attribute_loss_by_entity = {at:torch.zeros((len(dataset._encoder_attributes[at].get_tasks()), 2)) for at in model.get_attributes()} if dataset._encoder_attributes is not None else None
    attribute_predictions = {at:[] for at in model.get_attributes()} if dataset._encoder_attributes is not None else None
    entity_spans = [[], []]
    n_batches_ent = 0 if dataset._encoder_entity is not None else None
    n_instances_att = {x:0 for x in attribute_loss.keys()} if dataset._encoder_attributes is not None else None
    ## Batches
    eval_batches = list(chunks(list(range(len(dataset))), batch_size))
    ## Iterate Through Batches
    model.eval()
    with torch.no_grad():
        for bb, batch_inds in tqdm(enumerate(eval_batches),
                                   total=len(eval_batches),
                                   desc=f"[Running Evaluation ({desc}) w/ Batch Size {batch_size}]" if desc is not None else f"[Running Evaluation w/ Batch Size {batch_size}]",
                                   file=sys.stdout):
            ## Try With Batch
            try:
                batch_entity_loss, batch_attribute_loss, batch_entity_predictions, batch_ent_true, batch_ent_pred, batch_attribute_predictions = _evaluate_batch(batch_inds=batch_inds,
                                                                                                                                                                 dataset=dataset,
                                                                                                                                                                 model=model,
                                                                                                                                                                 entity_weights=None,
                                                                                                                                                                 attribute_weights=None,
                                                                                                                                                                 device=device,
                                                                                                                                                                 use_first_index=use_first_index)
                if dataset._encoder_entity is not None:
                    entity_loss += batch_entity_loss
                    n_batches_ent += 1
                    for e, ent_preds in enumerate(batch_entity_predictions):
                        entity_predictions[e].append(ent_preds)
                    entity_spans[0].extend(batch_ent_true)
                    entity_spans[1].extend(batch_ent_pred)
                if dataset._encoder_attributes is not None:
                    for attr, attr_loss in batch_attribute_loss.items():
                        attr_loss_m = attr_loss[:,0] * attr_loss[:,1]
                        attribute_loss[attr] += torch.nansum(attr_loss_m).item()                      
                        n_instances_att[attr] += attr_loss[:,1].sum().item()
                        attribute_loss_by_entity[attr] += torch.vstack([attr_loss_m, attr_loss[:,1]]).T
                    for attr, attr_preds in batch_attribute_predictions.items():
                        attribute_predictions[attr].append(attr_preds)
            except RuntimeError as e:
                raise e
            except Exception as e:
                raise e
    ## Average The Loss Over Batches
    if dataset._encoder_entity is not None:
        entity_loss = entity_loss / n_batches_ent
    if dataset._encoder_attributes is not None:
        for x, y in attribute_loss.items():
            attribute_loss[x] = (y / n_instances_att[x]) if n_instances_att[x] > 0 else np.nan        
        for x, y in attribute_loss_by_entity.items():
            y_avg = torch.where(y[:,1] > 0, y[:,0] / y[:,1], torch.zeros_like(y[:,1]) * torch.nan).detach().numpy()
            attribute_loss_by_entity[x] = {} 
            for e, ent in enumerate(dataset._encoder_attributes[x].get_tasks()):
                attribute_loss_by_entity[x][ent] = {"loss":y_avg[e], "support":int(y[e,1].item())}
    ## Format
    stack_preds = lambda x: torch.vstack(x) if len(x) > 0 else None
    entity_predictions = [stack_preds(preds) for preds in entity_predictions] if dataset._encoder_entity is not None else None
    attribute_predictions = {at:stack_preds(preds) for at, preds in attribute_predictions.items()} if dataset._encoder_attributes is not None else None
    ## Scoring (Entities)
    entity_scores, entity_level_scores, entity_level_scores_per_lbl = None, None, None
    if dataset._encoder_entity is not None:
        ## Entity-Level
        entity_level_scores, entity_level_scores_per_lbl = evaluate_ner_entity(entity_true=entity_spans[0],
                                                                               entity_pred=entity_spans[1],
                                                                               entity_classes=entity_classes)
        ## Re-formatted Scores
        entity_classes_r = {c:i for i, c in enumerate(entity_classes)}
        entity_spans_formatted = [[],[]]
        for i, espans in enumerate(entity_spans):
            for j, jspan in enumerate(espans):
                for lbl, (ind_start, ind_end) in jspan:
                    entity_spans_formatted[i].append([j, ind_start, ind_end, entity_classes_r[lbl]])
        ## Token-Level
        entity_scores = {}
        for entity, entity_preds in zip(model.get_entities(), entity_predictions):
            ## Classes
            entity_lbls_named = dataset._encoder_entity.get_classes(entity)
            entity_possible_labels = list(range(len(entity_lbls_named)))
            ## Report
            entity_scores_report = classification_report(y_true=entity_preds[:,1],
                                                         y_pred=entity_preds[:,0],
                                                         labels=entity_possible_labels)
            for c, cls_name in enumerate(entity_lbls_named):
                if c not in entity_scores_report:
                    continue
                entity_scores_report[cls_name] = entity_scores_report.pop(c)
            entity_scores_report["accuracy"] = metrics.accuracy_score(y_true=entity_preds[:,1],
                                                                      y_pred=entity_preds[:,0])
            entity_scores_report["confusion_matrix"] = metrics.confusion_matrix(y_true=entity_preds[:,1],
                                                                                y_pred=entity_preds[:,0],
                                                                                labels=entity_possible_labels)
            entity_scores_report["labels"] = entity_lbls_named
            ## Cache
            entity_scores[entity] = entity_scores_report
    ## Scoring (Attributes)
    attribute_scores = None
    attribute_scores_per_entity = None
    if attribute_loss is not None:
        ## Initialize Score Cache
        attribute_scores = {}
        attribute_scores_per_entity = {}
        ## Iterate Through Tasks
        for at, at_preds in attribute_predictions.items():
            ## Get Label Info
            at_lbls_named = model._encoder_attributes[at]._classes
            at_enti_types = model._encoder_attributes[at].get_tasks()
            ## Initialize Per Entity Dict
            attribute_scores_per_entity[at] = {}
            ## Iterate Through Entities Associated With Task (And Overall)
            for e, ent_type in enumerate(at_enti_types + [None]):
                ## Isolate Subset
                if ent_type is None or at_preds is None:
                    ent_type_at_preds = at_preds
                else:
                    ent_type_mask = at_preds[:,2] == e
                    ent_type_at_preds = at_preds[ent_type_mask]
                ## Handle Output Depending on Presence of Labels
                if ent_type_at_preds is None or ent_type_at_preds.shape[0] == 0:
                    ## Standard Classification Metrics
                    ent_score_dict = {key:{"precision":np.nan, "recall":np.nan, "f1-score":np.nan, "support":0} for key in at_lbls_named + ["macro avg","weighted avg"]}
                    ent_score_dict["accuracy"] = np.nan
                    ## Confusion Matrix
                    ent_score_dict["confusion_matrix"] = np.zeros((len(at_lbls_named),len(at_lbls_named)), dtype=int)
                    ## ROC/AUC
                    for lbl in at_lbls_named:
                        ent_score_dict[lbl].update({"roc_fpr":[], "roc_tpr":[], "roc_thresholds":[], "roc_auc":np.nan})
                    ## Label Information
                    ent_score_dict["labels"] = at_lbls_named
                else:
                    ## Named Labels
                    y_true_named = [at_lbls_named[i] for i in ent_type_at_preds[:,1].int()]
                    y_pred_named = [at_lbls_named[i] for i in ent_type_at_preds[:,0].int()]
                    ## Standard Classification Metric
                    ent_score_dict = classification_report(y_true=y_true_named,
                                                                y_pred=y_pred_named,
                                                                labels=at_lbls_named)
                    ent_score_dict["accuracy"] = metrics.accuracy_score(y_true=y_true_named,
                                                                            y_pred=y_pred_named)
                    ## Confusion Matirx
                    ent_score_dict["confusion_matrix"] = metrics.confusion_matrix(y_true=y_true_named,
                                                                                        y_pred=y_pred_named,
                                                                                        labels=at_lbls_named)
                    ## ROC/AUC
                    y_true = ent_type_at_preds[:,1].int()
                    y_pred_prob = ent_type_at_preds[:,-len(at_lbls_named):]
                    for l, lnamed in enumerate(at_lbls_named):
                        ## Format
                        y_true_l = (y_true == l).int()
                        y_pred_prob_l = y_pred_prob[:,l]
                        ## Score
                        if y_true_l.sum() == 0 or y_true_l.sum() == y_true_l.shape[0]:                    
                            tpr, fpr, thres, auc = [], [], [], np.nan
                        else:
                            fpr, tpr, thres = metrics.roc_curve(y_true_l.detach().numpy(), y_pred_prob_l.detach().numpy(), pos_label=1)
                            auc = metrics.auc(fpr, tpr)
                            fpr, tpr, thres = list(fpr), list(tpr), list(thres)
                        ## Cache
                        ent_score_dict[lnamed].update({
                            "roc_fpr":fpr,
                            "roc_tpr":tpr,
                            "roc_thresholds":thres,
                            "roc_auc":auc
                        })
                    ## Label Information
                    ent_score_dict["labels"] = at_lbls_named
                ## Cache
                if ent_type is None:
                    attribute_scores[at] = ent_score_dict
                else:
                    attribute_scores_per_entity[at][ent_type] = ent_score_dict
    ## Format For Output
    output = {
        "entity":{
            "scores_token":entity_scores,
            "scores_entity":entity_level_scores,
            "scores_entity_per_label":entity_level_scores_per_lbl,
            "loss":entity_loss if dataset._encoder_entity is not None else None,
            "predictions":entity_spans_formatted if return_predictions and dataset._encoder_entity is not None else None
         },
        "attributes":{
            "scores":attribute_scores,
            "scores_per_entity":attribute_scores_per_entity,
            "loss":attribute_loss if dataset._encoder_attributes is not None else None,
            "loss_per_entity":attribute_loss_by_entity if dataset._encoder_attributes is not None else None,
            "predictions":{x:[list(i) for i in y.detach().numpy()] for x, y in attribute_predictions.items()} if return_predictions and dataset._encoder_attributes is not None else None
         },
         "valid":{
            "entity":dataset._encoder_entity is not None,
            "attributes":dataset._encoder_attributes is not None,
            }
    }
    ## Return to Model Training
    model.train()
    return output

def display_evaluation(scores):
    """
    
    """
    ## Entity
    if scores["valid"]["entity"]:
        ## Entity-level
        print("~~~~~~~~~~~~~~~~~~ Entity Recognition (Entity-Level) ~~~~~~~~~~~~~~~~~~")
        print("Overall:\n", pd.DataFrame(scores["entity"]["scores_entity"]).applymap(lambda i: "{:.3f}".format(i)).to_string())
        for lbl, lbl_scores in scores["entity"]["scores_entity_per_label"].items():
            print(f"{lbl}:\n", pd.DataFrame(lbl_scores).applymap(lambda i: "{:.3f}".format(i)).to_string())
        ## Entity - Token-level
        print("~~~~~~~~~~~~~~~~~~ Entity Recognition (Token-Level) ~~~~~~~~~~~~~~~~~~")
        print("Average Loss: {:.4f}".format(scores["entity"]["loss"]))
        for score_type in ["macro avg", "weighted avg"]:
            scores_ = pd.DataFrame({x:y[score_type] for x, y in scores["entity"]["scores_token"].items()}).T
            print(f"{score_type}:\n", scores_.to_string())
    ## Attributes
    if scores["valid"]["attributes"]:
        for at in scores["attributes"]["loss"].keys():
            at_scores = {x:y for x, y in scores["attributes"]["scores"][at].items() if x not in ["confusion_matrix","labels"]}
            for x, y in at_scores.items():
                if isinstance(y, dict):
                    at_scores[x] = {i:j for i, j in y.items() if not i.startswith("roc_") or i == "roc_auc"}
                else:
                    at_scores[x] = y
            at_loss = scores["attributes"]["loss"][at]
            print(f"~~~~~~~~~~~~~~~~~~ Attribute Classification ['{at}'] ~~~~~~~~~~~~~~~~~~")
            print("Average Loss: {:.4f}".format(at_loss))
            print("Scores:\n", pd.DataFrame(at_scores).T.to_string())
