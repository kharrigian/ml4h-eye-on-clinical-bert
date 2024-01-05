
######################
### Imports
######################

## Standard Library
import os
from collections import Counter

## External Library
import numpy as np
import pandas as pd
import torch
import joblib

######################
### Classes
######################

class LabelEncoder(object):

    """
    
    """

    def __init__(self,
                 primary_key,
                 secondary_key=None,
                 many_to_one=False,
                 encode_label_position=False):
        """
        
        """
        ## Attributes
        self._primary_key = primary_key
        self._secondary_key = secondary_key
        self._many_to_one = many_to_one
        self._encode_label_position = encode_label_position
        ## Working Variables
        _ = self._initialize()
    
    def __repr__(self):
        """
        
        """
        return f"LabelEncoder(primary_key={self._primary_key}, secondary_key={self._secondary_key}, many_to_one={self._many_to_one})"

    def _initialize(self):
        """
        
        """
        ## Tasks
        self._task2id = {}
        self._id2task = []
        ## Classes
        self._class2id = {}
        self._id2class = {}
    
    def _fmt_label(self,
                   name,
                   is_new):
        """
        
        """
        if is_new:
            return f"B-{name}"
        else:
            return f"I-{name}"
    
    def _ignore_value(self,
                      value):
        """
        
        """
        if value is None:
            return True
        if value == "N/A":
            return True
        return False
    
    def _initialize_validity(self):
        """

        """
        ## Early Exit
        if self._secondary_key is None or not self._secondary_key.startswith("validity_"):
            return None
        ## Task Update
        task_ent = self._secondary_key.split("validity_")[1].replace("_"," ")
        task = (task_ent, )
        self._task2id[task] = len(self._task2id)
        self._id2task.append(task)
        ## Classes
        self._class2id[task] = {"O":0,"B-valid":1,"I-valid":2,"B-invalid":3,"I-invalid":4}
        self._id2class[task] = ["O","B-valid","I-valid","B-invalid","I-invalid"]

    def _update(self,
                tokens_and_labels,
                **kwargs):
        """
        
        """
        ## Iterate Over Tokens
        for _, lbl in tokens_and_labels:
            ## Iterate over Labels for Token
            for (ldict, _, _) in lbl:
                ## Indicator Values
                primary_value = ldict.get(self._primary_key)
                secondary_value = ldict.get(self._secondary_key) if self._secondary_key is not None else None
                ## Check Relevance
                if self._ignore_value(primary_value) or self._secondary_key is not None and self._ignore_value(secondary_value):
                    continue
                ## Formatted Indicator Value
                if self._secondary_key is not None:
                    value_init = self._fmt_label(secondary_value, True)
                    value_post = self._fmt_label(secondary_value, False)
                else:
                    value_init = self._fmt_label(primary_value, True)
                    value_post = self._fmt_label(primary_value, False)
                ## Task Update
                if self._many_to_one and self._secondary_key is not None:
                    task = (primary_value, secondary_value)
                elif self._many_to_one and self._secondary_key is None:
                    task = (primary_value, )
                elif not self._many_to_one and self._secondary_key is not None:
                    task = (primary_value, )
                elif not self._many_to_one and self._secondary_key is None:
                    task = ("overall", )
                ## Initialize with Null Label
                if task not in self._task2id:
                    self._task2id[task] = len(self._task2id)
                    self._id2task.append(task)
                if task not in self._class2id:
                    self._class2id[task] = {"O":0}
                    self._id2class[task] = ["O"]
                ## Class Update
                if value_init not in self._class2id[task]:
                    self._class2id[task][value_init] = len(self._class2id[task])
                    self._id2class[task].append(value_init)
                if value_post not in self._class2id[task]:
                    self._class2id[task][value_post] = len(self._class2id[task])
                    self._id2class[task].append(value_post)

    def _align_label_space(self):
        """
        
        """
        unique_labels = set()
        for task, task_labels in self._class2id.items():
            unique_labels.update(task_labels.keys())
        unique_labels = list(unique_labels)
        for task in list(self._class2id.keys()):
            self._class2id[task] = {l:i for i, l in enumerate(unique_labels)}
            self._id2class[task] = unique_labels

    def _sort_identifiers(self):
        """
        
        """
        ## Sort
        id2task_new = sorted(self._task2id.keys())
        task2id_new = {x:i for i, x in enumerate(id2task_new)}
        class2id_new = {}
        id2class_new = {}
        for task in id2task_new:
            id2class_new[task] = ["O"] + sorted(list(filter(lambda i: i != "O", self._class2id[task].keys())), key=lambda x: (x.replace("O-","").replace("B-","").replace("I-",""), 0 if x.startswith("B-") else 1 if x.startswith("I-") else 2))
            class2id_new[task] = {x:i for i, x in enumerate(id2class_new[task])}
        ## Update
        self._id2task = id2task_new
        self._task2id = task2id_new
        self._id2class = id2class_new
        self._class2id = class2id_new
    
    def _filter_identifiers(self):
        """

        """
        class2id_filt = {}
        id2class_filt = {}
        for task, task_lbl_dict in self._class2id.items():
            class2id_filt[task] = {}
            id2class_filt[task] = []
            for lbl, lbl_int in task_lbl_dict.items():
                if lbl == "O" or lbl.startswith("I-"):
                    continue
                class2id_filt[task][lbl_int] = len(class2id_filt[task])
                id2class_filt[task].append(lbl)
        self._class2id_filt = class2id_filt
        self._id2class_filt = id2class_filt
            
    def fit(self,
            data,
            sort=True,
            align_label_space=False,
            **kwargs):
        """
        
        """
        ## Validity Initialization (If Applicable)
        _ = self._initialize_validity()
        ## Iterate Through Examples
        for datum in data:
            _ = self._update(datum)
        ## Align
        if align_label_space:
            _ = self._align_label_space()
        ## Sort
        if sort:
            _ = self._sort_identifiers()
        ## Consolidate
        _ = self._filter_identifiers()
        return self

    def _labels_are_one_to_one(self,
                               labels):
        """
        
        """
        for task_labels in labels:
            if (task_labels.sum(axis=1) > 1).any():
                return False
        return True

    def _transform(self,
                   tokens_and_labels,
                   **kwargs):
        """
        
        """
        ## Initialize One-hot Encodings
        labels = [np.zeros((len(tokens_and_labels), len(self._class2id[task])), dtype=int) for task in self._id2task]
        ## Label Position Encodings
        if self._encode_label_position:
            labels_token_task = np.zeros((len(tokens_and_labels), len(self._id2task)*2), dtype=int)
        else:
            labels_token_task = np.zeros((len(tokens_and_labels), len(self._id2task)), dtype=int)
        ## Fill in Encoding
        for i, (_, lbl) in enumerate(tokens_and_labels):
            for (ldict, lnew, lvalid) in lbl:
                ## Information
                primary_value = ldict.get(self._primary_key)
                secondary_value = ldict.get(self._secondary_key) if self._secondary_key is not None else None
                ## Check Relevance
                if self._ignore_value(primary_value) or self._secondary_key is not None and self._ignore_value(secondary_value):
                    continue
                ## Formatted Value
                value_fmt = self._fmt_label(secondary_value, lnew) if self._secondary_key is not None else self._fmt_label(primary_value, lnew)
                ## Task
                if self._many_to_one and self._secondary_key is not None:
                    task = (primary_value, secondary_value)
                elif self._many_to_one and self._secondary_key is None:
                    task = (primary_value, )
                elif not self._many_to_one and self._secondary_key is not None:
                    task = (primary_value, )
                elif not self._many_to_one and self._secondary_key is None:
                    task = ("overall", )
                task_id = self._task2id[task]
                ## Encode
                if value_fmt.endswith("<PLACEHOLDER>"):
                    if lnew:
                        value_encoded = 1
                    else:
                        value_encoded = 2
                else:
                    value_encoded = self._class2id[task][value_fmt]
                ## Store Prior Info Regarding Token Relationship with Task
                if self._encode_label_position:
                    labels_token_task[i, task_id + len(self._id2task) * int(not lnew)] = 1
                else:
                    labels_token_task[i, task_id] = 1
                ## Store Valid Labels for Modeling
                if lvalid or (self._secondary_key is not None and self._secondary_key.startswith("validity_")):
                    labels[task_id][i, value_encoded] = 1
        ## Add Null
        for task_id, task_labels in enumerate(labels):
            task_labels[:,0] += np.where(task_labels.max(axis=1)==0,
                                         np.ones(task_labels.shape[0], dtype=int),
                                         np.zeros(task_labels.shape[0], dtype=int))
        ## Verify Labels
        if not self._labels_are_one_to_one(labels):
            raise ValueError("Found an instance in which labels do not have a one-to-one token mapping.")
        ## Argmax
        labels = [i.argmax(axis=1) for i in labels]
        ## Return
        return labels, labels_token_task
    
    def transform(self,
                  data,
                  **kwargs):
        """
        
        """
        ## Transform
        encoded = list(map(self._transform, data))
        ## Separate
        encoded_labels = [e[0] for e in encoded]
        encoded_token_task = [e[1] for e in encoded]
        ## Return
        return encoded_labels, encoded_token_task
    
    def fit_transform(self,
                      data,
                      **kwargs):
        """
        
        """
        _ = self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
    
    def _inverse_transform(self,
                           arrays,
                           **kwargs):
        """
        
        """
        values = []
        for task_id, task_array in enumerate(arrays):
            task = self._id2task[task_id]
            values_ = []
            for a in task_array:
                values_.append(self._id2class[task][a])
            values.append(values_)
        return values
    
    def inverse_transform(self,
                          data,
                          **kwargs):
        """
        
        """
        inverse_encoded = list(map(self._inverse_transform, data))
        return inverse_encoded
    
    def get_tasks(self):
        """
        
        """
        return self._id2task
    
    def get_classes(self,
                    task=None):
        """
        
        """
        if task is None:
            return self._id2class
        return self._id2class[task]
    
    def _extract_array_spans(self,
                             array):
        """
        
        """
        ## Base Case (All Zeros)
        if sum(array) == 0:
            return []
        ## Look for Spans
        spans = []
        cur_label = None
        cur_span_start = None
        for i, a in enumerate(array):
            if a == 0:
                if cur_span_start is not None:
                    spans.append((cur_label, (cur_span_start, i)))
                    cur_span_start = None
                    cur_label = None
                continue
            if a % 2 == 1: ## Start of New Span
                if cur_span_start is not None:
                    spans.append((cur_label, (cur_span_start, i)))
                cur_span_start = i
                cur_label = a
            elif a % 2 == 0: ## Continuation of Span
                if cur_label is None or a - cur_label != 1: ## Invalid Span (Ignore)
                    continue
        if cur_span_start is not None:
            spans.append((cur_label, (cur_span_start, i + 1)))
        return spans

    def _extract_spans(self,
                       arrays):
        """
        
        """
        array_spans = list(map(self._extract_array_spans, arrays))
        return array_spans
    
    def extract_spans(self,
                      data,
                      **kwargs):
        """
        
        """
        spans = list(map(self._extract_spans, data))
        return spans
    
    def save(self,
             filepath,
             **kwargs):
        """

        """
        _ = joblib.dump(self, filepath, **kwargs)
    
    @staticmethod
    def load(filepath):
        """

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Filepath doesn't exist: '{filepath}'")
        return joblib.load(filepath)

##############################
### Functions
##############################

def display_task_summary(entity_encoder,
                         attribute_encoders):
    """
    
    """
    ## Track Number of Tasks
    n_tasks = 1
    ## NER
    if entity_encoder is not None:
        print("Task {}) named_entity_recognition (Entities): {}".format(n_tasks, entity_encoder.get_tasks()))
        print("                                  (Classes): {}".format(["O","B-*","I-*"]))
        n_tasks += 1
    ## Sometimes Present (Attribute Classifiction)
    if attribute_encoders is not None:
        for a, (attribute, encoder) in enumerate(attribute_encoders.items()):
            ents = encoder.get_tasks()
            classes = [i.replace("B-","") for i in encoder.get_classes(ents[0]) if not i.startswith("I-") and i != "O"]
            prefix = f"Task {n_tasks}) {attribute}"
            prefix_n = " " * (len(prefix) + 1)
            print(f"{prefix} (Entities): {ents}")
            print(f"{prefix_n} (Classes): {classes}")
            n_tasks += 1

def _display_data_summary(task_name, counts):
    """
    
    """
    print(f"***** Label Distribution for Task: {task_name} *****")
    print("(N = {:,d})".format(sum(counts.values())))
    for x, y in counts.most_common():
        print(f"{x} :: {y}")

def display_data_summary(entity_encoder,
                         entity_labels,
                         attribute_encoders,
                         attribute_labels,
                         show_concept_breakdown=False):
    """
    
    """
    ## NER
    if entity_encoder is not None:
        entity_counts = Counter()
        for lbl in entity_labels:
            for target, target_lbl in zip(entity_encoder.get_tasks(), lbl):
                entity_counts[target] += (target_lbl == 1).sum()
        _ = _display_data_summary("Named Entity Recognition", entity_counts)
    ## Attributes
    if attribute_encoders is not None:
        for attribute, labels in attribute_labels.items():
            attribute_counts = Counter()
            attribute_counts_concept = Counter()
            for instance in labels:
                for concept, concept_lbl in zip(attribute_encoders[attribute].get_tasks(), instance):
                    concept_classes = attribute_encoders[attribute].get_classes(concept)
                    concept_classes = [(i,x) for i, x in enumerate(concept_classes) if x.startswith("B-")]
                    for ind, ind_lbl in concept_classes:
                        n_add = (concept_lbl == ind).sum()
                        if n_add == 0:
                            continue
                        attribute_counts[ind_lbl] += n_add
                        attribute_counts_concept[(concept, ind_lbl)] += n_add
            _ = _display_data_summary(f"{attribute} (Overall)", attribute_counts)
            if show_concept_breakdown:
                _ = _display_data_summary(f"{attribute} (Concept-Specific)", attribute_counts_concept)

def apply_attribute_limit(attribute_encoders,
                          attribute_labels,
                          document_groups,
                          spans_per_document=None,
                          spans_per_document_label_stratify=True,
                          documents_per_group=None,
                          spans_per_group=None,
                          spans_per_group_label_stratify=True,
                          random_state=42):
    """

    """
    ## Base Case: No Change
    if spans_per_document is None and documents_per_group is None and spans_per_group is None:
        return attribute_labels
    ## Extract All Labels
    flattened_spans = []
    for attr, attr_encoder in attribute_encoders.items():
        attr_labels = attribute_labels[attr]
        attr_labels_spans = attr_encoder.extract_spans(attr_labels)
        for i, lbls in enumerate(attr_labels_spans):
            for t, (task, task_lbls) in enumerate(zip(attr_encoder.get_tasks(), lbls)):
                task_classes = attr_encoder.get_classes(task)
                for cls, (span_start, span_end) in task_lbls:
                    flattened_spans.append({
                        "attribute":attr,
                        "document_index":i,
                        "document_group":document_groups[i],
                        "task":t,
                        "task_str":task,
                        "label":cls,
                        "label_str":task_classes[cls],
                        "span_start":span_start,
                        "span_end":span_end
                    })
    flattened_spans = pd.DataFrame(flattened_spans)
    flattened_spans["token_bounds"] = flattened_spans.apply(lambda row: (row["task"], row["label"], row["span_start"],row["span_end"]), axis=1)
    ## Consolidate
    if spans_per_document_label_stratify:
        flattened_spans_agg = pd.pivot_table(flattened_spans, index=["document_group","document_index","label"], columns="attribute", values="token_bounds", aggfunc=list)
    else:
        flattened_spans_agg = pd.pivot_table(flattened_spans, index=["document_group","document_index"], columns="attribute", values="token_bounds", aggfunc=list)
    flattened_docs_agg = flattened_spans.groupby(["document_group","attribute"])["document_index"].unique().map(list).unstack()
    ## Option 1: Restrict Number of Spans Per Attribute Per Document
    if spans_per_document is not None:
        span_seed = np.random.RandomState(random_state)
        flattened_spans_agg_s = flattened_spans_agg.stack().sort_index().to_frame("spans")
        flattened_spans_agg_s["spans_sampled"] = flattened_spans_agg_s["spans"].map(lambda s: set(span_seed.choice(len(s), min(spans_per_document, len(s)), replace=False)))
        flattened_spans_agg_s["spans_removed"] = flattened_spans_agg_s.apply(lambda row: [r for i, r in enumerate(row["spans"]) if i not in row["spans_sampled"]], axis=1)
    ## Option 2: Restrict Number of Documents Per Group
    if documents_per_group is not None:
        document_seed = np.random.RandomState(random_state)
        flattened_docs_agg_s = flattened_docs_agg.stack().sort_index().to_frame("documents")
        flattened_docs_agg_s["documents_sampled"] = flattened_docs_agg_s["documents"].map(lambda s: set(document_seed.choice(len(s), min(documents_per_group, len(s)), replace=False)))
        flattened_docs_agg_s["documents_removed"] = flattened_docs_agg_s.apply(lambda row: [r for i, r in enumerate(row["documents"]) if i not in row["documents_sampled"]], axis=1)
    ## Apply Span Restrictions
    if spans_per_document is not None:
        spans_to_remove = flattened_spans_agg_s.reset_index().set_index(["attribute","document_index"])["spans_removed"].sort_index()
        spans_to_remove = spans_to_remove.loc[spans_to_remove.map(len)>0]
        for attr in spans_to_remove.index.levels[0]:
            if attr not in spans_to_remove.index:
                continue
            for attr_doc in spans_to_remove.loc[attr].index:
                for (task_id, label_id, span_start, span_end) in spans_to_remove.loc[(attr,attr_doc)]:
                    attribute_labels[attr][attr_doc][task_id][span_start:span_end] = 0
    ## Apply Document Restricts
    if documents_per_group is not None:
        docs_to_remove = flattened_docs_agg_s.reset_index().set_index(["attribute"])["documents_removed"].sort_index()
        docs_to_remove = docs_to_remove.loc[docs_to_remove.map(len) > 0]
        for attr in set(docs_to_remove.index):
            for attr_row in docs_to_remove.loc[attr].values:
                for doc_index in attr_row:
                    attribute_labels[attr][doc_index] = [np.zeros_like(i) for i in attribute_labels[attr][doc_index]]
    ## If not limiting spans by group, done
    if spans_per_group is None:
        return attribute_labels
    ## Take Stake
    flattened_spans_updated = []
    for i, (attr, attr_labels) in enumerate(attribute_labels.items()):
        for a, lbl in enumerate(attr_labels):
            a_group = document_groups[a]
            attr_labels_spans = attribute_encoders[attr]._extract_spans(lbl)
            for e, ent_lbls in enumerate(attr_labels_spans):
                for lbl, (tok_start, tok_end) in ent_lbls:
                    flattened_spans_updated.append({
                        "attribute":attr,
                        "document_group":a_group,
                        "document_index":a,
                        "entity":e,
                        "label":lbl,
                        "span_start":tok_start,
                        "span_end":tok_end
                    })
    flattened_spans_updated = pd.DataFrame(flattened_spans_updated)
    ## Get Appropriate Limit Grouping
    if spans_per_group_label_stratify:
        grouping = flattened_spans_updated.groupby(["attribute","label","document_group"]).groups.items()
    else:
        grouping = flattened_spans_updated.groupby(["attribute","document_group"]).groups.items()
    ## Idenitfy Spans to Remove
    spans_to_remove = {attr:[] for attr in attribute_labels.keys()}
    span_seed = np.random.RandomState(random_state)
    for group, group_inds in grouping:
        if len(group_inds) <= spans_per_group:
            continue
        group_vals = flattened_spans_updated.loc[group_inds][["document_index","entity","span_start","span_end"]].apply(tuple,axis=1).values
        group_remove = span_seed.choice(group_vals, len(group_vals) - spans_per_group, replace=False)
        spans_to_remove[group[0]].extend(group_remove)
    ## Apply Removal
    for attr, attr_remove in spans_to_remove.items():
        for doc_ind, ent_ind, tok_start, tok_end in attr_remove:
            attribute_labels[attr][doc_ind][ent_ind][tok_start:tok_end] = 0
    ## Return
    return attribute_labels