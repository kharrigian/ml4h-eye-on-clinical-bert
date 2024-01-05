
#######################
### Imports
#######################

## External Libraries
import torch
import numpy as np

#######################
### Functions
#######################

def _get_tag_span_mapping(tags):
    """

    """
    mapping = {}
    for i, (tok, tok_lbls) in enumerate(tags):
        for lbl, _, _ in tok_lbls:
            lbl_t = lbl["label"]
            lbl_bounds = (lbl["start"], lbl["end"])
            if lbl_t not in mapping:
                mapping[lbl_t] = {}
            if lbl_bounds not in mapping[lbl_t]:
                mapping[lbl_t][lbl_bounds] = []
            mapping[lbl_t][lbl_bounds].append(i)
    for lbl, lbl_map in mapping.items():
        for chars, chars_i in lbl_map.items():
            mapping[lbl][chars] = (chars_i[0], chars_i[-1]+1)
    mapping_r = {}
    for lbl, lbl_map in mapping.items():
        mapping_r[lbl] = {}
        for char_bounds, tok_bounds in lbl_map.items():
            mapping_r[lbl][tok_bounds] = char_bounds
    return mapping_r

#######################
### Classes
#######################

class MLMTokenDataset(torch.utils.data.Dataset):

    """
    MLMTokenDataset
    """

    def __init__(self,
                 data,
                 **kwargs):
        """
        Args:
            data (list of list): List of examples. Each example is a truncated token list.
        """
        self._data = data
    
    def __len__(self):
        """
        
        """
        return len(self._data)
    
    def __getitem__(self, idx):
        """
        
        """
        ## Get Item
        tokens = self._data[idx]
        ## Prepare
        item = {
                "input_ids":torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens,
                "attention_mask":torch.tensor([1] * len(tokens)),
                "labels":torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
                }
        return item

class EntityAttributeDataset(torch.utils.data.Dataset):

    """
    
    """

    def __init__(self,
                 document_ids,
                 token_ids,
                 labels_entity=None,
                 labels_attributes=None,
                 task_tokens_entity=None,
                 encoder_entity=None,
                 encoder_attributes=None,
                 tagged_sequences=None,
                 text=None):
        """
        
        """
        self._document_ids = document_ids
        self._token_ids = token_ids
        self._labels_entity = labels_entity
        self._labels_attributes = labels_attributes
        self._task_tokens_entity = task_tokens_entity
        self._encoder_entity = encoder_entity
        self._encoder_attributes = encoder_attributes
        self._tagged_sequences = tagged_sequences
        self._text = text
    
    def __len__(self):
        """
        
        """
        return len(self._token_ids)

    def _get_entity_types(self):
        """
        
        """
        return self._encoder_entity.get_tasks() if self._encoder_entity is not None else None 

    def _get_attribute_names(self):
        """
        
        """
        return list(self._encoder_attributes.keys()) if self._encoder_attributes is not None else None
    
    def _get_attribute_types(self):
        """
        
        """
        return [a.get_tasks() for a in self._encoder_attributes.values()] if self._encoder_attributes is not None else None
    
    def __getitem__(self, idx):
        """
        
        """
        ## Get Relevant Data
        tokens = self._token_ids[idx]
        entities = self._labels_entity[idx] if self._labels_entity is not None else None
        entities_spans = self._encoder_entity._extract_spans(entities) if entities is not None else None
        task_token_entities = self._task_tokens_entity[idx] if entities is not None and self._task_tokens_entity is not None else None
        attributes = {a:attr[idx] for a, attr in self._labels_attributes.items()} if self._labels_attributes is not None else None
        attributes_spans = {a:ae._extract_spans(attributes[a]) for a, ae in self._encoder_attributes.items()} if attributes is not None else None
        seq = self._tagged_sequences[idx] if self._tagged_sequences is not None else None
        text = self._text[idx] if self._text is not None else None
        ## Align Attribute Spans with Primary Label
        if attributes_spans is not None:
            for attr, attr_spans in attributes_spans.items():
                attr_remap = list(self._encoder_attributes[attr]._class2id_filt.values())[0]
                attr_spans_fmt = []
                for span_group in attr_spans:
                    span_group_fmt = []
                    for (lbl, lbl_bounds) in span_group:
                        span_group_fmt.append((attr_remap[lbl], lbl_bounds))
                    attr_spans_fmt.append(span_group_fmt)
                attributes_spans[attr] = attr_spans_fmt
        ## Align Character Sequence Spans
        if attributes_spans is not None and seq is not None:
            tag_span_map = _get_tag_span_mapping(seq)
            attribute_character_spans = {}
            for attr, attr_spans in attributes_spans.items():
                attr_tasks = self._encoder_attributes[attr].get_tasks()
                if any(len(i) > 1 for i in attr_tasks):
                    raise ValueError("DEBUGGING ISSUE")
                attr_tasks = [i[0] for i in attr_tasks]
                attr_spans_fmt = []
                for span_group, task in zip(attr_spans, attr_tasks):
                    span_group_fmt = []
                    for (lbl, lbl_bounds) in span_group:
                        span_group_fmt.append((lbl, tag_span_map[task][lbl_bounds]))
                    attr_spans_fmt.append(span_group_fmt)
                attribute_character_spans[attr] = attr_spans_fmt
        ## Format
        datum = {
            "input_ids":torch.tensor(tokens),
            "attention_mask":torch.ones(len(tokens), dtype=int),
            "token_type_ids":torch.zeros(len(tokens), dtype=int),
            "entity_tokens":torch.tensor(task_token_entities) if task_token_entities is not None else None,
            "entity_labels":torch.tensor(np.vstack(entities)) if entities is not None else None,
            "attribute_labels":[torch.tensor(np.vstack(attr)) for a, attr in attributes.items()] if attributes is not None else None,
            "entity_spans":entities_spans,
            "attribute_spans":attributes_spans,
            "attribute_character_spans":attribute_character_spans if attributes is not None and seq is not None else None,
            "text":text,
            "index":idx,
        }
        return datum

#######################
### Functions
#######################

def mlm_collate_batch(batch,
                      mlm_probability=0.15,
                      mlm_max_per_sequence=None,
                      special_mask_id=103):
    """
    
    """
    ## Separate Components
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    ## Get Max Length in Batch
    padmax = min(512, max(list(map(lambda x: x.shape[0], input_ids))))
    ## Cache
    input_ids_new = []
    attention_mask_new = []
    labels_new = []
    ## Sample New Data with Masks
    for t, (inputs, attns, lbls) in enumerate(zip(input_ids, attention_mask, labels)):
        ## Sample The Mask
        mask = np.random.binomial(1, mlm_probability, (inputs.shape[0], ))
        ## Maximum Masking Criteria
        if mlm_max_per_sequence is not None and mask.sum() > mlm_max_per_sequence:
            ind_remove = np.random.choice(mask.nonzero()[0], mask.sum() - mlm_max_per_sequence, replace=False)
            mask[ind_remove] = 0
        ## Filter Indices
        mask_nn = np.where(mask==1)[0]
        mask_nn_op = np.where(mask==0)[0]
        ## Update The Inputs (Mask the Targets)
        inputs[mask_nn] = special_mask_id
        ## Update The Labels (Mask the Non-targets)
        lbls[mask_nn_op] = -100
        ## Batch-specific Padding
        inputs = torch.nn.functional.pad(inputs, (0, padmax-inputs.shape[0]), "constant", 0)
        lbls = torch.nn.functional.pad(lbls, (0, padmax-lbls.shape[0]), "constant", 0)
        attns = torch.nn.functional.pad(attns, (0, padmax-attns.shape[0]), "constant", 0)
        ## Cache
        input_ids_new.append(inputs)
        attention_mask_new.append(attns)
        labels_new.append(lbls)
    ## Reformat Batch
    items = {
        "input_ids":torch.stack(input_ids_new),
        "attention_mask":torch.stack(attention_mask_new),
        "labels":torch.stack(labels_new)
    }
    return items

def _get_pooler(start_ind,
                end_ind,
                size,
                use_first_index=False):
    """
    
    """
    pooler = torch.zeros((1, size)).float()
    if use_first_index:
        pooler[0, start_ind] = 1
    else:
        pooler[0, start_ind:end_ind] = 1 / (end_ind-start_ind)
    return pooler

def _get_flattened_spans(batch,
                         span_key,
                         attribute_key,
                         n_subspans,
                         size,
                         use_first_index=False):
    """
    
    """
    ## Extract
    metadata = []
    poolers = []
    for i in range(n_subspans):
        for j, b in enumerate(batch):
            cycle = b[span_key][i] if attribute_key is None else b[span_key][attribute_key][i]
            for i_lbl, (i_start, i_end) in cycle:
                metadata.append([i, j, b["index"], i_start, i_end, i_lbl])
                poolers.append(_get_pooler(start_ind=i_start, end_ind=i_end, size=size, use_first_index=use_first_index))
    ## Merge
    if len(metadata) == 0:
        metadata = None
        poolers = None
    else:
        metadata = torch.tensor(metadata)
        poolers = torch.stack(poolers)
    ## Format
    spans = {"metadata":metadata, "poolers":poolers}
    return spans

def collate_entity_attribute(batch,
                             use_first_index=False):
    """
    
    """
    ## Batch Metadata
    max_input_len = max([b["input_ids"].size(0) for b in batch])
    n_entities = len(batch[0]["entity_spans"]) if batch[0]["entity_spans"] is not None else None
    n_attributes = {a:len(ap) for a, ap in batch[0]["attribute_spans"].items()} if batch[0]["attribute_spans"] is not None else None
    ## Pad the Tensor Elements
    input_ids = torch.stack([torch.nn.functional.pad(b["input_ids"], (0, max_input_len-b["input_ids"].shape[0]), "constant", 0) for b in batch])
    token_type_ids = torch.stack([torch.nn.functional.pad(b["token_type_ids"], (0, max_input_len-b["token_type_ids"].shape[0]), "constant", 0) for b in batch])
    attention_mask = torch.stack([torch.nn.functional.pad(b["attention_mask"], (0, max_input_len-b["attention_mask"].shape[0]), "constant", 0) for b in batch])
    entity_labels = torch.stack([torch.nn.functional.pad(b["entity_labels"], (0, max_input_len-b["entity_labels"].shape[1]), "constant", 0) for b in batch]) if n_entities is not None else None
    entity_tokens = torch.stack([torch.nn.functional.pad(b["entity_tokens"], (0, 0, 0, max_input_len - b["entity_tokens"].shape[0]), "constant", 0) for b in batch]) if n_entities is not None and batch[0]["entity_tokens"] is not None else None
    attribute_labels = [torch.stack([torch.nn.functional.pad(b["attribute_labels"][i], (0, max_input_len-b["attribute_labels"][i].shape[1]), "constant", 0) for b in batch]) for i in range(len(n_attributes))] if n_attributes is not None else None
    ## Combine the Non-Tensor Elements
    entity_spans = None
    attribute_spans = None
    attribute_character_spans = None
    if n_entities is not None:
        entity_spans = _get_flattened_spans(batch=batch,
                                            span_key="entity_spans",
                                            attribute_key=None,
                                            n_subspans=n_entities,
                                            size=max_input_len,
                                            use_first_index=use_first_index)
    if n_attributes is not None:
        attribute_spans = {attribute:_get_flattened_spans(batch=batch,
                                                          span_key="attribute_spans",
                                                          attribute_key=attribute,
                                                          n_subspans=n,
                                                          size=max_input_len,
                                                          use_first_index=use_first_index) for attribute, n in n_attributes.items()}
        attribute_character_spans = {attribute:_get_flattened_spans(batch=batch,
                                                                    span_key="attribute_character_spans",
                                                                    attribute_key=attribute,
                                                                    n_subspans=n,
                                                                    size=max_input_len,
                                                                    use_first_index=use_first_index) for attribute, n in n_attributes.items()}
    ## Other Metadata
    index = torch.tensor([b["index"] for b in batch])
    text = [b["text"] for b in batch]
    ## Format Output
    collated = {
        "input_ids":input_ids,
        "token_type_ids":token_type_ids,
        "attention_mask":attention_mask,
        "entity_labels":entity_labels,
        "entity_tokens":entity_tokens,
        "attribute_labels":attribute_labels,
        "entity_spans":entity_spans,
        "attribute_spans":attribute_spans,
        "attribute_character_spans":attribute_character_spans,
        "index":index,
        "text":text
    }
    return collated

def move_batch_to_device(batch, device):
    """

    """
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["token_type_ids"] = batch["token_type_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    batch["entity_labels"] = batch["entity_labels"].to(device) if batch["entity_labels"] is not None else None
    batch["entity_tokens"] = batch["entity_tokens"].to(device) if batch["entity_tokens"] is not None else None
    batch["attribute_labels"] = [b.to(device) for b in batch["attribute_labels"]] if isinstance(batch["attribute_labels"], list) else None
    if isinstance(batch["entity_spans"], dict):
        if isinstance(batch["entity_spans"]["metadata"], torch.Tensor):
            batch["entity_spans"]["metadata"] = batch["entity_spans"]["metadata"].to(device)
        if isinstance(batch["entity_spans"]["poolers"], torch.Tensor):
            batch["entity_spans"]["poolers"] = batch["entity_spans"]["poolers"].to(device)
    if isinstance(batch["attribute_spans"], dict):
        for category in batch["attribute_spans"].keys():
            nadded = 0
            if isinstance(batch["attribute_spans"][category]["metadata"], torch.Tensor):
                batch["attribute_spans"][category]["metadata"] = batch["attribute_spans"][category]["metadata"].to(device)
                nadded += 1
            if isinstance(batch["attribute_spans"][category]["poolers"], torch.Tensor):
                batch["attribute_spans"][category]["poolers"] = batch["attribute_spans"][category]["poolers"].to(device)
                nadded += 1
            if nadded == 1:
                raise ValueError("Only one of 'metadata' or 'poolers' was not null.")
    if isinstance(batch["attribute_character_spans"], dict):
        for category in batch["attribute_character_spans"].keys():
            nadded = 0
            if isinstance(batch["attribute_character_spans"][category]["metadata"], torch.Tensor):
                batch["attribute_character_spans"][category]["metadata"] = batch["attribute_character_spans"][category]["metadata"].to(device)
                nadded += 1
            if isinstance(batch["attribute_character_spans"][category]["poolers"], torch.Tensor):
                batch["attribute_character_spans"][category]["poolers"] = batch["attribute_character_spans"][category]["poolers"].to(device)
                nadded += 1
            if nadded == 1:
                raise ValueError("Only one of 'metadata' or 'poolers' was not null.")
    return batch