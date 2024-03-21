
#######################
### Imports
#######################

## Standard Library
import os
from collections import Counter

## External Libraries
import torch
import numpy as np
from torchcrf import CRF
from transformers import AutoModel

#######################
### Globals
#######################

## BERT Dimensions
MAX_BERT_LENGTH = 512
BERT_HIDDEN_DIM = 768

## Repository Root
_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")

#######################
### Classes
#######################

class MLP(torch.nn.Module):
    
    """

    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim=None,
                 random_state=None):
        """

        """
        ## Inheritence
        _ = super(MLP, self).__init__()
        ## Random Initialization
        self._random_state = random_state
        if self._random_state is not None:
            _ = torch.manual_seed(self._random_state)
            np.random.seed(self._random_state)
        ## Layers
        if hidden_dim is None:
            self.classifier = torch.nn.Linear(in_features=in_dim,
                                              out_features=out_dim,
                                              bias=True)
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim,
                                out_features=hidden_dim,
                                bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=hidden_dim,
                                out_features=out_dim,
                                bias=True)
            )
        
    def forward(self, X):
        """

        """
        return self.classifier(X)


class NERTaggerModel(torch.nn.Module):

    """
    
    """

    def __init__(self,
                 encoder_entity=None,
                 encoder_attributes=None,
                 token_encoder="bert-base-cased",
                 token_vocab=None,
                 freeze_encoder=False,
                 use_crf=False,
                 use_lstm=False,
                 use_entity_token_bias=False,
                 use_attribute_concept_bias=False,
                 max_sequence_length=MAX_BERT_LENGTH,
                 sequence_overlap=None,
                 sequence_overlap_type="mean",
                 dropout=0.1,
                 lstm_hidden_size=768,
                 lstm_num_layers=1,
                 lstm_bidirectional=True,
                 entity_hidden_size=None,
                 entity_token_bias_type="uniform",
                 attributes_hidden_size=None,
                 random_state=None):
        """
        
        """
        ## Inheritence
        _ = super(NERTaggerModel, self).__init__()
        ## Random Initialization
        self._random_state = random_state
        if self._random_state is not None:
            _ = torch.manual_seed(self._random_state)
            np.random.seed(self._random_state)
        ## Store the Encoders
        self._encoder_entity = encoder_entity
        self._encoder_attributes = encoder_attributes
        ## Classes
        self._entities = self._encoder_entity.get_tasks() if self._encoder_entity is not None else None
        self._attributes = None
        if self._encoder_attributes is not None:
            self._attributes = list(self._encoder_attributes.keys())
            for attr, attr_encoder in self._encoder_attributes.items():
                self._encoder_attributes[attr]._classes = list(self._encoder_attributes[attr]._id2class_filt.values())[0]
        ## Class Initialization Properties
        self._token_encoder = token_encoder
        self._token_encoder_vocab = token_vocab
        self._token_encoder_frozen = freeze_encoder
        self._use_crf = use_crf
        self._use_lstm = use_lstm
        self._use_entity_token_bias = use_entity_token_bias
        self._use_attribute_concept_bias = use_attribute_concept_bias
        self._dropout = dropout
        self._max_sequence_length = max_sequence_length
        self._sequence_overlap = sequence_overlap
        self._sequence_overlap_type = sequence_overlap_type
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_bidirectional = lstm_bidirectional
        self._lstm_num_layers = lstm_num_layers
        self._entity_hidden_size = entity_hidden_size
        self._entity_token_bias_type = entity_token_bias_type
        self._attributes_hidden_size = attributes_hidden_size
        ## Initialize Token-level Encoder
        self.encoder = AutoModel.from_pretrained(token_encoder,
                                                 max_position_embeddings=self._max_sequence_length,
                                                 ignore_mismatched_sizes=True)
        ## NOTE: MODIFIED START
        self._token_encoder_dim = int(list(self.encoder.parameters())[-1].shape[0])
        ## NOTE: MODIFIED END
        ## Freeze Encoder if Desired
        if self._token_encoder_frozen:
            _ = self._freeze_encoder()
        ## Initialize LSTM if Desired
        self.combiner = None
        if self._use_lstm:
            self.combiner = torch.nn.LSTM(input_size=self._token_encoder_dim,
                                          hidden_size=self._lstm_hidden_size,
                                          num_layers=self._lstm_num_layers,
                                          bias=True,
                                          batch_first=True,
                                          dropout=self._dropout if self._lstm_num_layers > 1 else 0,
                                          bidirectional=self._lstm_bidirectional)
        ## Initialize Dropout
        self.dropout = torch.nn.Dropout(p=self._dropout)
        ## Initialize Entity Classifiers (Linear + CRF, If Relevant)
        if self._encoder_entity is not None:
            self.entity_heads = torch.nn.ModuleList()
            self.entity_crf = torch.nn.ModuleList() if self._use_crf else None
            ent_encoding_dim = self._lstm_hidden_size * (1 + self._lstm_bidirectional) if self._use_lstm else self._token_encoder_dim
            if self._use_entity_token_bias:
                ent_encoding_dim += len(self._encoder_entity.get_tasks()) * (1 + int(self._entity_token_bias_type == "positional"))
            for e, entity_type in enumerate(self._entities):
                self.entity_heads.append(MLP(in_dim=ent_encoding_dim,
                                             out_dim=3,
                                             hidden_dim=self._entity_hidden_size,
                                             random_state=self._random_state))
                if self._use_crf:
                    self.entity_crf.append(CRF(num_tags=3,
                                               batch_first=True))
        ## Initialize Attribute Classification Heads (If Relevant)
        if self._encoder_attributes is not None:
            self.attribute_heads = torch.nn.ModuleList()
            attr_encoding_dim = self._lstm_hidden_size * (1 + self._lstm_bidirectional) if self._use_lstm else self._token_encoder_dim
            for attribute, attribute_encoder in self._encoder_attributes.items():
                attribute_in_dim = attr_encoding_dim
                if self._use_attribute_concept_bias:
                    attribute_in_dim += len(attribute_encoder.get_tasks())
                self.attribute_heads.append(MLP(in_dim=attribute_in_dim,
                                                out_dim=len(attribute_encoder._classes),
                                                hidden_dim=self._attributes_hidden_size,
                                                random_state=self._random_state))

    def _freeze_encoder(self):
        """
        
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
 
    def _get_sequence_splits(self,
                             n_tokens,
                             max_sequence_length=MAX_BERT_LENGTH,
                             overlap=None,
                             overlap_type="mean"):
        """
        
        """
        ## Check Input
        if overlap is not None and overlap != 0 and overlap_type not in ["mean","first","last"]:
            raise KeyError("overlap_type not recognized.")
        ## Overlap
        overlap = 0 if overlap is None else overlap
        ## Range
        indices = list(range(n_tokens))
        ## Tracking Points
        ind_start = 0
        ind_end = max_sequence_length
        ## Build
        indices_split = []
        indices_weight = []
        while True:
            ## Add Indices to Cache
            indices_split.append(indices[ind_start:ind_end])
            ## Decide Inclusion Weights
            ind_weight = None
            if overlap == 0:
                ind_weight = [1.0 for _ in indices_split[-1]]
            else:
                ind_weight = []
                for i, s in enumerate(indices_split[-1]):
                    ## a) Only 1 subequence
                    if len(indices_split) == 1 and ind_end >= n_tokens:
                        ind_weight.append(1.0)
                    ## b) First subsequence with More to Come
                    elif len(indices_split) == 1:
                        if i < len(indices_split[-1]) - overlap:
                            ind_weight.append(1.0)
                        elif i >= len(indices_split[-1]) - overlap:
                            if overlap_type == "mean":
                                ind_weight.append(0.5)
                            elif overlap_type == "first":
                                ind_weight.append(1.0)
                            elif overlap_type == "last":
                                ind_weight.append(0.0)
                        else:
                            raise ValueError("Incorrect index logic.")
                    ## c) Final subsequence with More Before
                    elif ind_end >= n_tokens:
                        if i < overlap:
                            if overlap_type == "mean":
                                ind_weight.append(0.5)
                            elif overlap_type == "first":
                                ind_weight.append(0.0)
                            elif overlap_type == "last":
                                ind_weight.append(1.0)
                        elif i >= overlap:
                            ind_weight.append(1.0)
                        else:
                            raise ValueError("Incorrect index logic.")
                    ## d) Not the first or last subsequence
                    else:
                        if i < overlap:
                            if overlap_type == "mean":
                                ind_weight.append(0.5)
                            elif overlap_type == "first":
                                ind_weight.append(0.0)
                            elif overlap_type == "last":
                                ind_weight.append(1.0)
                            pass
                        elif i >= overlap and i < len(indices_split[-1]) - overlap:
                            ind_weight.append(1.0)
                        elif i >= len(indices_split[-1]) - overlap:
                            if overlap_type == "mean":
                                ind_weight.append(0.5)
                            elif overlap_type == "first":
                                ind_weight.append(1.0)
                            elif overlap_type == "last":
                                ind_weight.append(1.0)
                        else:
                            raise ValueError("Incorrect index logic.")
            ## Validate Inclusion Indices
            assert ind_weight is not None and len(ind_weight) == len(indices_split[-1])
            ## Cache Inclusion Indices
            indices_weight.append(ind_weight)
            ## Check For Break
            if ind_end >= n_tokens:
                break
            ## Update Boundaries
            ind_start = ind_end - overlap
            ind_end = ind_start + max_sequence_length
        ## Validate Boundaries
        assert indices[0] == indices_split[0][0]
        assert indices[-1] == indices_split[-1][-1]
        ## Return
        return indices_split, indices_weight
    
    def get_entities(self):
        """

        """
        return self._entities
    
    def get_attributes(self):
        """

        """
        return self._attributes
    
    def forward_encoding(self,
                         inputs):
        """

        """
        ## Get Splits
        input_splits, input_splits_weights = self._get_sequence_splits(n_tokens=inputs["input_ids"].size(1),
                                                                       max_sequence_length=self._max_sequence_length,
                                                                       overlap=self._sequence_overlap,
                                                                       overlap_type=self._sequence_overlap_type) 
        ## Build Token Encoding
        encoding = torch.zeros((inputs["input_ids"].size(0), inputs["input_ids"].size(1), self._token_encoder_dim)).to(inputs["input_ids"].device)
        for ind_split, ind_weight in zip(input_splits, input_splits_weights):
            ## Extract Subset of Encodings
            if self._token_encoder.startswith("random") or self._token_encoder.startswith("pretrained"):
                ind_enc = self.encoder(inputs["input_ids"][:,ind_split])
            else:
                ind_enc = self.encoder(input_ids=inputs["input_ids"][:,ind_split],
                                       token_type_ids=inputs["token_type_ids"][:,ind_split],
                                       attention_mask=inputs["attention_mask"][:,ind_split]).last_hidden_state
            ## Apply Appropriate Weighting
            ind_weight_t = torch.tensor([ind_weight for _ in range(ind_enc.size(0))]).unsqueeze(2).to(ind_enc.device)
            ## Add Weighted Subset to Full Encoding
            encoding[:,ind_split] = encoding[:,ind_split] + (ind_weight_t * ind_enc)
        ## Apply LSTM (To 'combine' separate subsets)
        if self._use_lstm:
            encoding = self.combiner(encoding)
            encoding = encoding[0]
        ## Return
        return encoding
    
    def decode_entity(self,
                      inputs,
                      logits):
        """

        """
        ## Check
        if self._encoder_entity is None:
            raise ValueError("Decoding not relevant without entities in the model.")
        ## Initialize Cache
        all_entity_tags = []
        ## Iterate Through Entities
        for l, ent_logits in enumerate(logits):
            ## Case 0: No CRF
            if not self._use_crf:
                ## Decode (Argmax)
                entity_tags = ent_logits.argmax(2)
            ## Case 1: CRF
            else:
                ## Decode
                entity_tags = self.entity_crf[l].decode(emissions=ent_logits,
                                                        mask=inputs["attention_mask"]==1)
                ## Pad
                entity_tags = torch.stack([torch.nn.functional.pad(torch.tensor(t), (0, ent_logits.size(1)-len(t)), "constant", 0) for t in entity_tags])
            ## Cache
            all_entity_tags.append(entity_tags)
        ## Stack
        all_entity_tags = torch.stack(all_entity_tags).permute(1,0,2)
        ## Return
        return all_entity_tags

    def forward_entity(self,
                       inputs,
                       encoding):
        """

        """
        ## Initialize Defaults
        all_entity_logits = None
        ## Concatenate Token-Entity Biases
        if self._use_entity_token_bias and inputs["entity_tokens"] is not None:
            if self._entity_token_bias_type == "positional":
                ent_in = torch.cat([encoding, inputs["entity_tokens"]], 2)
            elif self._entity_token_bias_type == "uniform":
                n_entities = len(self._encoder_entity.get_tasks())
                ent_in = torch.cat([encoding, inputs["entity_tokens"][:,:,:n_entities] + inputs["entity_tokens"][:,:,n_entities:]], 2)
            else:
                raise ValueError("Entity token bias type = '{}' not recognized".format(self._entity_token_bias_type))
        elif self._use_entity_token_bias and inputs["entity_tokens"] is None:
            raise ValueError("Expected non-null entity_tokens in inputs with `entity_token_bias` = True")
        elif not self._use_entity_token_bias:
            ent_in = encoding
        ## Apply Dropout
        ent_in = self.dropout(ent_in)
        ## Apply If Relevant
        if self._encoder_entity is not None:
            ## Initialize Cache
            all_entity_logits = []
            ## Iterate Through Layers and Compute Emissions
            for et, entity_head in enumerate(self.entity_heads):                    
                ## Pass Through Classification Head
                entity_logits = entity_head(ent_in)
                ## Cache Logits and Tags
                all_entity_logits.append(entity_logits)
            ## Reconstruct Output Format (Apply Appropriate Padding)
            all_entity_logits = torch.stack(all_entity_logits)
        ## Return
        return all_entity_logits

    def forward_attribute(self,
                          inputs,
                          encoding):
        """
        
        """
        ## Initialize Defaults
        all_attribute_logits = None
        ## Attribute Classification
        if self._encoder_attributes is not None:
            ## Initialize Cache
            all_attribute_logits = []
            ## Iterate Through Attributes
            for at, attribute_head in enumerate(self.attribute_heads):
                ## Meta
                attribute = self._attributes[at]
                n_attribute_concepts = len(self._encoder_attributes[attribute].get_tasks())
                ## Get Attribute spans
                at_spans = inputs["attribute_spans"][attribute]
                ## Skip If No Attributes in the Batch
                if at_spans["metadata"] is None:
                    all_attribute_logits.append(None)
                    continue
                ## Pool Relevant Embeddings
                at_representation = (at_spans["poolers"] @ encoding[at_spans["metadata"][:,1]]).squeeze(1)
                ## Add Concept Information
                if self._use_attribute_concept_bias:
                    ## One-hot Representation
                    at_concept_indicator = torch.nn.functional.one_hot(at_spans["metadata"][:,0], num_classes=n_attribute_concepts).float()
                    ## Concatenate with Encoding
                    at_representation = torch.cat([at_representation, at_concept_indicator], dim=1)
                ## Apply Dropout to Pooled Representations
                at_representation = self.dropout(at_representation)
                ## Forward Pass Through Attribute Classification Head
                at_logits = attribute_head(at_representation)
                ## Store
                all_attribute_logits.append(at_logits)
        ## Return
        return all_attribute_logits
    
    def forward(self,
                inputs):
        """

        """
        ## Individual Passes
        encoding = self.forward_encoding(inputs)
        entity_logits = self.forward_entity(inputs, encoding)
        attribute_logits = self.forward_attribute(inputs, encoding)
        ## Return
        return entity_logits, attribute_logits
    
class BaselineTaggerModel(object):

    """

    """

    def __init__(self,
                 encoder_entity=None,
                 encoder_attributes=None,
                 alpha=1,
                 use_char=False):
        """

        """
        ## Mode Tracking
        self._mode = None
        ## Smoothing
        self._alpha = alpha
        ## Whether to Use Raw Text Representation for "token" baseline
        self._use_char = use_char
        ## Encoders
        self._encoder_entity = encoder_entity
        self._encoder_attributes = encoder_attributes
        ## Classes
        self._entities = self._encoder_entity.get_tasks() if self._encoder_entity is not None else None
        self._attributes = None
        if self._encoder_attributes is not None:
            self._attributes = list(self._encoder_attributes.keys())
            for attr, attr_encoder in self._encoder_attributes.items():
                self._encoder_attributes[attr]._classes = list(self._encoder_attributes[attr]._id2class_filt.values())[0]
        ## Proxy Variables
        self._use_crf = False

    def get_entities(self):
        """

        """
        return self._entities
    
    def get_attributes(self):
        """

        """
        return self._attributes        

    def train(self):
        """

        """
        pass
    
    def eval(self):
        """

        """
        pass

    def _initialize_counts(self):
        """

        """
        ## Defaults
        self._counts_attributes = None
        ## Attribute Counts
        if self._encoder_attributes is not None:
            self._counts_attributes = {}
            for task, task_encoder in self._encoder_attributes.items():
                self._counts_attributes[task] = {}
                for task_ent in task_encoder.get_tasks():
                    self._counts_attributes[task][task_ent] = {}
                    for cls in task_encoder._classes:
                        self._counts_attributes[task][task_ent][cls] = Counter()
    
    def _transform_counts(self):
        """

        """
        if self._counts_attributes is not None:
            _counts_attributes_transformed = {"task":{}, "entity":{}, "token":{}}
            for task, task_encoder in self._encoder_attributes.items():
                cls2ind = dict(zip(task_encoder._classes, range(len(task_encoder._classes))))
                _counts_attributes_transformed["task"][task] = [self._alpha for cls in task_encoder._classes]
                _counts_attributes_transformed["entity"][task] = {ent:[self._alpha for cls in task_encoder._classes] for ent in task_encoder.get_tasks()}
                _counts_attributes_transformed["token"][task] = {ent:{} for ent in task_encoder.get_tasks()}
                for task_ent, ent_counts in self._counts_attributes.get(task, {}).items():
                    for cls, cls_counts in ent_counts.items():
                        cls_ind = cls2ind[cls]
                        for tok, tok_count in cls_counts.items():
                            if tok not in _counts_attributes_transformed["token"][task][task_ent]:
                                _counts_attributes_transformed["token"][task][task_ent][tok] = [self._alpha for cls in task_encoder._classes]
                            _counts_attributes_transformed["token"][task][task_ent][tok][cls_ind] += tok_count
                            _counts_attributes_transformed["entity"][task][task_ent][cls_ind] += tok_count
                            _counts_attributes_transformed["task"][task][cls_ind] += tok_count
            self._counts_attributes = _counts_attributes_transformed

    def fit(self,
            inputs):
        """

        """
        ## Initialize Counts
        self._initialize_counts()
        ## Gather Counts
        for x in inputs:
            ## Attribute Counts
            if self._encoder_attributes is not None:
                for task in x["attribute_spans"].keys():
                    task_spans = x["attribute_spans"][task]
                    task_char_spans = x["attribute_character_spans"][task]
                    task_entities = self._encoder_attributes[task].get_tasks()
                    for s, (task_ent, spans, char_spans) in enumerate(zip(task_entities, task_spans, task_char_spans)):
                        if len(spans) == 0:
                            continue
                        for ((cls, (tok_start, tok_end)), (cls_char, (char_start, char_end))) in zip(spans, char_spans):
                            assert cls == cls_char
                            cls_named = self._encoder_attributes[task]._classes[cls]
                            if self._use_char:
                                span_rep =  " ".join(x["text"][char_start:char_end].lower().split())
                            else:
                                span_rep = tuple(x["input_ids"][tok_start:tok_end].numpy())
                            self._counts_attributes[task][task_ent][cls_named][span_rep] += 1
        ## Transform Counts for Prediction
        _ = self._transform_counts()
        return self
    
    def set_mode(self,
                 mode):
        """

        """
        if mode not in ["task","entity","token"]:
            raise ValueError("Mode should be in [task, entity, token]")
        self._mode = mode
        return self

    def decode_entity(self,
                      inputs,
                      logits):
        """

        """
        ## Check
        if self._encoder_entity is None:
            raise ValueError("Decoding not relevant without entities in the model.")
        ## Initialize Cache
        all_entity_tags = []
        ## Iterate Through Entities
        for l, ent_logits in enumerate(logits):
            ## Decode (Argmax) and Cache
            all_entity_tags.append(ent_logits.argmax(2))
        ## Stack
        all_entity_tags = torch.stack(all_entity_tags).permute(1,0,2)
        ## Return
        return all_entity_tags

    def __call__(self,
                 inputs):
        """

        """
        ## Verify Mode Set
        if self._mode is None:
            raise ValueError("Mode has not yet been set. Call set_mode('<mode>') to choose a prediction mode.")
        if self._encoder_attributes is not None and self._counts_attributes is None:
            raise ValueError("Model has not yet been fit.")
        ## Initialize Default Out
        logits_entity, logits_attributes = None, None
        ## Apply Entity Logic (Propagate Entity Token Labels)
        if self._encoder_entity is not None:
            logits_entity = []
            entities = self._encoder_entity.get_tasks()
            for et, ent_type in enumerate(entities):
                o = torch.logical_and(inputs["entity_tokens"][:,:,et] == 0, inputs["entity_tokens"][:,:,et+len(entities)] == 0)
                b = torch.logical_and(inputs["entity_tokens"][:,:,et] == 1, inputs["entity_tokens"][:,:,et+len(entities)] == 0)
                i = torch.logical_and(inputs["entity_tokens"][:,:,et] == 0, inputs["entity_tokens"][:,:,et+len(entities)] == 1)
                logits_entity.append(torch.stack([o, b, i]).float().permute(1,2,0))
            logits_entity = torch.stack(logits_entity)
        ## Apply Attribute Logic
        if self._encoder_attributes is not None:
            logits_attributes = [[] for _ in self._encoder_attributes.keys()]
            for tt, task in enumerate(self._encoder_attributes.keys()):
                task_entities = self._encoder_attributes[task].get_tasks()
                if self._use_char:
                    input_task_spans = inputs["attribute_character_spans"].get(task)
                else:
                    input_task_spans = inputs["attribute_spans"].get(task)
                if input_task_spans is None or input_task_spans["metadata"] is None:
                    continue
                input_task_spans = input_task_spans["metadata"][:,[0,1,3,4]]
                for ii, (task_ent_id, input_ind, tok_start, tok_end) in enumerate(input_task_spans):
                    task_ent = task_entities[task_ent_id]
                    if self._mode == "task":
                        logits_attributes[tt].append(self._counts_attributes["task"][task])
                    elif self._mode == "entity":
                        if task_ent in self._counts_attributes["entity"][task]:
                            logits_attributes[tt].append(self._counts_attributes["entity"][task][task_ent])
                        else:
                            logits_attributes[tt].append(self._counts_attributes["task"][task])
                    elif self._mode == "token":
                        if self._use_char:
                            span_rep = " ".join(inputs["text"][input_ind][tok_start:tok_end].lower().split())
                        else:
                            span_rep = tuple(inputs["input_ids"][input_ind][tok_start:tok_end].numpy())
                        if task_ent in self._counts_attributes["token"][task] and span_rep in self._counts_attributes["token"][task][task_ent]:
                            logits_attributes[tt].append(self._counts_attributes["token"][task][task_ent][span_rep])
                        elif task_ent in self._counts_attributes["entity"][task]:
                            logits_attributes[tt].append(self._counts_attributes["entity"][task][task_ent])
                        else:
                            logits_attributes[tt].append(self._counts_attributes["task"][task])
            ## Normalize Counts as a Probability Distribution
            logits_attributes = list(map(lambda i: torch.tensor(i) if len(i) > 0 else None, logits_attributes))
            logits_attributes = list(map(lambda i: i / i.sum(1).unsqueeze(1) if i is not None else i, logits_attributes))
            logits_attributes = list(map(lambda i: i.float() if i is not None else i, logits_attributes))
        ## Return
        return logits_entity, logits_attributes
    
    def forward(self, inputs):
        """

        """
        return self.__call__(inputs)