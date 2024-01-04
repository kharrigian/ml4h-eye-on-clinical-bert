
#####################
### Imports
#####################

## Standard Library
import re
import string
from collections import Counter

## External Libraries
from transformers import AutoTokenizer

## Local
from ..util.helpers import flatten

#####################
### Classes
#####################

class AttributeTaggingTokenizer(object):
    
    """
    AttributeTaggingTokenizer
    """

    def __init__(self,
                 tokenizer_fcn,
                 max_sequence_length=None,
                 sequence_overlap=None,
                 split_type="continuous"):
        """
        Args:
            tokenizer_fcn (callable): Should take a string as input and return a list as an output
        """
        ## Class Attributes
        self._tokenizer_fcn = tokenizer_fcn
        self._max_sequence_length = max_sequence_length
        self._sequence_overlap = 0 if sequence_overlap is None else sequence_overlap
        self._split_type = split_type
        ## Verify Split Type
        if self._split_type not in ["continuous","centered"]:
            raise ValueError("'split_type' should either be set to 'continuous' or 'centered'.")
        if self._split_type == "centered" and (sequence_overlap is not None and sequence_overlap > 0):
            print(">> WARNING: sequence_overlap is not relevant when using a 'centered' split_type.")
        ## Check Initialization
        _ = self._verify_tokenizer_function(tokenizer_fcn)
    
    def _verify_tokenizer_function(self,
                                   tokenizer_fcn):
        """
        
        """
        ## Simple Text Input
        sample_input = "This is a text string"
        ## Attempt To Tokenize Text Input
        try:
            sample_output = tokenizer_fcn(sample_input)
        except:
            raise ValueError("Tokenizer function should accept simple text inputs.")
        ## Verify Type of Output
        if not isinstance(sample_output, list):
            raise TypeError("Tokenizer function should output a list.")
        ## All Good
        return True
    
    def __repr__(self):
        """
        
        """
        return "AttributeTaggingTokenizer()"
    
    def _verify_labels(self,
                       label):
        """
        
        """
        assert isinstance(label, dict)
        assert "start" in label.keys()
        assert "end" in label.keys()
        assert "valid" in label.keys()
        assert "label" in label.keys()

    def _assign_labels_to_text_spans(self,
                                     text_spans,
                                     labels):
        """
        
        """
        ## Group the Labels
        label_spans = {}
        for l in labels:
            l_span = (l.get("start"), l.get("end"))
            if l_span not in label_spans:
                label_spans[l_span] = []
            label_spans[l_span].append(l)
        ## Sort Boundaries and Labels
        text_span_boundaries = sorted(text_spans.keys())
        label_span_boundaries = sorted(label_spans.keys())
        ## Make Label Assignments to Each Text Spa
        i = 0
        j_bound = 0
        assignments = []
        while i < len(text_span_boundaries):
            cur_text_assignments = []
            cur_text_start, cur_text_end = text_span_boundaries[i]
            j = j_bound
            while j < len(label_span_boundaries):
                inner_lbl_start, inner_lbl_end = label_span_boundaries[j]
                if cur_text_end <= inner_lbl_start: ## Text still before next label in line
                    break
                if inner_lbl_end <= cur_text_start: ## Start of Text has passed End of Previous Label Bound
                    j_bound += 1
                if cur_text_start >= inner_lbl_start and cur_text_end <= inner_lbl_end: ## Text span lies within label boundary
                    cur_text_assignments.append(j)
                j += 1
            assignments.append(cur_text_assignments)
            i += 1
        ## Format Assignments
        text_spans_labels = {}
        for (tstart, tend), tassign in zip(text_span_boundaries, assignments):
            text_spans_labels[(tstart, tend)] = []
            for lind in tassign:
                for lbl in label_spans[label_span_boundaries[lind]]:
                    text_spans_labels[(tstart, tend)].append(lbl)
        return text_spans_labels
    
    def _split_sequence_centered(self,
                                 tokens_and_labels,
                                 max_length):
        """

        """
        ## Identify Token Boundaries
        token_boundaries = {}
        for t, (tok, tok_lbls) in enumerate(tokens_and_labels):
            for (lbl, lbl_new, lbl_valid) in tok_lbls:
                if lbl_new:
                    token_boundaries[(lbl["label"], lbl["start"], lbl["end"])] = [t]
                else:
                    token_boundaries[(lbl["label"], lbl["start"], lbl["end"])].append(t)
        ## Format
        token_boundaries = {x:(min(y), max(y)) for x, y in token_boundaries.items()}
        ## Reverse
        token_boundaries_r = {}
        for instance, instance_bounds in token_boundaries.items():
            if instance_bounds not in token_boundaries_r:
                token_boundaries_r[instance_bounds] = []
            token_boundaries_r[instance_bounds].append(instance)
        ## Iterate Through Token Boundaries
        sequences = []
        for (tok_start, tok_end), bound_labels in token_boundaries_r.items():
            i_seq = [(tok, [(tl, tnew, tval) for tl, tnew, tval in tok_lbls if (tl["label"],tl["start"],tl["end"]) in bound_labels]) for tok, tok_lbls in tokens_and_labels[tok_start:tok_end+1]]
            i_seq_l = tok_start
            i_seq_r = tok_end
            while True:
                if ((i_seq_r + 1) - i_seq_l == max_length) or (i_seq_l == 0 and i_seq_r == len(tokens_and_labels) - 1):
                    break
                if i_seq_l > 0:
                    i_seq_l -= 1
                if (i_seq_r + 1) - i_seq_l != max_length and i_seq_r < len(tokens_and_labels) - 1:
                    i_seq_r += 1
            l_window = [(tok, []) for tok, tl in tokens_and_labels[i_seq_l:tok_start]]
            r_window = [(tok, []) for tok, tl in tokens_and_labels[tok_end+1:i_seq_r+1]]
            i_seq = l_window + i_seq + r_window
            sequences.append(i_seq)
        ## Validate
        assert all(len(s) <= max_length for s in sequences)
        ## Done
        return sequences

    def _split_sequence_continuous(self,
                                   tokens_and_labels,
                                   max_length,
                                   sequence_overlap):
        """
        
        """
        ## Initialize Split
        split_tokens_and_labels = []
        ## Initialize Indices
        cur_split_start = 0
        cur_split_end = max_length
        ## Token Labels Helper
        get_token_labels = lambda tok: [(i["label"],i["start"],i["end"]) for i, _, _ in tok[1]]
        ## Add Until Done
        while True:
            ## Ensure Splits Preserve Entity Groups
            if cur_split_end < len(tokens_and_labels):
                ## Move End Back As Necessary
                while True:
                    ## Proposed End Token and Next Start Token
                    token_boundary = tokens_and_labels[cur_split_end-1]
                    token_boundary_next = tokens_and_labels[cur_split_end]
                    ## Get Boundary Labels
                    token_boundary_lbls = get_token_labels(token_boundary)
                    token_boundary_next_lbls = get_token_labels(token_boundary_next)
                    ## Check for Overlap
                    if len(set(token_boundary_lbls) & set(token_boundary_next_lbls)) > 0:
                        cur_split_end -= 1
                        ## Check for Error
                        if cur_split_start == cur_split_end:
                            raise ValueError("Encountered error segmenting tokens while preserving label groups.")
                    else:
                        break
            ## Add Sequence of Max Length
            split_tokens_and_labels.append(tokens_and_labels[cur_split_start:cur_split_end])
            ## Check Completion
            if cur_split_end >= len(tokens_and_labels):
                break
            ## Update Sequence Boundaries
            cur_split_start = cur_split_end - sequence_overlap
            cur_split_end = cur_split_start + max_length
        ## Validate Boundaries
        assert tokens_and_labels[0] == split_tokens_and_labels[0][0]
        assert tokens_and_labels[-1] == split_tokens_and_labels[-1][-1]
        ## Return
        return split_tokens_and_labels
    
    def _add_special_tokens(self,
                            tokens_and_labels,
                            cls_token,
                            sep_token):
        """
        
        """
        for i, tl in enumerate(tokens_and_labels):
            if cls_token is not None:
                tl = [(cls_token, [])] + tl
            if sep_token is not None:
                tl = tl + [(sep_token, [])]
            tokens_and_labels[i] = tl
        return tokens_and_labels
            
    def tokenize(self,
                 text,
                 labels,
                 cls_token=None,
                 sep_token=None):
        """
        Args:
            text (str): Input text associated with labels.
            labels (list of dict): Each input dict should have the following keys: start, end, valid, label
            cls_token (str or None): If desired, add a classification token to start of tokenized instance
            sep_token (str or None): If desired, add a separation token to end of tokenized_instance
        
        Returns:
            tokens_and_labels (list of tuple): [(token, list of token_labels), ...]
        
        ```
        ## Example
        text = "His name is Paul and he loves Ferrari"
        labels = [
            {"start":12,"end":16,"label":"name","valid":True},
            {"start":24,"end":37,"label":"interest","valid":True},
            {"start":30,"end":37,"label":"org","valid":True}
        ]
        IN: tokenizer.tokenize(text, labels)
        OUT: [('His', []),
         ('name', []),
         ('is', []),
         ('Paul', [({'start': 12, 'end': 16, 'label': 'name', 'valid': True}, True)]),
         ('and', []),
         ('he', []),
         ('loves',
          [({'start': 24, 'end': 37, 'label': 'interest', 'valid': True}, True)]),
         ('Ferrari',
          [({'start': 24, 'end': 37, 'label': 'interest', 'valid': True}, False),
           ({'start': 30, 'end': 37, 'label': 'org', 'valid': True}, True)])]
        ```
        """
        ## Verify Text
        if text is None:
            return None
        ## Verify Label Formatting
        _ = list(map(self._verify_labels, labels))
        ## Isolate Valid and Invalid Labels
        valid_labels = set([(l["start"], l["end"], l["label"]) for l in labels if l["valid"]])
        invalid_labels = set([(l["start"], l["end"], l["label"]) for l in labels if not l["valid"]])
        ## Edgecase: No Labels
        if len(labels) == 0:
            ## Format As Null
            tokens_and_labels = [(tok, []) for tok in  self._tokenizer_fcn(text)]
        ## Primary Case: Labels Present
        else:
            ## Identify and Sort Span Endpoints
            endpoints = sorted(set(flatten(list(map(lambda l: [l.get("start"),l.get("end")], labels)))))
            if endpoints[0] != 0:
                endpoints = [0] + endpoints
            if endpoints[-1] != len(text):
                endpoints = endpoints + [len(text)]
            ## Break Up Text Input and Tokenizer
            text_spans = {}
            for estart, eend in zip(endpoints[:-1], endpoints[1:]):
                text_spans[(estart, eend)] = self._tokenizer_fcn(text[estart:eend])
            ## Make Label Assignments to Text Spans
            text_spans_labels = self._assign_labels_to_text_spans(text_spans=text_spans, labels=labels)
            ## Token-level Label Assignments
            tokens_and_labels = []
            labels_seen = set()
            for text_span, text_span_tokens in text_spans.items():
                for token in text_span_tokens:
                    token_labels = []
                    for l in text_spans_labels[text_span]:
                        lid = (l["start"], l["end"], l["label"]) ## Labels are uniquely identified by location and concept label
                        token_labels.append((l, lid not in labels_seen, lid in valid_labels))
                        labels_seen.add(lid)
                    tokens_and_labels.append((token, token_labels))
        ## Splitting/Overlap
        if self._max_sequence_length is not None and self._split_type == "continuous":
            tokens_and_labels = self._split_sequence_continuous(tokens_and_labels,
                                                                max_length=self._max_sequence_length - (int(cls_token is not None) + int(sep_token is not None)),
                                                                sequence_overlap=self._sequence_overlap)
        elif self._max_sequence_length is not None and self._split_type == "centered":
            tokens_and_labels = self._split_sequence_centered(tokens_and_labels,
                                                              max_length=self._max_sequence_length - (int(cls_token is not None) + int(sep_token is not None)))
        else:
            tokens_and_labels = [tokens_and_labels]
        ## Add Separation/Classification tokens
        tokens_and_labels = self._add_special_tokens(tokens_and_labels, cls_token=cls_token, sep_token=sep_token)
        return tokens_and_labels

class PostProcessTokenizer(object):

    """
    
    """

    def __init__(self,
                 tokenizer):
        """
        
        """
        ## Assign Primary Tokenizer
        self._tokenizer = tokenizer
        self._vocab = self._tokenizer.vocab
        self._vocab_r = {y:x for x, y in self._vocab.items()}
        ## Validate Primary Tokenizer
        if not hasattr(self._tokenizer, "tokenize") or not callable(self._tokenizer.tokenize):
            raise ValueError("'tokenizer' should be a class object with a tokenize method.")
        ## Assign Certain Attributes
        for att in ["cls_token","sep_token","unk_token","pad_token","mask_token"]:
            _ = setattr(self, att, getattr(self._tokenizer, att))

    def __repr__(Self):
        """
        
        """
        return "PostProcessTokenizer()"

    def tokenize(self,
                 text):
        """
        
        """
        ## Run Tokenization
        tokens = self._tokenizer.tokenize(text)
        ## Return
        return tokens

    def get_vocab(self):
        """
        
        """
        return self._vocab

    def _reset_vocab(self):
        """
        
        """
        ## Reset
        self._vocab = {x:self._tokenizer.vocab[x] for x in self._tokenizer.all_special_tokens}
        self._vocab_r = {y:x for x, y in self._vocab.items()}
        ## Add Unused Token Slots
        max_ind = max(self._vocab_r.keys())
        for i in range(max_ind):
            if i not in self._vocab_r:
                self._vocab[f"[unused {i}]"] = i
                self._vocab_r[i] = f"[unused {i}]"
 
    def _get_token_id(self,
                      token):
        """
        
        """
        if token in self._vocab:
            return self._vocab[token]
        elif token not in self._vocab:
            return self._vocab[self.unk_token]
        else:
            raise ValueError("This will never happen.")
    
    def convert_tokens_to_ids(self,
                              tokens):
        """
        
        """
        token_ids = list(map(self._get_token_id, tokens))
        return token_ids 
