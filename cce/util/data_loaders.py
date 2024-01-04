
"""
Data loader utilities
"""

#######################
### Imports
#######################

## Standard Library
import os
import re
import sys
import json
import gzip
from copy import copy
from functools import partial
from collections import Counter
from multiprocessing import Pool

## External
import numpy as np
import pandas as pd
from tqdm import tqdm

## Local
from .patterns import (ICD10_DM_AUTO_LABELS,
                       ICD10_DM_AUTO_LABELS_GROUPS,
                       ICD10_AUTO_LABELS,
                       AUTO_LABELS)

#######################
### Classes
#######################

class AutoLabeler(object):
    
    """
    AutoLabeler
    """

    def __init__(self,
                 auto_labels=None,
                 icd10_auto_labels=None,
                 handle_icd10_strings=None,
                 handle_icd10_codes=None,
                 handle_icd10_expand=None):
        """
        When using 'unseen' for the handling of icd10 codes/strings, we use the following priority rank: free text -> ICD code -> ICD headers

        Args:
            auto_labels (None, dict, str): Regular expression patterns to apply to free text in note. Set = "default" to use those in patterns file
            icd10_auto_labels (None, dict, str): Regular expression patterns to apply to ICD codes. Set = "default" to use those in patterns file
            handle_icd10_codes (None or str): None = ignore, 'unseen' = labels not found in free text but found in codes, 'all' = include code matches regardless
            handle_icd10_strings (None or str): None = ignore, 'unseen' = labels not found in free text but found in headers, 'all' = include header matches regardless
            handle_icd10_expand (None or str): If True, adds default set of labels (DR / ME / DM, ...) for diabetes related ICD codes
        """
        ## Defaults
        if isinstance(auto_labels, str):
            if auto_labels != "default":
                raise ValueError("auto_labels string not recognized")
            auto_labels = AUTO_LABELS
        if isinstance(icd10_auto_labels, str):
            if icd10_auto_labels != "default":
                raise ValueError("icd10_auto_labels not recognized")
            icd10_auto_labels= ICD10_AUTO_LABELS
        ## Set Attributes
        self._handle_icd10_strings = handle_icd10_strings
        self._handle_icd10_codes = handle_icd10_codes
        self._handle_icd10_expand = handle_icd10_expand
        ## Initialize Patterns
        self._init_regex(auto_labels, icd10_auto_labels)
        ## Validate Attributes
        if self._handle_icd10_strings is not None and self._handle_icd10_strings not in ["unseen","all"]:
            raise ValueError("Expected handle_icd10_strings to be either None (ignore), 'unseen', or 'all'")
        if self._handle_icd10_codes is not None and self._handle_icd10_codes not in ["unseen","all"]:
            raise ValueError("Expected handle_icd10_codes to be either None (ignore), 'unseen', or 'all'")
        if self._handle_icd10_expand is not None and self._handle_icd10_expand not in ["unseen","all"]:
            raise ValueError("Expected handle_icd10_expand to be either None (ignore), 'unseen', or 'all'")

    def _init_regex(self,
                    auto_labels=None,
                    icd10_auto_labels=None):
        """
        
        """
        ## Auto Label Patterns
        self._patterns = {}
        if auto_labels is not None:
            for lbl, lbl_pats in auto_labels.items():
                pats_sens = [pat for pat, cs in lbl_pats if cs]
                pats_inse = [pat for pat, cs in lbl_pats if not cs]
                self._patterns[lbl] = [
                    re.compile("|".join(["({})".format(pat) for pat in pats_sens]), re.UNICODE) if len(pats_sens) > 0 else None,
                    re.compile("|".join(["({})".format(pat) for pat in pats_inse]), re.UNICODE | re.IGNORECASE) if len(pats_inse) > 0 else None
                ]
                self._patterns[lbl] = list(filter(lambda i: i is not None, self._patterns[lbl]))
        ## ICD-10 Auto Label Patterns
        self._patterns_icd = {}
        if icd10_auto_labels is not None:
            for lbl, lbl_pats in icd10_auto_labels.items():
                if len(lbl_pats) > 0:
                    self._patterns_icd[lbl] = re.compile("|".join([r"(\b{}\b)".format(pat) for pat in lbl_pats]), re.UNICODE)
        ## ICD-10 DM Auto Labels
        self._patterns_dm_icd = {}
        self._patterns_e = None
        if self._handle_icd10_expand is not None:
            ## Generic E-Code
            self._patterns_e = re.compile(r"(E((0(8|9))|(1(0|1|3))))(\w|\.)*\b", flags=re.UNICODE)
            ## E-Code to Label
            for lbl, lbl_pats in ICD10_DM_AUTO_LABELS.items():
                if len(lbl_pats) == 0:
                    continue
                self._patterns_dm_icd[lbl] = re.compile("|".join([r"(\b{}\b)".format(pat) for pat in lbl_pats]), re.UNICODE)
        ## Cleaning of Headers
        self._pcode = re.compile(r"\[\[\[ENCOUNTER ICD-10 CODES\]\]\]\n")
        self._pnote = re.compile(r"\[\[\[PROGRESS NOTE\]\]\]\n")
        self._plist_sec = re.compile(r"\[\[\[PROBLEM LIST\]\]\]\n\n")
        self._plist_problem_head = re.compile(r"\[\[(.*\:\s.*)\]\]\n")
        self._plist_problem_overview = re.compile(r"\n\[OVERVIEW\]\s")
        self._plist_problem_assessment = re.compile(r"\n\[ASSESSMENT & PLAN\]\s")

    def _get_header_spans(self,
                          document):
        """
        
        """
        ## Find Headers
        matches = []
        for pattern in [self._pcode, self._pnote, self._plist_sec, self._plist_problem_head, self._plist_problem_overview, self._plist_problem_assessment]:
            for pat_match in pattern.finditer(document):
                span = list(pat_match.span())
                if document[span[0]] == "\n":
                    span[0] += 1
                matches.append(tuple(span))
        ## Sort
        matches = sorted(set(matches))
        return matches
    
    def _expand_icd_codes(self,
                          document):
        """
        
        """
        ## Check Handle
        if self._handle_icd10_expand is None:
            return None
        ## Identify Relevant E-Codes
        ecodes = [(i.span(), document[i.start():i.end()]) for i in self._patterns_e.finditer(document)]
        if len(ecodes) == 0:
            return None
        ## Identify any Known Concept Labels (e.g., NPDR, ME)
        ecodes_labels = {}
        for ec in ecodes:
            ## Ignore Duplicate Codes
            if ec[1] in ecodes_labels:
                continue
            ## Established Matched Cache
            ecodes_labels[ec[1]] = {"matched":set(),"automatic":set(), "span":ec[0]}
            ## Search For Known Labels
            for lbl, lbl_re in self._patterns_dm_icd.items():
                if lbl_re.search(ec[1]) is not None:
                    ecodes_labels[ec[1]]["matched"].add(lbl)
        ## Add Candidates Not Specified by Existing Code
        for ec, ec_set in ecodes_labels.items():
            for dlbl_def, dlbls in ICD10_DM_AUTO_LABELS_GROUPS.items():
                ## Ignore If ICD Code Already Encodes the Attribute
                if any(d in ec_set["matched"] for d in dlbls):
                    continue
                ## Add Automatic Label if It Doesn't Exist Based on Code
                else:
                    ec_set["automatic"].add(dlbl_def)
        ## Format Relevant Labels (A/B/C label per Code | F-G label per return because they don't have laterality)
        ecodes_out = []
        ecodes_count = Counter()
        ## Prioritize Matched Labels Over Auto Labels (Due to count limit)
        for ec, ec_set in ecodes_labels.items():
            for lbl in ec_set["matched"]:
                ## Max Addition Check
                lbl_limit = 2 if any(lbl.startswith(char) for char in ["F","G"]) else 1
                if ecodes_count[lbl] >= lbl_limit:
                    continue
                ## Add
                ecodes_out.append([*ec_set["span"], lbl])
                ecodes_count[lbl] += 1
        ## Include Any Remaining Auto Labels (don't exceed count limit)
        for ec, ec_set in ecodes_labels.items():
            for lbl in ec_set["automatic"]:
                ## Max Addition Check
                lbl_limit = 2 if any(lbl.startswith(char) for char in ["F","G"]) else 1
                if ecodes_count[lbl] >= lbl_limit:
                    continue
                ## Add
                ecodes_out.append([*ec_set["span"], f"{lbl} <<AUTO>>"])
                ecodes_count[lbl] += 1
        return ecodes_out
    
    def _add_to_labels(self,
                       labels,
                       labels_proposed,
                       handle):
        """
        
        """
        ## Check Proposed
        if labels_proposed is None:
            return labels
        ## Copy The Labels
        labels = copy(labels)
        ## Case 1: Include Everything
        if handle == "all":
            labels = labels + labels_proposed
        ## Case 2: Include New + Count Limited
        elif handle == "unseen":
            ## Current Label Count
            labels_count = Counter([i[-1].replace(" <<AUTO>>","") for i in labels])
            ## Check Proposed Against Count
            labels_proposed_filtered =[]
            for lbl in labels_proposed:
                ## Label Name
                lbl_name = lbl[-1].replace(" <<AUTO>>","")
                ## Check Label Specific Limit for Additions
                lbl_limit = 1 if any(lbl_name.startswith(char) for char in ["F","G"]) else 2
                if labels_count[lbl_name] >= lbl_limit:
                    continue
                ## Cache
                labels_proposed_filtered.append(lbl)
                labels_count[lbl_name] += 1
            ## Append Appropriate Labels
            labels = labels + labels_proposed_filtered
        else:
            raise ValueError("Handle parameter not recognized.")
        return labels

    def _search_text(self,
                     document,
                     problem_spans):
        """
        
        """
        ## Initialize Cache
        labels = []
        labels_problems = []
        ## Iterate Through Patterns
        for lbl, lbl_pat_regexes in self._patterns.items():
            ## Iterate through pattern sets
            for lbl_pat_regex in lbl_pat_regexes:
                ## Iterate Over Matches
                for match in lbl_pat_regex.finditer(document):
                    ## Keep Track of Header Matches
                    if self._handle_icd10_strings is not None and any(match.start() >= ph[0] and match.end() <= ph[1] for ph in problem_spans):
                        labels_problems.append([*match.span(), lbl])
                        continue
                    ## Check Against Note Headers
                    if self._handle_icd10_strings is None and any(match.start() >= ph[0] and match.end() <= ph[1] for ph in problem_spans):
                        continue
                    ## If In Main Text, Add Match
                    labels.append([*match.span(), lbl])
        return labels, labels_problems
    
    def _search_icd(self,
                    document,
                    problem_spans,
                    formatted):
        """
        
        """
        ## Cache
        labels_icd = []
        ## Iterate Through Patterns
        for lbl, lbl_pat_regex in self._patterns_icd.items():
            ## Iterate Over Potential Matches
            for match in lbl_pat_regex.finditer(document):
                ## If Formatted Note, Ensure Match in Header
                if formatted and not any(match.start() >= dhs[0] and match.end() <= dhs[1] for dhs in problem_spans):
                    continue
                labels_icd.append([*match.span(), lbl])
        ## Remove Any Duplicate ICD Code Entries
        if len(labels_icd) > 0:
            labels_icd = list({(document[s:e], lbl):[s, e, lbl] for s, e, lbl in labels_icd[::-1]}.values())
        return labels_icd

    def get_auto_labels(self,
                        document,
                        formatted=True):
        """
        Args:
            document (str): Input text
            formatted (bool): Whether the input text is formatted as a human-readable note (i.e., with section headers)
        
        Returns:
            labels (list): Any identified text spans and their relevant indices
        """
        ## Check For Null or Non-String Document
        if document is None or not isinstance(document, str):
            return []
        ## Isolate Problem List Patterns
        document_problem_spans = [i.span() for i in self._plist_problem_head.finditer(document)]
        ## Add Text Labels
        labels, labels_problems = self._search_text(document=document,
                                                    problem_spans=document_problem_spans)
        ## Add ICD Labels
        labels_icd = self._search_icd(document=document,
                                      problem_spans=document_problem_spans,
                                      formatted=formatted)
        ## Add Expanded ICD Labels
        labels_icd_expand = self._expand_icd_codes(document)
        ## First Priority: Add Automatic Labels Based on ICD E-Codes (Diabetes)
        if self._handle_icd10_expand is not None:
            labels = self._add_to_labels(labels=labels,
                                         labels_proposed=labels_icd_expand,
                                         handle=self._handle_icd10_expand)

        ## Second Priority: Add Labels Identified by ICD 10 Codes (All)
        if self._handle_icd10_codes is not None:
            labels = self._add_to_labels(labels=labels,
                                         labels_proposed=labels_icd,
                                         handle=self._handle_icd10_codes)
        ## Third Priority: Add Labels Identified by Strings in the ICD 10 Problem Headers
        if self._handle_icd10_strings is not None:
            labels = self._add_to_labels(labels=labels,
                                         labels_proposed=labels_problems,
                                         handle=self._handle_icd10_strings)
        ## Dedpuplicate and Sort the Labels Based on Location
        labels = set(list(map(tuple, labels)))
        labels = list(map(list, sorted(labels, key=lambda x: (x[0], x[-1]))))
        return labels

#######################
### Complied Expressions
#######################

_SURGICAL_LABEL_SPLITTER_RE = re.compile(r"((\s)and(\s))|((\,|\/|\+)(\s)?((and|or)(\s))?)")
_PNOTE_RE = re.compile(r"\[\[\[PROGRESS NOTE\]\]\]")
_PLIST_RE = re.compile(r"\[\[\[PROBLEM LIST\]\]\]")

#######################
### Functions
#######################

def _load_formatted_json(filename):
    """
    
    """
    ## Load
    with open(filename,"r") as the_file:
        data = json.load(the_file)
    ## Check Type
    assert isinstance(data, list)
    ## Check Format
    for d in data:
        assert isinstance(d, dict)
        assert "text" in d
        assert "document_id" in d
    ## Format
    data_fmt = {
        "labels":sorted(AUTO_LABELS.keys()),
        "documents":data
    }
    ## Return
    return data_fmt
    
def _load_formatted_csv(filename):
    """
    Load a hashed-notes CSV and format as if it were a PINE collection
    """
    ## Load and Validate
    data = pd.read_csv(filename)
    if not all(c in data.columns for c in ["document_id","text"]):
        raise KeyError("Missing expected columns from data: [document_id, text]")
    ## Re-Format
    data_fmt = {
        "labels":sorted(AUTO_LABELS.keys()),
        "documents":[]
    }
    for _, row in data.iterrows():
        data_fmt["documents"].append({
            "text":row["text"],
            "metadata":{"document_id":row["document_id"]},
        })
    ## Return
    return data_fmt

def _split_surgical_label_span(lbl_span,
                               lbl_start,
                               lbl_end):
    """
    
    """
    ## Base Case Validation
    if lbl_span is None or not isinstance(lbl_span, str):
        return [(lbl_start, lbl_end)]
    ## Apply Split
    lbl_span_split = [(i.start(), i.end()) for i in list(_SURGICAL_LABEL_SPLITTER_RE.finditer(lbl_span))]
    ## Case 1: No Additional Splits
    if len(lbl_span_split) == 0:
        return [(lbl_start, lbl_end)]
    ## Case 2: Additional Splits
    lbl_cur_start = lbl_start
    lbl_splits = []
    for lstart, lend in [(lbl_start + i, lbl_start + j) for i, j in lbl_span_split]:
        lbl_splits.append((lbl_cur_start, lstart))
        lbl_cur_start = lend
    lbl_splits.append((lbl_cur_start, lbl_end))
    ## Return
    return lbl_splits

def _merge_surgical_label_span(text,
                               lbl_spans):
    """
    
    """
    ## Check Type
    if text is None or not isinstance(text, str):
        return lbl_spans
    ## Base Case: No Possible Joins
    if len(lbl_spans) == 1:
        return lbl_spans
    ## Sort the Spans
    lbl_spans_s = sorted(lbl_spans, key=lambda x: x[0])
    ## Make Merge Decisions
    merge_bool = []
    for (_, l1, _), (l2, _, _) in zip(lbl_spans_s[:-1], lbl_spans_s[1:]):
        ## Get Text Between Current Spans
        between = text[l1:l2].strip()
        ## Base Case (Relatively Long Character Span)
        if len(between) >= 50:
            merge_bool.append(False)
        else:
            ## Remove Expected Joiners From Count
            between = re.sub(r"(s\/p)", "sp", between, flags=re.IGNORECASE) ## status post normalization
            between = re.sub(r"\b(OS|OD|OU|RE|LE|SP|status((\s|\-)+)post|(left|right)((\s+)eye)?)\b", "", between, flags=re.IGNORECASE) ## laterality
            between = re.sub(r"\b(inferior|removal|possible|injection((\s|\-)+)of|extensive|epi(\-)?retinal((\s|\-)+)membrane|epi|erm|prp)\b","", between, flags=re.IGNORECASE) ## short-hand surgical
            between = re.sub(r"(cataract(\s+)(extraction|surgery))", "", between, flags=re.IGNORECASE) ## cataract-related procedures
            between = re.sub(r"((posterior|laser|anterior)(\s+))?(capsulotomy)","", between, flags=re.IGNORECASE)
            between = re.sub(r"((peripheral(\s+))?irid(ec|o)tomy)", "", between, flags=re.IGNORECASE)
            between = re.sub(r"\b(CEIOL)\b", "", between, flags=re.IGNORECASE)
            between = re.sub(r"\b(\d(\d?)\/\d\d?(\/\d\d{1,3})?)\b", "", between) ## dates
            between = re.sub(r"\b((19|20)\d{2})\b","",between) ## years
            between = re.sub(r"\W"," ",between).strip().split() ## Non-punctuation
            ## Check Token Count
            merge_bool.append(len(between) <= 1)
    ## Second Base Case (No Merging)
    if not any(merge_bool):
        return lbl_spans
    ## Apply Merging
    merged_spans = []
    cur_start, cur_end = lbl_spans_s[0][:2]
    for (lstart, lend, _), apply_merge in zip(lbl_spans_s[1:], merge_bool):
        if apply_merge:
            cur_end = lend
        else:
            merged_spans.append((cur_start, cur_end))
            cur_start, cur_end = lstart, lend
    if len(merged_spans) == 0 or merged_spans[-1] != (cur_start, cur_end):
        merged_spans.append((cur_start, cur_end))
    ## Format
    lbl_spans_merged = [[s, e, text[s:e]] for s,e in merged_spans]
    return lbl_spans_merged

def _merge_anti_vegf_label_span(text,
                                lbl_spans):
    """
    
    """
    ## Check Type
    if text is None or not isinstance(text, str):
        return lbl_spans
    ## Base Case (No Merging)
    if len(lbl_spans) <= 1:
        return lbl_spans
    ## Sort the Spans
    lbl_spans_s = sorted(lbl_spans, key=lambda x:x[0])
    ## Make Merge Decisions
    merge_bool = []
    for (_, l1, _), (l2, _, _) in zip(lbl_spans_s[:-1], lbl_spans_s[1:]):
        ## Get Text Between Current Spans
        between = text[l1:l2].strip()
        ## Base Case (Relatively Long Character Span)
        if len(between) >= 250:
            merge_bool.append(False)
        else:
            ## Acceptable Mergers
            between = between.replace(",","").replace(".","") ## Simple punctuation
            between = re.sub(r"(s\/p)", "sp", between, flags=re.IGNORECASE) ## status post normalization
            between = re.sub(r"\b(OS|OD|OU|SP|RE|LE|(left|right)((\s+)eye)?)\b", "", between, flags=re.IGNORECASE) ## laterality (and status-post)
            between = re.sub(r"\b(\d(\d?)\/\d\d?(\/\d\d{1,3}))\b","",between) ## dd/mm/yyyy dates
            between = re.sub(r"\b((\d\d\/\d\d)(\-\d)?(\s+)?(ph)?)\b", "", between, flags=re.IGNORECASE) ## visual acuity
            between = re.sub(r"\b(\d*)(ph|gel|x|csf|mg)\b", "", between, flags=re.IGNORECASE) ## visual acuity/dosage
            between = re.sub(r"\d+","",between) ## all remaining numbers
            between = re.sub(r"\b((no(\s+)(rx|treatment))|missed(\s+)visit(s)?|last|skipped|defer(ral|red|ring)?|consider|(after(\s+)(nearly(\s+))?)?(week|month|year|mo(n)?|wk|yr)(s)?)\b", "",  between,flags=re.IGNORECASE) ## timing
            between = re.sub(r"\W"," ",between).strip().split() ## isolate alpha-numeric
            if len(between) > 1 and between[0].isdigit():
                between = between[1:]
            ## Make Decision
            merge_bool.append(len(between) <= 1)
    ## Apply Merging
    merged_spans = []
    cur_start, cur_end = lbl_spans_s[0][:2]
    n_merged = 1
    for (lstart, lend, _), apply_merge in zip(lbl_spans_s[1:], merge_bool):
        if apply_merge:
            cur_end = lend
            n_merged += 1
        else:
            merged_spans.append((cur_start, cur_end, n_merged))
            cur_start, cur_end = lstart, lend
            n_merged = 1
    if len(merged_spans) == 0 or merged_spans[-1] != (cur_start, cur_end, n_merged):
        merged_spans.append((cur_start, cur_end, n_merged))
    ## Filter (No Merging allowed - Merged and Short Tend to Be Very Rare)
    merged_spans = [i for i in merged_spans if i[-1] == 1]
    ## Format
    lbl_spans_merged = [[s, e, text[s:e]] for s,e,_ in merged_spans]
    return lbl_spans_merged

def _merge_spans(spans,
                 prefer_large=True):
    """
    
    """
    ## Base Case
    if len(spans) <= 1:
        return spans
    ## Sort the Spans
    spans = sorted(spans, key=lambda x: x[0])
    ## Cache
    spans_merged = {}
    span_set = [spans[0]]
    ## Merge Overlapping Spans
    i = 1
    lower, upper = spans[0]
    while i < len(spans):
        ## Get Span
        slow, supp = spans[i]
        ## Case 1: New Span Outside Range
        if slow >= upper:
            spans_merged[(lower, upper)] = span_set
            lower, upper = slow, supp
            span_set = []
        ## Case 2: New Upper Bound
        elif slow >= lower and supp > upper:
            upper = supp
        ## Update Counter
        i += 1
        ## Update Span Set
        span_set.append((slow, supp))
    ## End
    spans_merged[(lower, upper)] = span_set
    ## Format
    if prefer_large:
        spans_merged = list(spans_merged.keys())
    else:
        min_span = lambda x: sorted(x, key=lambda v: (v[1]-v[0], -v[1]))[0] ## Smallest Length, Higher Span End
        spans_merged = list(map(min_span, spans_merged.values()))
    return spans_merged

def _resolve_retinopathy_hierarchy(labels):
    """
    Only keep the most specific retinopathy extraction
    """
    ## Whether To Ignore The Generic (Presence of a More Specific)
    ignore_dr = "A1 - DR (Generic)" in labels and ("A2 - NPDR" in labels or "A3 - PDR" in labels)
    ## Resolve
    labels_resolved = {}
    for lk, lv in labels.items():
        if lk == "A1 - DR (Generic)" and ignore_dr:
            continue
        labels_resolved[lk] = lv
    return labels_resolved

def _extract_document_annotations(document,
                                  autolabeler,
                                  formatted=True,
                                  handle_surgical=None,
                                  handle_anti_vegf=True,
                                  resolve_retinopathy_hierarchy=False):
    """
    
    """
    ## Check Input
    assert isinstance(document, dict)
    assert "text" in document
    assert "metadata" in document and "document_id" in document["metadata"]
    ## Automatic Labels
    autolabel_extractions = autolabeler.get_auto_labels(document["text"],
                                                        formatted=formatted)
    ## Document Cache
    document_annotations = {}
    ## Iterate Over Spans
    for label in autolabel_extractions:
        ## Check Type
        if not isinstance(label, list):
            raise TypeError("Expected all labels to be of type list.")
        ## Check for Label in Cache
        if label[-1] not in document_annotations:
            document_annotations[label[-1]] = []
        ## Split List Labels for Surgical Procedures
        labels_split = [label]
        if handle_surgical is not None and handle_surgical == "split" and label[-1] == 'E1 - Retina Surgery':
            spans_split = _split_surgical_label_span(document["text"][label[0]:label[1]], label[0], label[1])
            labels_split = [[s, e, label[-1]] for s, e in spans_split]
        ## Cache
        for ls in labels_split:
            document_annotations[ls[-1]].append([ls[0], ls[1], document["text"][ls[0]:ls[1]]])
    ## Surgical Label Merging
    if handle_surgical is not None and handle_surgical == "merge" and "E1 - Retina Surgery" in document_annotations:
        document_annotations["E1 - Retina Surgery"] = _merge_surgical_label_span(document["text"],
                                                                                 document_annotations["E1 - Retina Surgery"])
    ## Anti VEGF Merging
    if handle_anti_vegf and "D1 - Anti-VEGF" in document_annotations:
        document_annotations["D1 - Anti-VEGF"] = _merge_anti_vegf_label_span(document["text"],
                                                                             document_annotations["D1 - Anti-VEGF"])
    ## Remove Empty Concepts
    for lbl in list(document_annotations.keys()):
        if len(document_annotations[lbl]) == 0:
            _ = document_annotations.pop(lbl, None)
    ## Resolve DR Hierarchy
    if resolve_retinopathy_hierarchy:
        document_annotations = _resolve_retinopathy_hierarchy(document_annotations)
    ## Cache
    output = {"document_id":document["metadata"]["document_id"],
              "text":document["text"],
              "annotations":document_annotations}
    return output

def get_autolabels(text,
                   autolabeler,
                   formatted=True,
                   handle_surgical="merge",
                   handle_anti_vegf=True,
                   resolve_retinopathy_hierarchy=False):
    """
    
    """
    result = _extract_document_annotations(document={"text":text,"metadata":{"document_id":None}},
                                           autolabeler=autolabeler,
                                           formatted=formatted,
                                           handle_surgical=handle_surgical,
                                           handle_anti_vegf=handle_anti_vegf,
                                           resolve_retinopathy_hierarchy=resolve_retinopathy_hierarchy)
    return result

def _ldc_extractor(document,
                   autolabeler,
                   handle_surgical,
                   handle_anti_vegf,
                   formatted=True):
    """
    Helper function used by load_document_collection. Should not be called directly.
    """
    ## Split Input
    index, document = document
    ## Get Result
    result = _extract_document_annotations(document,
                                           autolabeler=autolabeler,
                                           handle_surgical=handle_surgical,
                                           handle_anti_vegf=handle_anti_vegf,
                                           formatted=formatted)
    ## Return Index and Result
    return index, result

def load_document_collection(filename,
                             autolabeler,
                             handle_surgical=None,
                             handle_anti_vegf=False,
                             sample_rate=None,
                             random_state=42,
                             verbose=False,
                             jobs=1):
    """
    
    """
    ## Check File Existence
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found: '{}'".format(filename))
    ## Check File Type
    if filename.endswith(".csv"):
        data = _load_formatted_csv(filename)
    elif filename.endswith(".json"):
        data = _load_formatted_json(filename)
    else:
        raise NotImplementedError("Filetype not recognized. Expected .csv or .json")
    ## Filter Null
    data["documents"] = list(filter(lambda d: d["text"] is not None and isinstance(d["text"], str), data["documents"]))
    ## Downsampling
    sample_seed = np.random.RandomState(random_state)
    if sample_rate is not None:
        ## Number of samples
        if sample_rate >= 1:
            n_sample = int(sample_rate)
        else:
            n_sample = int(np.floor(len(data["documents"]) * sample_rate))
        ## Validate
        if n_sample > len(data["documents"]):
            raise ValueError("Too many samples requested for size of document cache.")
        ## Get Indices
        sample_inds = set(sample_seed.choice(len(data["documents"]), n_sample, replace=False))
    else:
        sample_inds = set(range(len(data["documents"])))
    ## Filter Document
    data["documents"] = [doc for d, doc in enumerate(data["documents"]) if d in sample_inds]
    ## Parameterize Extractor
    extractor = partial(_ldc_extractor,
                        autolabeler=autolabeler,
                        handle_surgical=handle_surgical,
                        handle_anti_vegf=handle_anti_vegf,
                        formatted=True)
    ## Format Extractor Iterator
    iterator = enumerate(data["documents"]) if not verbose else tqdm(enumerate(data["documents"]), total=len(data["documents"]), file=sys.stdout, desc="[Extracting Annotations]")
    ## Run Extractor
    if jobs > 1:
        with Pool(jobs) as mp:
            annotations = list(mp.imap_unordered(extractor, iterator))
    else:
        annotations = list(map(extractor, iterator))
    ## Re-Sort
    annotations = sorted(annotations, key=lambda x: x[0])
    annotations = [a[1] for a in annotations]
    ## Return
    return annotations, data["labels"]

def load_annotated_worksheet(filename):
    """
    
    """
    ## Load File
    df = pd.read_excel(filename, header=None, engine="openpyxl")
    ## Cache
    df_parsed = []
    ## Initialize Temp Vars
    doc_id = None
    doc_text = None
    doc_cols = None
    doc_df = []
    ## Iterate Over Rows
    for r, row in df.iterrows():
        if row.isnull().all() or (r == df.shape[0] - 1):
            ## Final Row
            if (row.values[0] != "document_id") and not pd.isnull(row.values[1:]).all() and not (~(row.isnull())).all():
                doc_df.append(row.tolist())
            ## Format
            doc_df = pd.DataFrame(doc_df, columns=doc_cols)
            doc_df["document_id"] = doc_id
            doc_df["text"] = doc_text
            ## Cache
            df_parsed.append(doc_df)
            ## Reset
            doc_id = None
            doc_text = None
            doc_cols = None
            doc_df = []
            ## Move Ahead
            continue
        ## Parse Row Type
        if row.values[0] == "document_id":
            ## Handle Edge case (notes in the expected blank space)
            if doc_id is not None:
                ## Format
                doc_df = pd.DataFrame(doc_df, columns=doc_cols)
                doc_df["document_id"] = doc_id
                doc_df["text"] = doc_text
                ## Cache
                df_parsed.append(doc_df)
                ## Reset
                doc_id = None
                doc_text = None
                doc_cols = None
                doc_df = []
            ## Start New Document
            doc_id = row.values[1]
        elif row.values[0] == "encounter_date":
            continue
        elif pd.isnull(row.values[1:]).all():
            doc_text = row.values[0]
        elif (~(row.isnull())).all():
            doc_cols = row.tolist()
        else:
            doc_df.append(row.tolist())
    ## Concatenate and Format
    df_parsed = pd.concat(df_parsed, axis=0, sort=False)
    df_parsed = df_parsed[["document_id","text"] + [c for c in df_parsed.columns if c not in ["document_id","text"]]].copy()
    df_parsed = df_parsed.reset_index(drop=True)
    return df_parsed
