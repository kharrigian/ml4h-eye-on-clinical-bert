
"""
Transform a completed worksheet into a file of formatted annotations for model input.
"""

######################
### Imports
######################

## Standard Library
import os
import sys
import json
import argparse

## External
import pandas as pd

## Local
from cce.util.labels import CONCEPT_ATTRIBUTE_MAP, CONCEPT_BINARY_STATUS
from cce.util.data_loaders import load_annotated_worksheet
from cce.util.patterns import PHEADER

#######################
### Functions
#######################

def parse_command_line():
    """
    
    """
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--input_file", type=str, default=None)
    _ = parser.add_argument("--input_file_src", type=str, default=None)
    _ = parser.add_argument("--input_file_metadata", type=str, default=None)
    _ = parser.add_argument("--output_dir", type=str, default=None)
    _ = parser.add_argument("--rm_existing", action="store_true")
    args = parser.parse_args()
    return args

def validate_paths(args):
    """
    
    """
    if args.input_file is None or not os.path.exists(args.input_file):
        raise FileNotFoundError("Input file not found (--input_file)")
    if args.input_file_src is None or not os.path.exists(args.input_file_src):
        raise FileNotFoundError("Input file source extractions not found (--input_file_src)")
    if args.output_dir is None:
        raise ValueError("Must provide an --output_dir")
    if os.path.exists(args.output_dir) and not args.rm_existing:
        raise FileExistsError("Include --rm_existing flag to remove existing output directory.")
    elif os.path.exists(args.output_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.output_dir}")
    _ = os.makedirs(args.output_dir)

def rename_invalid(main_label,
                   sublabels,
                   sublabels_cols):
    """
    Check labels against accepted set to indicate if anything is Invalid for the
    main concept

    Args:
        main_label (str): Concept / Entity
        sublabels (tuple): Attributes assigned to the concept
        sublabels_cols (list): Type of attribute associated with each element of the sublabels tuple
    """
    ## Initialize New Output
    renamed = []
    ## Iterate Through Label Dimensions
    for (lcv, lc) in zip(sublabels, sublabels_cols):
        ## Negation (Null or Negated)
        if lc == "Negation":
            if not (pd.isnull(lcv) or lcv == "NEGATED" or lcv == "XXX"):
                lcv = "INVALID"
        ## Skipped
        elif lc == "Skipped":
            if not (pd.isnull(lcv) or lcv == "SKIPPED"):
                lcv = "INVALID"
        ## Incorrect (Null or Incorrect)
        elif lc == "Incorrect Span":
            if not (pd.isnull(lcv) or lcv == "INCORRECT"):
                lcv = "INVALID"
        ## Label Exists Where it Shouldn't
        elif lc not in CONCEPT_ATTRIBUTE_MAP[main_label] and lcv != "XXX":
            lcv = "INVALID"
        ## Label From A Different Category
        elif lc in CONCEPT_ATTRIBUTE_MAP[main_label] and (lcv not in set(CONCEPT_ATTRIBUTE_MAP[main_label][lc]) and not pd.isnull(lcv)):
            lcv = "INVALID"
        renamed.append(lcv)
    ## Format and Return
    renamed = tuple(renamed)
    return renamed

def get_value(sublabels,
              fields,
              sublabels_cols_2_ind):
    """
    Extract tuple for a specified set of values

    Args:
        sublabels (tuple): Attributes assigned to an instance
        fields (list): Which attributes to extract
        sublabels_cols_2_ind (dict): Map between attribute name and index in the sublabels tuple
    """
    ## Get Values for Each Field
    values = [sublabels[sublabels_cols_2_ind[f]] for f in fields]
    ## Negation Handling (Double-Negatives)
    if "Negation" in fields and fields == ["Negation","Status"]:
        if values[0] == "NEGATED" and values[1] in ["No History of","Not Present"]:
            values[0] = None
        if values[0] == "NEGATED":
            values[0] = "Negated"
        elif pd.isnull(values[0]):
            values[0] = ""
    elif "Negation" in fields and fields == ["Negation"]:
        if values[0] == "NEGATED":
            values[0] = "Negated"
        elif pd.isnull(values[0]):
            values[0] = "Present"
    elif "Negation" in fields and not (fields == ["Negation"] or fields == ["Negation","Status"]):
        raise ValueError("Unexpected inputs for Negation")
    ## Invalid If Any Subfields Are Invalid
    values = ["Invalid"] if any(v == "INVALID" for v in values) else values
    ## Null Values Considered "Not Specified"
    values = list(map(lambda v: "Not Specified" if pd.isnull(v) else v, values))    
    ## Non-Applicable Examples (All Fields Not Relevant)
    values = list(map(lambda v: "N/A" if v == "XXX" else v, values))
    if all(v == "N/A" for v in values):
        return ("N/A",)
    ## Filter Non-applicable (mainly case with non-negation binary status)
    values = list(filter(lambda i: i != "N/A", values))
    ## If Example Labeled as Incorrect, Ignore Other Labels Entirely
    if sublabels[sublabels_cols_2_ind["Incorrect Span"]] == "INCORRECT":
        return ("Incorrect", )
    if "Skipped" in sublabels_cols_2_ind and sublabels[sublabels_cols_2_ind["Skipped"]] == "SKIPPED":
        return ("Skipped", )
    ## Ignore Empty Strings
    values = list(filter(lambda i: len(i) > 0, values))
    ## Return
    return tuple(values)

def span_in_header(span_start,
                   span_end,
                   header_spans):
    """
    
    """
    for dh_start, dh_end in header_spans:
        if span_start >= dh_start and span_end <= dh_end:
            return True
    return False

def load_worksheet_metadata(args):
    """
    
    """
    if args.input_file_metadata is None:
        return None
    if not os.path.exists(args.input_file_metadata):
        raise FileNotFoundError("Could not find expected --input_file_metadata")
    input_annotations_metadata = pd.read_csv(args.input_file_metadata)
    if "document_id" not in input_annotations_metadata.columns:
        raise KeyError("Missing Document ID in Metadata")
    if input_annotations_metadata["document_id"].duplicated().any():
        raise ValueError("Document IDs are not unique.")
    input_annotations_metadata["document_id"] = input_annotations_metadata["document_id"].astype(str)
    input_annotations_metadata = input_annotations_metadata.set_index("document_id").to_dict(orient="index")
    return input_annotations_metadata

def load_worksheet_source(args,
                          input_annotations,
                          input_annotations_metadata):
    """
    
    """
    input_annotations_src = {}
    input_annotations_ids = set(input_annotations["document_id"].unique())
    with open(args.input_file_src, "r") as the_file:
        for line in the_file:
            ## Format Line
            line_data = json.loads(line)
            line_data["document_id"] = str(line_data["document_id"])
            ## Check Relevance
            if line_data["document_id"] not in input_annotations_ids:
                continue
            ## Extract Problem Headers
            line_data["headers"] = [i.span() for i in PHEADER.finditer(line_data["text"])]
            ## Optional: Add Metadata
            line_data["metadata"] = {}
            if input_annotations_metadata is not None:
                line_data["metadata"] = input_annotations_metadata.get(line_data["document_id"], {})
            ## Cache
            input_annotations_src[line_data["document_id"]] = line_data
    ## Check
    print("[Checking Annotations]")
    if len(input_annotations_ids & set(input_annotations_src.keys())) != len(input_annotations_ids):
        raise ValueError("Missing at least one document_id from source file.")
    ## Return
    return input_annotations_src

def standardize_worksheet_annotations(input_annotations,
                                      input_annotations_src):
    """
    
    """
    ## Annotation Formatting
    if "Skipped" in input_annotations.columns:
        input_annotations = input_annotations.loc[input_annotations["Skipped"].isnull(),:]
        input_annotations = input_annotations.reset_index(drop=True)
    ## Standardize Expanded Labels (Included due to loose concept connection)
    input_annotations["in_autolabel_postprocess"] = input_annotations["label"].map(lambda i: "<<AUTO>>" in i)
    input_annotations["label"] = input_annotations["label"].map(lambda i: i.replace(" <<AUTO>>",""))
    ## Remove Invalid Labels
    lbl_cols_expected = ["Laterality","Severity/Type","Status","Negation","Incorrect Span","Skipped"]
    lbl_cols_expected = list(filter(lambda i: i in input_annotations.columns, lbl_cols_expected))
    lbl_cols_2_ind = dict(zip(lbl_cols_expected, range(len(lbl_cols_expected))))   
    ## Make Tuple
    input_annotations["attribute_tuple"] = input_annotations[lbl_cols_expected].apply(tuple, axis=1)
    ## Handle and Invalid Labels
    input_annotations["attribute_tuple"] = input_annotations.apply(lambda row: rename_invalid(row["label"], sublabels=row["attribute_tuple"], sublabels_cols=lbl_cols_expected), axis=1)
    ## Format Labels
    input_annotations["laterality"] = input_annotations["attribute_tuple"].map(lambda x: get_value(x, ["Laterality"], lbl_cols_2_ind)).map(", ".join)
    input_annotations["severity_type"] = input_annotations["attribute_tuple"].map(lambda x: get_value(x, ["Severity/Type"], lbl_cols_2_ind)).map(", ".join)
    input_annotations["status"] = input_annotations["attribute_tuple"].map(lambda x: get_value(x, ["Negation","Status"], lbl_cols_2_ind)).map(", ".join)
    ## Span in Header
    input_annotations["in_header"] = input_annotations.apply(lambda row: span_in_header(row["text_span_start"], row["text_span_end"], input_annotations_src[row["document_id"]]["headers"]), axis=1)
    ## Remove Unnecessary Information
    input_annotations = input_annotations.loc[:,["document_id","text_span_start","text_span_end","label","laterality","severity_type","status","in_autolabel_postprocess","in_header"]]
    ## Return
    return input_annotations

def transform_worksheet_annotations(input_annotations,
                                    input_annotations_src):
    """
    
    """
    input_annotations_fmt = {}
    for _, row in input_annotations.iterrows():
        ## Document Initialization (If Necessary)
        if row["document_id"] not in input_annotations_fmt:
            input_annotations_fmt[row["document_id"]] = {
                    "text":input_annotations_src[row["document_id"]]["text"],
                    "metadata":input_annotations_src[row["document_id"]]["metadata"],
                    "labels":[]
            }
        ## Is the Row Valid?
        row_valid = not any(row[i] == "Incorrect" for i in ["laterality","severity_type","status"])
        ## Label Information
        row_label = {
            "label":row["label"],
            "start":row["text_span_start"],
            "end":row["text_span_end"],
            "in_header":row["in_header"],
            "in_autolabel_postprocess":row["in_autolabel_postprocess"],
            "valid":row_valid,
            "laterality":row["laterality"] if row_valid and not pd.isnull(row["laterality"]) else None,
            "severity_type":row["severity_type"] if row_valid and not pd.isnull(row["severity_type"]) else None,
            "status":row["status"] if row_valid and not pd.isnull(row["status"]) else None
        }
        ## Store
        input_annotations_fmt[row["document_id"]]["labels"].append(row_label)
    ## Re-Format
    input_annotations_fmt = [{"document_id":x, "text":y["text"], "labels":y["labels"], "metadata":y["metadata"]} for x, y in input_annotations_fmt.items()]
    ## Return
    return input_annotations_fmt

def main():
    """
    
    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Check Paths (Input and Output)
    print("[Validating Paths]")
    _ = validate_paths(args)
    ## Load Annotations
    print("[Loading Annotations]")
    input_annotations = load_annotated_worksheet(filename=args.input_file)
    input_annotations["document_id"] = input_annotations["document_id"].astype(str)
    ## Metadata
    print("[Loading Metadata (If Provided)]")
    input_annotations_metadata = load_worksheet_metadata(args=args)
    ## Load Source
    print("[Loading Annotations Source]")
    input_annotations_src = load_worksheet_source(args=args,
                                                  input_annotations=input_annotations,
                                                  input_annotations_metadata=input_annotations_metadata)
    ## Format
    print("[Standardizing Annotations]")
    input_annotations = standardize_worksheet_annotations(input_annotations=input_annotations,
                                                          input_annotations_src=input_annotations_src)
    ## Merge Agreements and Source
    print("[Transforming Annotations]")
    input_annotations = transform_worksheet_annotations(input_annotations=input_annotations,
                                                        input_annotations_src=input_annotations_src)
    ## Cache
    print("[Caching Formatting Annotations]")
    with open(f"{args.output_dir}/annnotations.formatted.json","w") as the_file:
        for example in input_annotations:
            the_file.write(f"{json.dumps(example)}\n")
    ## Done
    print("[Script Complete]")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()