
"""
Align predicted outputs from apply.py script with original input format
to ensure splits are consistent across semi-supervised learning iterations.
"""

###########################
### Imports
###########################

## Standard Library
import os
import json
import argparse

###########################
### Functions
###########################

def parse_command_line():
    """

    """
    ## Parser
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--reference_data", nargs="*", default=None, type=str)
    _ = parser.add_argument("--alignment_data", nargs="*", default=None, type=str)
    _ = parser.add_argument("--output_dir", type=str, default=None)
    _ = parser.add_argument("--rm_existing", action="store_true", default=False)
    args = parser.parse_args()
    ## Check for Data
    has_ref, has_ali = True, True
    if args.reference_data is None or len(args.reference_data) == 0:
        print(">> WARNING - No reference data found.")
        has_ref = False
    if args.alignment_data is None or len(args.alignment_data) == 0:
        print(">> WARNING - No alignment data found.")
        has_ali = False
    if not (has_ref or has_ali):
        raise ValueError("Must provide at least one of --reference_data or --alignment_data")
    ## Check for Output
    if args.output_dir is None:
        raise ValueError("Must provide an --output_dir")
    if os.path.exists(args.output_dir) and not args.rm_existing:
        raise FileExistsError("Include --rm_existing flag to overwrite output.")
    ## Return
    return args

def _load_json(filename):
    """

    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Couldn't find file: '{filename}'")
    data = []
    with open(filename, "r") as the_file:
        for line in the_file:
            data.append(json.loads(line))
    return data
    
def load_json_data(filenames):
    """

    """
    data = []
    for filename in filenames:
        data.extend(_load_json(filename))
    return data

def cache_json_data(data, filename):
    """

    """
    with open(filename,"w") as the_file:
        for instance in data:
            the_file.write(json.dumps(instance)+"\n")

def main():
    """

    """
    ## Parse Command Line
    print("[Parsing Command Line]")
    args = parse_command_line()
    ## Output Directory
    print("[Initializing Output Directory]")
    if os.path.exists(args.output_dir) and args.rm_existing:
        _ = os.system(f"rm -rf {args.output_dir}")
    elif os.path.exists(args.output_dir) and not args.rm_existing:
        raise FileExistsError("Include --rm_existing flag to overwrite output.")
    if not os.path.exists(args.output_dir):
        _ = os.makedirs(args.output_dir)
    ## Cache Config
    print("[Storing Config]")
    with open(f"{args.output_dir}/cfg.json","w") as the_file:
        json.dump(vars(args), the_file, indent=1)
    ## Load
    print("[Loading Data]")
    reference_data = load_json_data(args.reference_data) if args.reference_data is not None and len(args.reference_data) > 0 else None
    alignment_data = load_json_data(args.alignment_data) if args.alignment_data is not None and len(args.alignment_data) > 0 else None
    ## Ordering
    if reference_data is not None and alignment_data is not None:
        print("[Ordering Data]")
        ## Get Ordering
        reference_document_order = [i["document_id"] for i in reference_data]
        reference_document_order = dict(zip(reference_document_order, range(len(reference_document_order))))
        alignment_document_order = [i["document_id"] for i in alignment_data]
        alignment_document_order = dict(zip(alignment_document_order, range(len(alignment_document_order))))
        ## Ensure All Reference Found in the Alignment (Okay to Have Alignment not in Reference)
        assert all(r in alignment_document_order for r in reference_document_order)
        ## Sort Alignment Data
        alignment_data = sorted(alignment_data, key=lambda d: reference_document_order[d["document_id"]])
        ## Verify
        assert [i["document_id"] for i in alignment_data] == [i["document_id"] for i in reference_data]
    ## Cache
    if reference_data is not None:
        print("[Caching Reference Data]")
        _ = cache_json_data(reference_data, f"{args.output_dir}/annotations.reference.json")
    if alignment_data is not None:
        print("[Caching Aligned Data]")
        _ = cache_json_data(alignment_data, f"{args.output_dir}/annotations.aligned.json")
    ## Done
    print("[Script Complete]")

############################
### Execute
############################

if __name__ == "__main__":
    _ = main()
