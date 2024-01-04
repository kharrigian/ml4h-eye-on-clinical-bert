
"""

"""

#####################
### Imports
#####################

## Standard Library
import re

#####################
### Functions
#####################

def _parse_formatted_problem_list(problem_list):
    """
    
    """
    ## Get Problems
    mspans = []
    for m in re.finditer("\\n\[\[.*\]\]\\n", problem_list):
        mspans.append(m.span())
    if len(mspans) == 0:
        return None
    mspans.append((len(problem_list), len(problem_list)))
    ## Group
    problems = {}
    for i, mspan in enumerate(mspans[:-1]):
        ## Split
        mname = problem_list[mspan[0]:mspan[1]]
        mtext = problem_list[mspan[1]:mspans[i+1][0]]
        ## Parse
        if "[OVERVIEW]" in mtext and "[ASSESSMENT & PLAN]" in mtext:
            overview, plan = mtext.split("[OVERVIEW]")[1].split("[ASSESSMENT & PLAN]")
        elif "[OVERVIEW]" in mtext:
            overview, plan = mtext.split("[OVERVIEW]")[1], None
        elif "[ASSESSMENT & PLAN]" in mtext:
            overview, plan = None, mtext.split("[ASSESSMENT & PLAN]")[1]
        ## Format
        overview = overview.strip() if overview is not None and len(overview.strip()) > 0 else None
        plan = plan.strip() if plan is not None and len(plan.strip()) >0 is not None else None
        ## Cache
        problems[mname.strip()[2:-2]] = {"overview":overview, "assessment_and_plan":plan}
    return problems

def _parse_formatted_codes(codes):
    """
    
    """
    codes = codes.strip()
    codes = [i.strip() for i in codes.strip().split("\n")]
    codes_fmt = []
    for c in codes:
        codes_fmt.append(c[2:-2].split(": ", 1))
    codes_fmt = [{x:y} for x, y in codes_fmt]
    return codes_fmt

def parse_formatted_note_text(text):
    """
    
    """
    ## Validate Type
    if not isinstance(text, str):
        return None
    ## Header Locations
    code_span = re.search(r"\[\[\[ENCOUNTER ICD-10 CODES\]\]\]\n", text, flags=re.UNICODE)
    prog_span = re.search(r"\[\[\[PROGRESS NOTE\]\]\]\n", text, flags=re.UNICODE)
    prob_span = re.search(r"\[\[\[PROBLEM LIST\]\]\]\n", text, flags=re.UNICODE)
    ## Parse
    codes, progress_note, problem_list = None, None, None
    if code_span:
        codes = text[0:min([len(text)] + [i.start() for i in [prog_span, prob_span] if i])]
        codes = codes.replace("[[[ENCOUNTER ICD-10 CODES]]]","")
    if prog_span:
        progress_note = text[prog_span.start():min([len(text)] + [i.start() for i in [prob_span] if i])]
        progress_note = progress_note.replace("[[[PROGRESS NOTE]]]","")
    if prob_span:
        problem_list = text[prob_span.start():]
        problem_list = problem_list.replace("[[[PROBLEM LIST]]]","")
    ## Validate
    if all(i is None for i in [codes, progress_note, problem_list]):
        return None
    ## Format Codes
    if codes is not None:
        codes = _parse_formatted_codes(codes)
    ## Parse Problem List
    if problem_list is not None:
        problem_list = _parse_formatted_problem_list(problem_list)
    ## Format Progress Note
    if progress_note is not None:
        progress_note = progress_note.strip()
    ## Check
    if codes is None and progress_note is None and (problem_list is None or all(y.get("overview") is None and y.get("assessment_and_plan") is None for x, y in problem_list.items())):
        return None
    ## Final Return
    return {"progress":progress_note, "problem_list":problem_list, "diagnostic_codes":codes}