
##########################################
### Imports
##########################################

from collections import Counter

##########################################
### Annotation
##########################################

## Generic Labels
LABELS_LATERALITY = ["OS",
                     "OD",
                     "OU"]
LABELS_TEMPORALITY_PROCEDURE = ["History of",
                                "Performed Today",
                                "Recommended Today",
                                "Considering"]

## Map of Labels
CONCEPT_ATTRIBUTE_MAP = {
    "A1 - DR (Generic)":{
        "Laterality":LABELS_LATERALITY,
        "Status": ["History of","Active"],
    },
    "A2 - NPDR":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "Mild",
            "Mild-Moderate",
            "Moderate",
            "Moderate-Severe",
            "Severe",
            "Very Severe"
        ],
        "Status": ["History of","Active"],
    },
    "A3 - PDR":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "HR",
            "NHR"
        ],
        "Status": ["History of","Active"]
    },
    "A4 - NV":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "Iris",
            "Iris + NVD or NVE",
            "NVD",
            "NVE",
            "NVD/NVE",
            "AMD",
            "Other"
        ],
        "Status": ["Active","Resolved"]
    },
    "B1 - ME":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "DME",
            "CI-DME",
            "Non-CI-DME",
            "CS-DME",
            "Non-CS-DME",
            "CME",
            "AMD",
            "Other"
        ],
        "Status": ["History of","Active"]
    },
    "C1 - VH":{
        "Laterality":LABELS_LATERALITY,
        "Status": ["History of","Active","Resolving","Resolved"] ## last three refer to acute episode
    },
    "C2 - RD":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "RRD",
            "TRD",
            "Combined RRD/TRD",
            "Serous"
        ],
        "Status": ["History of","Active"]
    },
    "C3 - NVG":{
        "Laterality":LABELS_LATERALITY,
        "Status": ["Not Present","Present"]
    },
    "D1 - Anti-VEGF":{
        "Laterality":LABELS_LATERALITY,
        "Status":LABELS_TEMPORALITY_PROCEDURE
    },
    "D2 - PRP":{
        "Laterality":LABELS_LATERALITY,
        "Status":LABELS_TEMPORALITY_PROCEDURE
    },
    "D3 - Focal Grid Laser":{
        "Laterality":LABELS_LATERALITY,
        "Status":LABELS_TEMPORALITY_PROCEDURE
    },
    "D4 - Intravitreal Injections (Other)":{
        "Laterality":LABELS_LATERALITY,
        "Status": LABELS_TEMPORALITY_PROCEDURE
    },
    "E1 - Retina Surgery":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "Indication VH",
            "Indication RD"
        ],
        "Status":LABELS_TEMPORALITY_PROCEDURE
    },
    "E2 - NVG Surgery":{
        "Laterality":LABELS_LATERALITY,
        "Severity/Type":[
            "Tube",
            "Trab",
            "MIGS"
        ],
        "Status": LABELS_TEMPORALITY_PROCEDURE
    },
    "F1 - Diabetes Mellitus":{
        "Severity/Type":[
            "Type 1",
            "Type 2",
            "Gestational",
            "Other"
        ],
        "Status": ["Active","Resolved"]
    },
    "G1 - Nephropathy":{
        "Status": ["Not Present","Present"]
    },
    "G2 - Neuropathy":{
        "Status": ["Not Present","Present"]
    },
    "G3 - Heart Attack":{
        "Status": ["No History of", "History of"]
    },
    "G4 - Stroke":{
        "Status": ["No History of", "History of"]
    },
    "X1 - Emerging Entity":{
        "Severity/Type":[
            "Corporation",
            "Creative-Work",
            "Group",
            "Location",
            "Person",
            "Product"
        ]
    }
}

## Set of Labels Where Negation Should Not Be Allowed (Binary Status)
CONCEPT_BINARY_STATUS = set([
    "C3 - NVG",
    "G1 - Nephropathy",
    "G2 - Neuropathy",
    "G3 - Heart Attack",
    "G4 - Stroke",
    "X1 - Emerging Entity"
])

##########################################
### Classification Setup
##########################################

## Consolidated Mapping
CLASSIFIER_TASK_MAP = {
    "named_entity_recognition":{
        "field":"valid",
        "classes":{
            False:[False],
            True:[True]
        },
        "concepts":["A1","A2","A3","A4","B1","C1","C2","C3","D1","D2","D3","D4","E1","E2","F1","G1","G2","G3","G4","X1"]
    },
    "laterality":{
        "field":"laterality",
        "classes":{
            "OS":["OS"],
            "OD":["OD"],
            "OU":["OU"],
            "Not Specified":["Not Specified"]
        },
        "concepts":['A1','A2','A3','A4','B1','C1','C2','C3','D1','D2','D3','D4','E1','E2']
    },
    "diagnostic_status_eye":{
        "field":"status",
        "classes":{
            "Present":["Active","Negated, Resolved","Resolving","Present"],
            "Not Present":["Negated, Active","History of","Negated, History of","Resolved","Not Present"],
            "Not Specified":["Not Specified"],
            "Negated, Not Specified":["Negated, Not Specified"],
        },
        "concepts":["A1","A2","A3","A4","B1","C1","C2","C3"]
    },
    "diagnostic_status_dm":{
        "field":"status",
        "classes":{
            "Active":["Active","Negated, Resolved"],
            "Inactive":["Negated, Active","Resolved"],
            "Not Specified":["Not Specified"],
            "Negated, Not Specified":["Negated, Not Specified"],
        },
        "concepts":["F1"]
    },
    "diagnostic_status_nep_neur_heart_stroke":{
        "field":"status",
        "classes":{
            "Present":["Present","History of","Active","Negated, Resolved"],
            "Not Present":["Not Present","No History of","Negated, Active","Resolved"],
            "Not Specified":["Not Specified"],
            "Negated, Not Specified":["Negated, Not Specified"],
        },
        "concepts":["G1","G2","G3","G4"]
    },
    "procedure_status_all":{
        "field":"status",
        "classes":{
            "History of":["History of"],
            "Performed Today":["Performed Today"],
            "No Action":["Negated, History of", "Negated, Performed Today","Recommended Today","Considering","Negated, Recommended Today", "Negated, Considering"],
            "Not Specified":["Not Specified"],
            "Negated, Not Specified":["Negated, Not Specified"]
        },
        "concepts":["D1","D2","D3","D4","E1","E2"]
    },
    "npdr_severity":{
        "field":"severity_type",
        "classes":{
            'Mild':["Mild"],
            "Moderate":["Mild-Moderate","Moderate"],
            "Severe":["Moderate-Severe","Severe","Very Severe"],
            'Not Specified':["Not Specified"]
        },
        "concepts":["A2"]
    },
    "pdr_severity":{
        "field":"severity_type",
        "classes":{
            "HR":["HR"],
            "NHR":["NHR"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["A3"]
    },
    "nv_type":{
        "field":"severity_type",
        "classes":{
            "NVD and/or NVE":["NVD","NVE","NVD/NVE"],
            "Iris + NVD and/or NVE":["Iris + NVD or NVE"],
            "Iris":["Iris"],
            "AMD":["AMD"],
            "Other":["Other"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["A4"]
    },
    "me_type":{
        "field":"severity_type",
        "classes":{
            "DME":["DME","CI-DME","Non-CI-DME","CS-DME","Non-CS-DME"],
            "Other":["CME","AMD","Other"],
            "Not Specified":["Not Specified"],
        },
        "concepts":["B1"]
    },
    "rd_type":{
        "field":"severity_type",
        "classes":{
            'RRD':["RRD"],
            'TRD':["TRD"],
            'Combined RRD/TRD':["Combined RRD/TRD"],
            'Serous':["Serous"],
            'Not Specified':["Not Specified"]
        },
        "concepts":["C2"]
    },
    "retina_surgery_type":{
        "field":"severity_type",
        "classes":{
            "Indication VH":["Indication VH"],
            "Indication RD":["Indication RD"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["E1"]
    },
    "nvg_surgery_type":{
        "field":"severity_type",
        "classes":{
            "Tube":["Tube"],
            "Trab":["Trab"],
            "MIGS":["MIGS"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["E2"]
    },
    "dm_type":{
        "field":"severity_type",
        "classes":{
            "Type 1":["Type 1"],
            "Type 2":["Type 2"],
            "Other":["Gestational","Other"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["F1"]
    },
    "wnut":{
        "field":"severity_type",
        "classes":{
            "Corporation":["Corporation"],
            "Creative-Work":["Creative-Work"],
            "Group":["Group"],
            "Location":["Location"],
            "Person":["Person"],
            "Product":["Product"],
            "Not Specified":["Not Specified"]
        },
        "concepts":["X1"]
    }
}

########################
### Helper Functions
########################

def create_concept_id_to_name_map():
    """
    
    """
    ## Name Map (ID to Name)
    lbl2full = {}
    for lbl_full, _ in CONCEPT_ATTRIBUTE_MAP.items():
        lbl_prefix = lbl_full.split(" - ")[0]
        lbl2full[lbl_prefix] = lbl_full
    return lbl2full

def create_negation_map(separate_negation):
    """

    """
    if separate_negation is None:
        return None
    if len(separate_negation) == 1 and separate_negation[0] == "all":
        concepts = create_concept_id_to_name_map()
        negation_groups_r = {x:"all" for x in concepts.keys()}
    else:
        negation_groups = {i:i.split(",") for i in separate_negation}
        negation_groups_r = {}
        for x, y in negation_groups.items():
            for j in y:
                if j in negation_groups_r:
                    raise ValueError("Duplicate negation groups")
                negation_groups_r[j] = x
    return negation_groups_r
    
def create_task_map(task_map,
                    separate_negation=None):
    """
    
    """
    ## Concept ID to Name Map
    concept_id_2_name = create_concept_id_to_name_map()
    ## Negation Map
    negation_map = create_negation_map(separate_negation)
    ## Task Map
    lbl2task = {}
    for task_name, task_items in task_map.items():
        for concept_id in task_items["concepts"]:
            concept_full = concept_id_2_name[concept_id]
            if concept_full not in lbl2task:
                lbl2task[concept_full] = set()
            lbl2task[concept_full].add((task_name, task_items["field"]))
            if task_items["field"] == "status" and negation_map is not None and concept_id in negation_map:
                lbl2task[concept_full].add(("negation_{}".format(negation_map[concept_id]), "negation"))
    ## Return
    return lbl2task

def create_attribute_rename_map(task_map,
                                ignore_non_specified=False):
    """
    
    """
    ## Concept ID to Name Map
    concept_id_2_name = create_concept_id_to_name_map()
    ## Map
    rename_map = {}
    for _, task_items in task_map.items():
        task_field = task_items["field"]
        task_concepts = task_items["concepts"]
        for concept in task_concepts:
            concept_full = concept_id_2_name[concept]
            for remap_label, remap_ats in task_items["classes"].items():
                if ignore_non_specified and isinstance(remap_label, str) and "Not Specified" in remap_label:
                    continue
                for at_lbl in remap_ats:
                    if ignore_non_specified and isinstance(at_lbl, str) and "Not Specified" in at_lbl:
                        continue
                    remap_key = (concept_full, task_field, at_lbl)
                    if remap_key in rename_map:
                        raise ValueError("There appears to be a one-to-many case for the proposed label remapping")
                    rename_map[remap_key] = remap_label
    return rename_map

def rename_labels(lbl,
                  attr_label_rename_map,
                  ignore_non_specified=False,
                  separate_negation=None):
    """
    
    """
    ## Make Update for Each Attribute Type
    for field in ["valid","laterality","status","severity_type"]:
        ## Check Relevance
        if field != "valid" and not lbl["valid"]:
            continue
        if lbl[field] == "N/A":
            continue
        ## Negation
        if separate_negation is not None and field == "status":
            lbl["negation"] = "Negated" in lbl[field]
        ## Case 0: Non-Specified
        if ignore_non_specified and field != "valid" and "Not Specified" in lbl[field]:
            lbl[field] = None
        ## Case 1: Allow Non-Specified or Other
        else:
            ## Make Change in Place
            lbl[field] = attr_label_rename_map[(lbl["label"], field, lbl[field])]
        ## Negation
        if separate_negation is not None and field == "status" and lbl[field] is not None:
            lbl[field] = lbl[field].replace("Negated,","").replace("Negated","").strip()
    return lbl

########################
### Classes
########################

class EncounterResolver(object):

    """

    """

    def __init__(self):
        """

        """
        ## Function Mapping
        self._fcn_map = {
            "OS":[
                self._resolve_a1_a2_a3,
                self._resolve_a4,
                self._resolve_b1,
                self._resolve_c1,
                self._resolve_c2,
                self._resolve_c3,
                self._resolve_d1_d2_d3_d4,
                self._resolve_e1,
                self._resolve_e2,
            ],
            "OD":[
                self._resolve_a1_a2_a3,
                self._resolve_a4,
                self._resolve_b1,
                self._resolve_c1,
                self._resolve_c2,
                self._resolve_c3,
                self._resolve_d1_d2_d3_d4,
                self._resolve_e1,
                self._resolve_e2,
            ],
            "Other":[
                self._resolve_f1,
                self._resolve_g1,
                self._resolve_g2,
                self._resolve_g3,
                self._resolve_g4
            ]
        }

    def split_laterality(self,
                         labels):
        """

        """
        labels_split = {"OS":[],"OD":[],"Other":[]}
        for lbl in labels:
            lbl_lat = {"OS":["OS"],"OD":["OD"],"OU":["OS","OD"]}.get(lbl.get("laterality"),["Other"])
            for lat in lbl_lat:
                ## Ignore Inferred Invalid Matches
                if not lbl["valid"]:
                    continue
                ## Cache Concept and Attributes
                labels_split[lat].append({
                    "concept":lbl["concept"],
                    "severity_type":lbl.get("severity_type"),
                    "status":lbl.get("status"),
                })
        ## Return
        return labels_split
    
    def _linear_resolve(self,
                        labels,
                        concept,
                        order,
                        order_key,
                        status=None,
                        severity_type=None):
        """

        """
        ## Isolate Concept Labels
        concept_labels = list(filter(lambda lbl: lbl["concept"]==concept, labels))
        ## Additional Concept Isolation
        if status is not None:
            concept_labels = list(filter(lambda lbl: lbl.get("status")==status, concept_labels))
        if severity_type is not None:
            concept_labels = list(filter(lambda lbl: lbl.get("severity_type")==severity_type, concept_labels))
        ## None Relevant
        if len(concept_labels) == 0:
            return []
        ## Initialize Prefix
        prefix = [concept]
        if status is not None:
            prefix = prefix + [status]
        if severity_type is not None:
            prefix = prefix + [severity_type]
        ## Translate to Tuple
        concept_labels = set([tuple([lbl[key] for key in order_key]) for lbl in concept_labels])
        ## Search
        for oname, otup in order:
            if otup in concept_labels:
                return [prefix + [oname]]
        ## Raise if No Matches
        raise ValueError("This shouldn't happen")

    def _resolve_a1_a2_a3(self,
                          labels):
        """

        """
        ## Initialize Resolution
        resolved = []
        ## Resolve NPDR
        resolved.extend(
            self._linear_resolve(labels,
                                 concept="A2 - NPDR" ,
                                 order=[
                                    ("Severe", ("Present", "Severe")),
                                    ("Moderate", ("Present", "Moderate")),
                                    ("Mild", ("Present", "Mild")),
                                    ("Not Present", ("Not Present", "Severe")),
                                    ("Not Present", ("Not Present", "Moderate")),
                                    ("Not Present", ("Not Present", "Mild"))
                                 ],
                                 order_key=["status","severity_type"],
                                 )
        )
        ## Resolve PDR
        resolved.extend(
            self._linear_resolve(labels,
                                 concept="A3 - PDR",
                                 order=[
                                    ("HR-PDR", ("Present", "HR")),
                                    ("NHR-PDR", ("Present", "NHR")),
                                    ("Not Present", ("Not Present", "HR")),
                                    ("Not Present", ("Not Present", "NHR"))
                                 ],
                                 order_key=["status","severity_type"]
                                 )
        )
        ## If Resolved Already, Return
        resolved = list(filter(lambda r: len(r) > 0, resolved))
        if len(resolved) > 0:
            return resolved
        ## General Resolution
        return self._linear_resolve(labels,
                                    concept="A1 - DR (Generic)",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present", ))
                                    ],
                                    order_key=["status"])
        

    
    def _resolve_a4(self,
                    labels):
        """

        """
        ## Get Concepts
        concept_labels = list(filter(lambda lbl: lbl["concept"]=="A4 - NV", labels))
        ## Initial Check
        if len(concept_labels) == 0:
            return []
        ## Merged Categories
        concept_labels = set((lbl["status"],lbl["severity_type"]) for lbl in concept_labels)
        if ("Present","NVD and/or NVE") in concept_labels and ("Present","Iris") in concept_labels:
            concept_labels.add(("Present","Iris + NVD and/or NVE"))
        ## Resolve
        for oname, otup in [
                ("Iris + NVD and/or NVE", ("Present", "Iris + NVD and/or NVE")),
                ("NVD and/or NVE", ("Present", "NVD and/or NVE")),
                ("Iris", ("Present", "Iris")),
                ("AMD", ("Present", "AMD")),
                ("Other", ("Present", "Other")),
                ("Not Present", ("Not Present", "Iris + NVD and/or NVE")),
                ("Not Present", ("Not Present", "NVD and/or NVE")),
                ("Not Present", ("Not Present", "Iris")),
                ("Not Present", ("Not Present", "AMD")),
                ("Not Present", ("Not Present", "Other"))
            ]:
            if otup in concept_labels:
                return [["A4 - NV", oname]]
        ## Raise
        raise ValueError("This shouldn't happen.")

    def _resolve_b1(self,
                    labels):
        """

        """
        resolved = []
        for severity_type in ["DME","Other"]:
            resolved.extend(
                self._linear_resolve(labels=labels,
                                     concept="B1 - ME",
                                     order=[
                                        ("Present", ("Present",severity_type)),
                                        ("Not Present", ("Not Present",severity_type))
                                     ],
                                     order_key=["status","severity_type"],
                                     severity_type=severity_type
                                    )
            )
        return resolved
    
    def _resolve_c1(self,
                    labels):
        """

        """
        return self._linear_resolve(labels=labels,
                                    concept="C1 - VH",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",)),
                                    ],
                                    order_key=["status"])    

    def _resolve_c2(self,
                    labels):
        """

        """
        ## Get Concept Labels
        concept_labels = list(filter(lambda lbl: lbl["concept"]=="C2 - RD", labels))
        ## No Matches
        if len(concept_labels) == 0:
            return []
        ## Add And Logic
        concept_labels = set((lbl["status"], lbl["severity_type"]) for lbl in concept_labels)
        if ("Present", "RRD") in concept_labels and ("Present", "TRD") in concept_labels:
            concept_labels.add(("Present", "Combined RRD/TRD"))
        ## Hierarchy Resolve
        for oname, otup in [
            ("Combined RRD/TRD", ("Present","Combined RRD/TRD")),
            ("RRD", ("Present", "RRD")),
            ("TRD", ("Present", "TRD")),
            ("Serous", ("Present", "Serous")),
            ("Not Present", ("Not Present", "Combined RRD/TRD")),
            ("Not Present", ("Not Present", "RRD")),
            ("Not Present", ("Not Present", "TRD")),
            ("Not Present", ("Not Present", "Serous")),
            ]:
            if otup in concept_labels:
                return [["C2 - RD", oname]]
        ## Error check
        raise ValueError("This shouldn't happen.")

    def _resolve_c3(self,
                    labels):
        """

        """
        return self._linear_resolve(labels=labels,
                                    concept="C3 - NVG",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",)),
                                    ],
                                    order_key=["status"])    
        
    def _resolve_d1_d2_d3_d4(self,
                             labels):
        """

        """
        resolved = []
        for concept in ["D1 - Anti-VEGF","D2 - PRP","D3 - Focal Grid Laser","D4 - Intravitreal Injections (Other)"]:
            resolved.extend(
                self._linear_resolve(labels,
                                     concept=concept,
                                     order=[
                                        ("Performed Today", ("Performed Today",)),
                                        ("History of", ("History of",)),
                                        ("No Action", ("No Action",))
                                     ],
                                     order_key=["status"])
            )
        return resolved

    def _resolve_e1(self,
                    labels):
        """

        """
        resolved = []
        for severity_type in ["Indication VH","Indication RD"]:
            resolved.extend(
                self._linear_resolve(labels,
                                     concept="E1 - Retina Surgery",
                                     order=[
                                        ("Performed Today", ("Performed Today",)),
                                        ("History of", ("History of",)),
                                        ("No Action", ("No Action",))
                                     ],
                                     order_key=["status"],
                                     severity_type=severity_type)
            )
        return resolved
    
    def _resolve_e2(self,
                    labels):
        """

        """
        resolved = []
        for severity_type in ["Tube","Trab","MIGS"]:
            resolved.extend(
                self._linear_resolve(labels,
                                     concept="E2 - NVG Surgery",
                                     order=[
                                        ("Performed Today", ("Performed Today",)),
                                        ("History of", ("History of",)),
                                        ("No Action", ("No Action",))
                                     ],
                                     order_key=["status"],
                                     severity_type=severity_type)
            )
        return resolved
    
    def _resolve_f1(self,
                    labels):
        """

        """
        ## Filter
        concept_labels = list(filter(lambda lbl: lbl["concept"]=="F1 - Diabetes Mellitus", labels))
        ## No Matches
        if len(concept_labels) == 0:
            return []
        ## Count Types
        severity_type_count = Counter(cl["severity_type"] for cl in concept_labels)
        ## Get Most Common
        if severity_type_count.get("Type 2",0) >= severity_type_count.get("Type 1",0):
            return [["F1 - Diabetes Mellitus","Type 2"]]
        else:
            return [["F1 - Diabetes Mellitus","Type 1"]]
        ## Format and REturn
        return [["F1 - Diabetes Mellitus", most_common]]
    
    def _resolve_g1(self,
                    labels):
        """

        """
        return self._linear_resolve(labels,
                                    concept="G1 - Nephropathy",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",))
                                    ],
                                    order_key=["status"])

    def _resolve_g2(self,
                    labels):
        """

        """
        return self._linear_resolve(labels,
                                    concept="G2 - Neuropathy",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",))
                                    ],
                                    order_key=["status"])

    def _resolve_g3(self,
                    labels):
        """

        """
        return self._linear_resolve(labels,
                                    concept="G3 - Heart Attack",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",))
                                    ],
                                    order_key=["status"])
                        
    def _resolve_g4(self,
                    labels):
        """

        """
        return self._linear_resolve(labels,
                                    concept="G4 - Stroke",
                                    order=[
                                        ("Present", ("Present",)),
                                        ("Not Present", ("Not Present",))
                                    ],
                                    order_key=["status"])

    def resolve(self,
                labels):
        """

        """
        ## Split Labels By (Non-)Laterality
        labels_split = self.split_laterality(labels)
        ## Apply Function Map
        labels_split_resolved = {}
        for src, src_labels in labels_split.items():
            labels_split_resolved[src] = []
            for fcn in self._fcn_map[src]:
                labels_split_resolved[src].extend(
                    fcn(labels=src_labels)
                )
            labels_split_resolved[src] = list(filter(lambda i: len(i) > 0, labels_split_resolved[src]))
        ## Return
        return labels_split_resolved