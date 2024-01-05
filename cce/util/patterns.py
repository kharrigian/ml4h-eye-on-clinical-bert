
"""
Helpful Regular Expression Groups
"""

##########################
### Imports
##########################

## Standard Library
import re

##########################
### Useful Expressions
##########################

## Problem Header
PHEADER = re.compile(r"\[\[(.*\:\s.*)\]\]\n", flags=re.UNICODE|re.IGNORECASE)

##########################
### ICD-10 Codes
##########################

## Concepts to Add Automatically for DM Codes (Key Represents Addition if No Match From Group)
ICD10_DM_AUTO_LABELS_GROUPS = {
    "A1 - DR (Generic)":set(["A1 - DR (Generic)","A2 - NPDR","A3 - PDR"]),
    "B1 - ME":set(["B1 - ME"]),
    "F1 - Diabetes Mellitus":set(["F1 - Diabetes Mellitus"]),
    "G1 - Nephropathy":set(["G1 - Nephropathy"]),
    "G2 - Neuropathy":set(["G2 - Neuropathy"]),
    "G3 - Heart Attack":set(["G3 - Heart Attack"]),
    "G4 - Stroke":set(["G4 - Stroke"])
}

## DM ICD Codes
ICD10_DM_AUTO_LABELS = {
    "A1 - DR (Generic)":[
        r"(E((0(8|9))|(1(0|1|3))))(\.)31([0-9]*)", ## E08.31x, E09.31x, E10.31x, E11.31x, E13.31x,
        r"(E((0(8|9))|(1(0|1|3))))(\.)9", ## E08.9, E09.9, E10.9, E11.9, E13.9
    ],
    'A2 - NPDR': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)3([2-4]+)([0-9]*)", ## e.g. E08.32, E08.321, E08.3311, E09.3421
    ],
    'A3 - PDR': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)35([0-9]*)", ## e.g. E08.35, E09.351
    ],
    'B1 - ME': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)3([1-9])(1|9)([1-3]|9)?", ## e.g. E08.3111, E08.2913, E08.3519
        r"(E((0(8|9))|(1(0|1|3))))(\.)37X([1-3]|9)", ## e.g. E08.37X1, E08.37X9
    ],
    'F1 - Diabetes Mellitus': [
        r"(E((0(8|9))|(1(0|1|3))))",
    ],
    'G1 - Nephropathy': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)2([1-2]|9)",
    ],
    'G2 - Neuropathy': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)4([0-4]|9)",
        r"(E((0(8|9))|(1(0|1|3))))(\.)61(0|8)",
    ]
}

## Relevant ICD 10 Codes
ICD10_AUTO_LABELS = {
    'A1 - DR (Generic)': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)31([0-9]*)", ## E08.31x, E09.31x, E10.31x, E11.31x, E13.31x
        r"(E((0(8|9))|(1(0|1|3))))(\.)9", ## E08.9, E09.9, E10.9, E11.9, E13.9
        ],
    'A2 - NPDR': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)3([2-4]+)([0-9]*)", ## e.g. E08.32, E08.321, E08.3311, E09.3421
        ],
    'A3 - PDR': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)35([0-9]*)", ## e.g. E08.35, E09.351
        r"H35\.2([0-9]*)", ## e.g. H35.2, H35.25
        ],
    'A4 - NV': [
        r"H21\.1(X)?([0-9]*)", ## e.g. H21.1X0, H21.1X9
        r"H35\.05([0-9]*)", ## e.g. H35.051, H35.059
        r"H35\.09", ## H35.09
        r"H35\.32([1-3|9])([1-2])", ## e.g. H35.3211, H35.3292
        r"H44\.2A([1-3]|9)?", ## e.g. H44.2A9, H44.2A3
        ],
    'B1 - ME': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)3([1-9])(1|9)([1-3]|9)?", ## e.g. E08.3111, E08.2913, E08.3519
        r"(E((0(8|9))|(1(0|1|3))))(\.)37X([1-3]|9)", ## e.g. E08.37X1, E08.37X9
        r"H34\.8(1|3)([1-3]|9)0", ## e.g. H34.8110, H34.8390, H34.8330
        r"H35\.81", ## H35.81
        r"H35\.35([1-3]|9)?", ## e.g. H35.35, H35.353
        r"H35\.71([1-3]|9)?", ## e.g. H35.71, H35.713
        r"H59\.03([1-3]|9)?", ## e.g. H59.03, H59.039
        ],
    'C1 - VH': [
        r"H35\.6([0-3]?)", ## e.g. H35.60, H35.6
        r"H43\.1([0-3]?)", ## e.g. H43.1, H43.11
        ],
    'C2 - RD': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)35([2-4])([1-3]|9)",
        r"H33(\.)?([0-9]*)",
        r"H59\.81([1-3]|9)?",
        ],
    'C3 - NVG': [
        r"H40(\.)?([0-9]*)X?([0-9]*)",
        r"H42(\.)?([0-9]*)X?([0-9]*)",
        ],
    'D1 - Anti-VEGF': [],
    'D2 - PRP': [],
    'D3 - Focal Grid Laser': [],
    'D4 - Intravitreal Injections (Other)': [],
    'E1 - Retina Surgery': [
        ],
    'E2 - NVG Surgery': [
        r"Z98\.83",
        ],
    'F1 - Diabetes Mellitus': [
        r"(E((0(8|9))|(1(0|1|3))))",
        r"E16\.([1-2])",
        r"O24\.([0-1]|[3-4]|[8-9])(1|3)([1-3]|9)?",
        r"R73\.(0)?9",
        r"Z13\.1",
        r"Z83\.3",
        r"Z86\.3(2|9)",
        ],
    'G1 - Nephropathy': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)2([1-2]|9)",
        r"N17(\.)?([0-9]*)",
        r"N18(\.)?([0-9]*)", 
        r"N19(\.)?([0-9]*)",
        r"M32\.15",
        r"M35\.04",
        r"N0(0|[3-5])\.([0-9]+)",
        r"N02\.8",
        r"R80(\.)?([0-9]*)",
        ],
    'G2 - Neuropathy': [
        r"(E((0(8|9))|(1(0|1|3))))(\.)4([0-4]|9)",
        r"(E((0(8|9))|(1(0|1|3))))(\.)61(0|8)",
        r"B02\.2([0-9]*)",
        r"G5([0-2]|[6-7])(\.)?([0-9]*)",
        r"G60(\.)?([0-9]*)",
        r"G62(\.)?([0-9]*)",
        r"G73\.3",
        r"G90\.(0|8|9)(1|9)?",
        r"H49(\.)?([0-9]*)",
        r"H81(\.)?([0-9]*)",
        r"I95\.1",
        r"K31\.84",
        r"K59\.1",
        r"M14\.6([0-9]+)",
        r"M54\.8(1|9)",
        r"M79\.2",
        r"N31\.9",
        r"S04\.([0-9]+)",
        ],
    'G3 - Heart Attack': [
        r"I11\.(0|9)",
        r"I13\.([0-2])",
        r"I2([1-3])\.((A(1|9))|[0-4]|[8-9])",
        r"I25\.2",
        r"I4([6-9])\.([0-9]+)",
        r"I50(\.)?([0-9]*)",
        r"I70\.2(5|6[0-9]+)",
        r"I71\.([0-9]+)",
        ],
    'G4 - Stroke': [
        r"G43\.5([0-1])(1|9)?",
        r"G45(\.)?([0-9]*)",
        r"I61\.9",
        r"I6(3|[5-6])\.([0-9]+)",
        r"I65(\.)?([0-9]*)",
        r"I69\.3([0-9]+)",
        r"Z86\.73",
        ]
}

##########################
### Free Text Extraction
##########################

## Retina Surgery Regular Expression Construction
_SURGICAL_PROCEDURES = [
    r"((2(5|3)(\-)?g(\s|\-)?)?(\b(ppv)\b|pars(\s+)plana(\s+)victrectomy)((\-|[ ])?2(5|3)(\-)?g)?)",
    r"(((2(5|3)-gauge)(\s+))?pars((\s|\-)+)plana((\s|\-)+))?(vitrectomy)((\s+)with(\s+)removal(\s+)of(\s+)perfluoro\-n\-octane)?",
    r"((air|fluid|gas)((\s|\-)+)(air|fluid|gas)((\s|\-)+)exchange(d|s)?)",
    r"(((extensive|internal|limiting|((epi|pre|sub)(\-)?retinal))(\s+)(and(\s+))?)*(\b(membrane|ilm|irm|erm)\b)((\s|\-)+)peel(s|ing)?)",
    r"(intraocular(\s+)lens(\s+)(placement|insertion|removal))",
    r"(endolaser)((\s+)photocoagulation)?",
    r"(perfluoro\-n\-octane)",
    r"(((temporary(\s+))?(injection|extrusion)((\s+)and(\s+))?)*(\s+)of(\s+))?(perfluoron)",
    r"((air|gas|oil)(\s+)(\+|and|vs(\.)?)(\s+)(air|gas|oil))",
    r"(autolog(ous|enous)(\s+))?(retina(l?)((\-|\s)+)(surgery|transplant))",
    r"(\b(victrectomy)\b)",
    r"(\b(scleral(\s+)buckle)\b)",
    r"((pars(\s+)plana(\s+))?lensectomy)",
    r"((dislocated(\s+))?IOL(\s+)removal)",
    r"(iris(\s+)hooks((\s+)use)?)",
    r"(\b(tamponade)\b)",
    r"(anterior(\s+)chamber(\s+)washout)",
    r"(insertion(\s+)of(\s+)so)",
    r"(\b(so(\s+)insertion)\b)",
    r"((injection(\s+)of(\s+))?(\d+\%(\s)?)?c3f8(\s?\d+\%)?((\s+)gas)?)",
    r"((si|si(l+)icon(e)?)((\s|\-)+)oil((\s|\-)+)(removal|drainage|insertion))",
    r"(\b(si(o)?(\s+)oil)\b)",
    r"(anterior(\s+)synechialysis)",
    r"(posterior(\s+)hyloid(\s+)removal)",
    r"(dissection(\s+)of(\s+)(erm|DM|fibrovascular(\s+)(tissue|membrane(s)?)))",
    r"(((completion of )?((\d+((\s|\-)+)degree|small|superior|membrane|preretinal|inferior|temporal|relaxing|drain(age|ing))(\s+))?)*(retinectomy|retinotomy)(\s+\d+(\s|\-)+degrees)?)",
    r"((endo)?diathermy)",
    r"(perfluoron(\s+)injection)",
    r"((removal|drainage)(\s+)of(\s+)(subretinal|intraretinal|interstitial|preretinal)(\s+)(fluid|membrane))",
    r"((injection(\s+)of(\s+))?silicone(\s+)oil((\s+)?tamponade)?)",
    r"(((\d+)\%(\s+)?)?sf6((\s+)(gas|tamponade))*)",
    r"(retinal(\s+)(graft|(re(\-?))?attachment))",
    r"((?<!focal(\s)grid(\s))(?<!grid(\s))(?<!scatter(\s))(?<!focal(\s))(?<!panretinal(\s))(?<!pan\-retinal(\s))(indirect(\s))?(laser((\s+)?(retino)?pexy)?))",
    r"(\b(((t|r)?)rd(\s+)repair)\b)",
]
_SURGICAL_PROCEDURES = r"({})(((\,)?(\s+)(and|combined(\s+)with|with)(\s+)|\,(\s+)|\/|(\s+)\+(\s+))({}))*".format("|".join(_SURGICAL_PROCEDURES[:-1]), 
                                                                                                                                                      "|".join(_SURGICAL_PROCEDURES))

## Diabetes Smartphrases
_DM_SMART = [
    r"(importance(\s+)of(\s+))((((continue(d)?(\s+))?(proper(ly)?|continu(ed|e|ing)?|good|optimal(ly)?|tight(ly)?|strict(ly)?)|manag(e(d?)|ing)|improv(e|ing)|optimiz(e|ing)|maintain(ing)?|keep(ing)|monitor(ing)?|control(ling)?|lower(ing)?)(\s+))+)?((hb(\-?)a1c|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?|pressure)((\s+)(control(led)?|(well((\-|\s)+))?maintain(ed)?|maintenance|manage(d|ment)))?)(((\,)?(\s+)((and(\s+))|as(\s+)well(\s+)as(\s+))|\,(\s+))((\b(hb(\-?)a1c)\b|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?|pressure))((\s)(manage(d|ment)|maintenance|(well((\-|\s)+))?maintain(ed)?|control(led)?))?)*",
    r"((keep)(\s+))(\b(hb(\-?)a1c)\b|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?(\s+level(s)?)?)(((\,)?(\s+)((and(\s+))|as(\s+)well(\s+)as(\s+))|\,(\s+)|\s+)?(\b(hb(\-?)a1c)\b|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?)(\s+level(s)?)?)*((\s+)(control(led)?|low|(well((\-|\s)+))?maintain(ed)?|manage(d|ment)|maintenance))",
    r"((((keep(ing)?|continu(ed|ing|e))(\s+))?(proper(ly)?|continu(ed|e|ing)?|good|optimal(ly)?|tight(ly)?|strict(ly)?)|manag(e(d?)|ing)|improv(e|ing)|optimiz(e|ing)|maintain(ing)?|keep(ing)|control(ling)?|monitor(ing)?|lower(ing)?)(\s+))?(((hb(\-?)a1c|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?)(\s?)((as(\s+)well(\s+)as|and)(\s+))?)+(control(led)?|(well((\-|\s)+))?maintain(ed)?|maintenance|manage(d|ment)))+",
    r"((((keep(ing)?|continu(ed|ing|e))(\s+))?(proper(ly)?|continu(ed|e|ing)?|good|optimal(ly)?|tight(ly)?|strict(ly)?)|manag(e(d?)|ing)|improv(e|ing)|optimiz(e|ing)|maintain(ing)?|keep(ing)|monitor(ing)?|control(ling)?|lower(ing)?)(\s+))+(\b(hb(\-?)a1c)\b|glycemic|hypertensive|\b(bg|bp|dm)\b|(blood(\s+)(sugar|glucose|lipid(s?)|pressure))|cholesterol|lipid(s)?)((\s+)(control(led)?|(well((\-|\s)+))?maintain(ed)?|manage(d|ment)|maintenance))?"
]
_DM_SMART = r"\b" + "|".join(_DM_SMART) + r"\b"

## Automatic Labels ## {label[str] : [(pattern [str], case sensitive[bool] )]}
AUTO_LABELS = {
    "A1 - DR (Generic)":[
        (r"(?<!myopic(\s))(?<!induced(\s))(?<!sickle(\s)cell(\s))(?<!exudative\s)(?<!autoimmune\s)(?<!hypertensive\s)(?<!radiation\s)(?<!arteriosclerotic\s)(?<!outer\s)(?<!pigmentary\s)(?<!cellophane\s)(?<!purtscher\'s\s)(?<!valsalva\s)(?<!rubella\s)(?<!solar\s)(?<!serous\s)(?<!central\s)(?<!hypertensive\s)(?<!proliferative\s)(?<!proliferative\sdiabetic\s)((diabetic((\-|\s)+))?(\b)(vitreo)?retinopathy)(([ ]+)(due([ ]+)to|associated([ ]+)with)([ ]+)diabetes(([ ]+)mellitus)?)?(?![ ](associated|due)[ ](with|to)[ ](tamoxifen|(acquired(\s+))?immunodeficiency))(?!(\,)?([ ]+)((non(\-)?)?proliferative|hypertensive))(?!([ ]+)(of([ ]+))?prematurity)", False),
        (r"(?<!proliferative\sdiabetic\s)(?<!proliferative\s)(retinopathy(?!(\,)?([ ]+)(non(\-)?)?proliferative)([ ]+)(due([ ]+)to([ ]+)|associated([ ]+)with([ ]+))diabetes(([ ])mellitus)?)", False),
        (r"(diabetic(\s+)eye(\s+)disease)", False),
        (r"(?<!PROLIFERATIVE(\s))(?<!proliferative(\s))\b(DR)\b(?!(\,?)(\s+)((NON|(n|N)on)?)(PROLIFERATIVE|(p|P)roliferative))", True),
        (r"\b(IRMA)\b", True),
        (r"(diabetes(\s+)and(\s+)((no(\s+))?((significant|center((\-|\s)+)involved|mild|mild((\-|\s)+)moderate|moderate|moderate((\-|\s)+)severe|severe)\s+)?)?retinopathy)", False),
    ],
    "A2 - NPDR":[
        (r"(non(\s|\-)?proliferative(((\s|\-)+)diabetic)?((\s|\-)+)(vitreo)?retinopathy)", False),
        (r"(non(\s|\-)?proliferative(\s)dr\b)", False),
        (r"(diabetic(\s+)(vitreo)?retinopathy(\,)(\s+)non(\-)?proliferative)", False),
        (r"(\b(NPDR)\b)", False),
    ],
    "A3 - PDR":[
        (r"(?<!non)(?<!non\s)(?<!non\-)(proliferative(((\s|\-)+)diabetic)?(\s+)(vitreo)?retinopathy)", False),
        (r"(?<!non)(?<!non\s)(?<!non\-)(proliferative((\s|\-)+)disease)", False),
        (r"(?<!non)(?<!non\s)(?<!non\-)(proliferative(\s+)dr\b)", False),
        (r"(diabetic(\s+)(vitreo)?retinopathy\,(\s+)((?<!non)(\b)proliferative))", False),
        (r"\b((N?HR)(\-|\s)?)?(PDR)\b", False),
    ],
    "A4 - NV":[
        (r"(?<!ocular(\s))(?<!corneal(\s))\b((iris|retina(l)?)(\s+))?(neo(\-)?vascularization)((\s+)of(\s+)the(\s+)(disc|retina((\s+)elsewhere)?))?\b", False),
        (r"(?<!ocular(\s))(?<!corneal(\s))\b(neo(\-)?vascular)((\s+)changes(((\s+)or)?(\s+)(of|at|in)(\s+)the(\s+)(retina|disc))*)\b", False),
        (r"(?<!ocular(\s))(?<!corneal(\s))(neo(\-)?vascular)(?!\s+glaucoma)", False),
        (r"\b(nvd\/e|nvi\/nva|nv|nvi|nve|nvd|fvp)\b", False),
        (r"\b(((abnormal|new)((\s|\-)+))(blood((\s|\-)+))?vessel(s?))\b", False),
        (r"(pre(\-)?retinal((\s|\-)+)fibrovascular((\s|\-)+)membrane)", False),
        (r"(fibrous((\s|\-)+)(proliferan|remnant)(s)?)", False),
        (r"\b(rubeosis((\s+)iridis)?)\b",False),
    ],
    "B1 - ME":[
        (r"((cystoid|diabetic)((\s|\-)+))?(macular((\s|\-)+)edema)((\s+)(caused(\s+)by|associated(\s+)with|due(\s+)to)(\s+)diabetes)?", False),
        (r"\b((d|(ci|cs)(\s+|\-)?|c)me)\b", False),
        (r"\b(ME)\b", True),
        (r"fovea(l?)((\s|\-)+)dry", False),
        (r"(?<!chronic(\s))(?<!localized(\s))(?<!dependent(\s))(?<!peripheral(\s))(?<!arm(\s))(?<!facial(\s))(?<!papilla(\s))(?<!nerve(\s))(?<!vasogenic(\s))(?<!cerebral(\s))(?<!disk(\s))(?<!disc(\s))(?<!cornea(\s))(?<!corneal(\s))(?<!eyelid(\s))(?<!orbital(\s))(?<!conjunctival(\s))\b(foveal(\s+))?(edema|exudation)\b(?!\s+eyelid)(?!(\s+)(of).{0,50}(extremity|optic|eyelid|cornea|orbit(s)?))", False),
        (r"\b(minimal(\s+))?(srf|irf)\b", False),
        (r"(?<!removal(\s)of(\s))(?<!drainage(\s)of(\s))(?<!air(\-))(?<!air(\s))(?<!air)(((sub|intra|inter)(\-|\s+)?(retinal|stitial)((\s|\-)+))fluid)(?!(\s)(removal|drainage))", False),
    ],
    "C1 - VH":[
        (r"((vitreous((\s|\-)+))h(a?)emorrhage(s)?)", False),
        (r"((macular|((pre|sub)(\-)?)retina(l?))((\s|\-)+)hemorrhage(s)?)", False),
        (r"(subhyaloid|vitreous)((\s|\-)+)(heme|hemorrhage(s)?)", False),
        (r"\b(VH|PRH)\b",False)
    ],
    "C2 - RD":[
        (r"(retinal(\s|\-)(horseshoe(\s+))?((traction(al)?)(\s|\-)+)?(hole(s)?|break(s)?|detachment(s)?|separation(s)?|tear(s?)))",False),
        (r"(detachment|separation|tear(s)?)(\s+)of(\s+).{0,20}(\b(retina)\b)",False),
        (r"(retina(l)?(.){0,30}(with(out)?(\s)detachment))", False),
        (r"((detach(ed)?|separate(d)?)(\s+)retina)", False),
        (r"(retina(\s+)is(\s+)(attached|separated|torn))",False),
        (r"\b(T|R)?(RD)\b(?!(\s+)repair)",False),
    ],
    "C3 - NVG":[
        (r"((neo(\-)?vascular(\b)|(secondary)((\s+|\-)?(open|angle|closure))*)(\s+))(glaucoma)(?!(\s+)(due(\s+))?to(\s+)(hyphema|drug(s)?|(eye(\s))?(trauma|inflammation)))",False),
        (r"\b(glaucoma)(\,)?((\s+)of(\s+)(the(\s+))?(left|right|both)(\s+)eye(s)?)?(\s+)(neovascular|secondary((\s+|\-)?(open|angle|closure))*)(?!(\s+)(due(\s+))?to(\s+)(hyphema|drug(s)?|(eye(\s))?(trauma|inflammation)))", False),
        (r"\b(NVG)\b",False),
    ],
    "D1 - Anti-VEGF":[
        (r"\b(lucentis|eyl(e?)a|avastin|aflibercept|ramivimizab)\b",False),
        (r"\b(intra(\-)?vitreal((\s|\-)+)injection(s)?(\s+)of(\s+))?(anti(\-|\s)+)?(v((\s+)?)e((\s+)?)g((\s+)?)f)\b",False),
        (r"(anti(\-|\s)+)?(vascular((\-|\s)+)endothelial((\-|\s)+)growth((\-|\s)+)factor(((\-|\s)+)therapy)?)",False),
        (r"(intra(\-)?vitreal((\s|\-)+)injection(s)?)",False),
    ],
    "D2 - PRP":[
        (r"(pan((\-|\s)+)?retinal((\s|\-)+)(\S)?photocoagulation)",False),
        (r"\b(PRP)\b",False),
        (r"(scatter(\s+)laser)\b",False),
        (r"(laser(\s+)photocoagulation)", False)
    ],
    "D3 - Focal Grid Laser":[
        (r"(focal((\-|\s)+)(grid((\-|\s)+))?laser)((\s+)(photocoagulation|treatment))?",False),
    ],
    "D4 - Intravitreal Injections (Other)":[
        (r"\b(ozurdex|kenalog)\b",False),
    ],
    "E1 - Retina Surgery":[
        (_SURGICAL_PROCEDURES, False),
        (r"\b(EL|PPL|PFO|SOR|SB|SO|MP|AFX|AFx|FAx)\b(?!(.){0,20}amaurosis)", True),
    ],
    "E2 - NVG Surgery":[
      (r"(trabeculectomy|\b(migs)\b|\b(tube)\b)",False),
      (r"(aqueous((\-|\s)+))?(shunt(s?))",False),
      (r"(((minimal(ly)?((\-|\s)+)invasive)|incisional)((\-|\s)+))?(glaucoma((\-|\s)+)surgery)",False),
      (r"(glaucoma((\-|\s)+)drainage)((\s+)implant(\s+)insertion)?",False),
    ],
    "F1 - Diabetes Mellitus":[
        (r"\b(t1dm|t2dm|dka|(hb)?a1c)\b",False),
        (r"((?<!dissection(\s))\b(DM)\b)",False),
        (r"(?<!pre\-)(?<!borderline\s)(\b(diabetes)(((\s|\-)+)mellitus)?)(?!(\s+)insipidus)", False),
        (r"(?<!non(\-))(?<!pre(\-))(?<!pre(\s))(?<!borderline(\s))(\b(diabetic))", False),
        (r"(?<!fasting(\s))\b((hyper|hypo)(glycemia|insulinism))\b",False),
        (_DM_SMART, False),
        (r"(?<!\,\s)(?<!exudative(\s)vitreoretinopathy(\s))(?<!albinism\s)(?<!syndrome\s)(?<!imperfecta\s)(?<!malformation\,\s)(?<!malformation\s)(?<!telangiectasia\s)(?<!hyperlipidemia\s)(?<!managed(\s)as(\s))(?<!neurofibromatosis(\s))(?<!dystrophy\s)(?<!aniridia\s)(?<!aniridia\,\s)(?<!\b(dm)(\s))(?<!diabetes(\,\s))(?<!diabetes(\s))(?<!mellitus(\,\s))(?<!mellitus(\s))(?<!borderline(\s))(\b(t)|type)((\s|\-)*)(\b1\b|\b2\b|\b(i)\b|\b(ii)\b)(?!(\s|\-)*(dm|diabetes))(?!(\s)duane)(?!(\s)posterior)(?!(\s)macular)", False)
    ],
    "G1 - Nephropathy":[
        (r"\b((poly)?nephropathy|proteinuria)\b",False),
    ],
    "G2 - Neuropathy":[
        (r"(?<!optic\s)\b(neuropath(ic|y)|neuritis|neuralgia)\b",False),
    ],
    "G3 - Heart Attack":[
        (r"\b(heart((\s|\-)+)(attack|failure))\b",False),
        (r"\b(cardiac((\s|\-)+)(arrest))\b",False),
        (r"\b((myocardial|coronary)((\s|\-)+)infarct(ion)?)\b",False),
        (r"\b(MI)\b",False),
        (r"\b(conjestive(\s+)heart(\s+)failure)\b", False),
    ],
    "G4 - Stroke":[
        (r"\b(CVA)\b",False),
        (r"(?<!pituitary(\s))\b(stroke(s)?|apoplexy)\b(?!(\s+)syndrome)",False),
        (r"\b((cerebrovascular|cerebellar|cerebral)(\s+)(vascular(\s+))?(infarct(ion)?|accident|event))\b",False),
    ]
}
