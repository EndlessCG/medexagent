"""
Tool schemas and utilities for medical exam tool calls.

Provides OpenAI function-calling schemas for all exam types,
parsing of exam strings from the dataset, and findings formatting.
"""

import re
import json


# ---------------------------------------------------------------------------
# OpenAI function-calling tool schemas for all exam categories
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "xray",
            "description": "Order an X-ray imaging study.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region to image (e.g. chest, abdomen, knee)"}
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ct",
            "description": "Order a CT (computed tomography) scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region to scan"},
                    "contrast": {"type": "string", "description": "Contrast usage (with contrast, no, yes)"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mri",
            "description": "Order an MRI (magnetic resonance imaging) scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region to scan"},
                    "contrast": {"type": "string", "description": "Contrast usage"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ultrasound",
            "description": "Order an ultrasound examination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region to examine"}
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blood_test",
            "description": "Order a blood test / laboratory panel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific test name"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cbc",
            "description": "Order a complete blood count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Specific CBC variant (e.g. peripheral smear, differential)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bmp",
            "description": "Order a basic metabolic panel.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cmp",
            "description": "Order a comprehensive metabolic panel.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lipid_panel",
            "description": "Order a lipid panel.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "liver_function_test",
            "description": "Order liver function tests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "renal_function_test",
            "description": "Order renal function tests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thyroid_function_test",
            "description": "Order thyroid function tests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "coagulation_panel",
            "description": "Order coagulation studies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Specific coagulation test"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "iron_studies",
            "description": "Order iron studies.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inflammatory_marker",
            "description": "Order inflammatory markers (CRP, ESR, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string", "description": "Specific marker name"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cardiac_enzyme",
            "description": "Order cardiac enzyme / biomarker tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string", "description": "Specific cardiac marker"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tumor_marker",
            "description": "Order tumor marker tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "marker": {"type": "string", "description": "Specific tumor marker"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hormone_level",
            "description": "Order hormone level tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hormone": {"type": "string", "description": "Hormone to measure"}
                },
                "required": ["hormone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drug_level",
            "description": "Order drug level / therapeutic drug monitoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {"type": "string", "description": "Drug to measure"}
                },
                "required": ["drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blood_gas",
            "description": "Order arterial or venous blood gas analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "arterial or venous"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ecg",
            "description": "Order an electrocardiogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "ECG type (e.g. standard 12-lead, surface)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echocardiogram",
            "description": "Order an echocardiogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Echo type (transthoracic/TTE, transesophageal/TEE, etc.)"},
                    "contrast": {"type": "string", "description": "Whether contrast is used"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echocardiography",
            "description": "Order echocardiography (alias).",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Echo type"},
                    "region": {"type": "string", "description": "Region"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "holter_monitor",
            "description": "Order Holter / ambulatory ECG monitoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Monitoring duration (e.g. 24h)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stress_test",
            "description": "Order a cardiac or orthopedic stress test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Stress test type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cardiac_catheterization",
            "description": "Order cardiac catheterization / coronary angiography.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Type (left heart, right heart, both, coronary angiography)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eeg",
            "description": "Order an electroencephalogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "EEG type (e.g. video, quantitative)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emg",
            "description": "Order electromyography / nerve conduction studies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region / nerves to test"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evoked_potentials",
            "description": "Order evoked potential studies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Type (visual, auditory, somatosensory)"},
                    "site": {"type": "string", "description": "Site"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pulmonary_function_test",
            "description": "Order pulmonary function tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Specific PFT type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "endoscopy",
            "description": "Order an endoscopic procedure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Endoscopy type (e.g. upper GI, colonoscopy, bronchoscopy)"},
                    "region": {"type": "string", "description": "Body region"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "biopsy",
            "description": "Order a tissue biopsy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {"type": "string", "description": "Biopsy site"},
                    "type": {"type": "string", "description": "Biopsy type (e.g. core, fine needle, excisional)"},
                    "method": {"type": "string", "description": "Guidance method (e.g. CT-guided)"},
                },
                "required": ["site"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bone_marrow_biopsy",
            "description": "Order a bone marrow biopsy / aspiration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Type (aspiration, biopsy, both)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lumbar_puncture",
            "description": "Order a lumbar puncture / CSF analysis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csf_analysis",
            "description": "Order cerebrospinal fluid analysis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arthrocentesis",
            "description": "Order joint aspiration / arthrocentesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint": {"type": "string", "description": "Joint to aspirate"}
                },
                "required": ["joint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paracentesis",
            "description": "Order abdominal paracentesis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thoracentesis",
            "description": "Order thoracentesis / pleural tap.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "amniocentesis",
            "description": "Order amniocentesis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "culture",
            "description": "Order a microbiological culture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Culture source / specimen type"}
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "serology",
            "description": "Order serological tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific serological test"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pcr_test",
            "description": "Order a PCR-based test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "PCR target (pathogen, gene, etc.)"}
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "viral_load",
            "description": "Order a viral load test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Viral load test type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genetic_test",
            "description": "Order a genetic / molecular genetic test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific genetic test"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "molecular_test",
            "description": "Order a molecular test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific molecular test"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "histopathology",
            "description": "Order histopathological examination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specimen": {"type": "string", "description": "Specimen to examine"}
                },
                "required": ["specimen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "immunohistochemistry",
            "description": "Order immunohistochemistry staining.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markers": {"type": "string", "description": "Markers to test"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cytology",
            "description": "Order cytological examination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {"type": "string", "description": "Site / specimen"},
                    "type": {"type": "string", "description": "Cytology type"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flow_cytometry",
            "description": "Order flow cytometry analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specimen": {"type": "string", "description": "Specimen type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "urinalysis",
            "description": "Order a urinalysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Urinalysis type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "urine_test",
            "description": "Order a urine test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific urine test"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "urine_cytology",
            "description": "Order urine cytology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Cytology type"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stool_test",
            "description": "Order a stool test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test": {"type": "string", "description": "Specific stool test"}
                },
                "required": ["test"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pet_scan",
            "description": "Order a PET (positron emission tomography) scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Body region"},
                    "tracer": {"type": "string", "description": "Radiotracer (e.g. FDG)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nuclear_scan",
            "description": "Order a nuclear medicine scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Scan type (e.g. bone scan, thyroid scan)"},
                    "region": {"type": "string", "description": "Body region"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mammography",
            "description": "Order a mammogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Mammography type (screening, diagnostic)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dexa_scan",
            "description": "Order a DEXA bone density scan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Region to scan"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fluoroscopy",
            "description": "Order a fluoroscopic study.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Fluoroscopy type / study name"}
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "angiography",
            "description": "Order an angiography study.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Vascular region"},
                    "type": {"type": "string", "description": "Angiography type (e.g. digital subtraction, CT angiography)"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eye_exam",
            "description": "Order an ophthalmological examination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Exam type (e.g. fundoscopy, slit-lamp, OCT)"},
                    "region": {"type": "string", "description": "Eye region"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audiometry",
            "description": "Order an audiometric / hearing evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Audiometry variant",
                        "enum": ["pure_tone", "tympanometry", "oae", "abr", "full_battery"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "physical_examination",
            "description": "Order a focused physical examination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Examination type"},
                    "region": {"type": "string", "description": "Body region to examine"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "allergy_test",
            "description": "Order allergy testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Test type (skin prick, RAST, specific IgE)"},
                    "allergens": {"type": "string", "description": "Allergens to test"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skin_test",
            "description": "Order a skin test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "Test type (patch testing, tuberculin, etc.)"},
                    "allergens": {"type": "string", "description": "Allergens / substances to test"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rheumatoid_factor",
            "description": "Order rheumatoid factor test.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hysterosalpingogram",
            "description": "Order a hysterosalpingogram.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "other",
            "description": "Order another examination or test not covered by specific categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Description of the exam/test"}
                },
                "required": ["description"],
            },
        },
    },
]

# Build a lookup: function_name -> schema
TOOL_SCHEMA_MAP = {t["function"]["name"]: t for t in TOOL_SCHEMAS}


# ---------------------------------------------------------------------------
# Allowed parameter values (enums) per tool
# ---------------------------------------------------------------------------

# Shared region sets
_IMAGING_REGIONS = [
    "head", "neck", "chest", "abdomen", "pelvis", "spine",
    "upper_extremity", "lower_extremity", "whole_body",
]
_ULTRASOUND_REGIONS = _IMAGING_REGIONS + [
    "thyroid", "breast", "cardiac", "obstetric", "vascular",
    "musculoskeletal", "scrotal",
]
_CONTRAST_VALUES = ["yes", "no"]

TOOL_PARAM_VALUES: dict[str, dict[str, list[str]]] = {
    # ===== IMAGING =====
    "xray": {
        "region": _IMAGING_REGIONS,
    },
    "ct": {
        "region": _IMAGING_REGIONS,
        "contrast": _CONTRAST_VALUES,
    },
    "mri": {
        "region": _IMAGING_REGIONS,
        "contrast": _CONTRAST_VALUES,
    },
    "ultrasound": {
        "region": _ULTRASOUND_REGIONS,
    },
    "pet_scan": {
        "region": _IMAGING_REGIONS,
        "tracer": ["fdg", "gallium_68", "florbetapir", "choline", "psma"],
    },
    "nuclear_scan": {
        "type": [
            "bone_scan", "thyroid_scan", "renal_scan", "hida",
            "vq_scan", "mibg", "gallium_scan", "parathyroid_scan",
            "cardiac_perfusion", "octreotide_scan", "white_blood_cell_scan",
        ],
        "region": _IMAGING_REGIONS,
    },
    "angiography": {
        "type": ["conventional", "ct_angiography", "mr_angiography", "digital_subtraction"],
        "region": [
            "coronary", "cerebral", "pulmonary", "aortic",
            "peripheral", "renal", "mesenteric", "carotid",
        ],
    },
    "fluoroscopy": {
        "type": [
            "barium_swallow", "barium_enema", "upper_gi_series",
            "voiding_cystourethrogram", "intravenous_pyelogram",
            "contrast_enema", "esophagram", "small_bowel_follow_through",
        ],
    },
    "mammography": {
        "type": ["screening", "diagnostic"],
    },
    "dexa_scan": {
        "region": ["hip", "spine", "whole_body", "forearm"],
    },

    # ===== CARDIAC =====
    "ecg": {
        "type": ["standard_12_lead", "rhythm_strip", "signal_averaged"],
    },
    "echocardiogram": {
        "type": ["transthoracic", "transesophageal", "stress", "fetal"],
        "contrast": _CONTRAST_VALUES,
    },
    "echocardiography": {
        "type": ["transthoracic", "transesophageal", "stress", "fetal"],
        "region": _IMAGING_REGIONS,
    },
    "cardiac_catheterization": {
        "type": ["left_heart", "right_heart", "both", "coronary_angiography"],
    },
    "holter_monitor": {
        "duration": ["24h", "48h", "72h", "7d", "14d", "30d"],
    },
    "stress_test": {
        "type": ["exercise", "pharmacologic", "nuclear"],
    },

    # ===== LAB - BLOOD =====
    "cbc": {
        "type": ["cbc", "cbc_with_differential", "peripheral_smear"],
    },
    "coagulation_panel": {
        "type": ["pt_inr", "aptt", "d_dimer", "fibrinogen", "full_panel"],
    },
    "blood_gas": {
        "type": ["arterial", "venous"],
    },
    "tumor_marker": {
        "marker": [
            "cea", "afp", "ca_125", "ca_19_9", "psa", "beta_hcg",
            "ldh", "ca_15_3", "chromogranin_a", "nse",
        ],
    },
    "inflammatory_marker": {
        "marker": ["crp", "esr", "procalcitonin", "ferritin"],
    },
    "cardiac_enzyme": {
        "marker": ["troponin", "ck_mb", "bnp", "nt_probnp", "myoglobin"],
    },
    "hormone_level": {
        "hormone": [
            "cortisol", "acth", "growth_hormone", "igf_1", "insulin",
            "estrogen", "progesterone", "testosterone", "pth",
            "prolactin", "aldosterone", "renin", "catecholamines",
            "tsh", "fsh", "lh", "dhea_s", "vitamin_d",
        ],
    },
    "drug_level": {
        "drug": [
            "vancomycin", "gentamicin", "digoxin", "lithium",
            "phenytoin", "valproic_acid", "carbamazepine", "theophylline",
            "methotrexate", "cyclosporine", "tacrolimus", "warfarin",
            "toxicology_screen",
        ],
    },
    "blood_test": {
        "test": [
            "general_laboratory_panel", "coombs_test", "ana",
            "anca", "hba1c", "ldh", "uric_acid", "amylase", "lipase",
            "complement_levels", "immunoglobulin_levels", "vitamin_b12",
            "folate", "hemoglobin_electrophoresis", "blood_type",
            "glucose_tolerance_test", "reticulocyte_count",
            "direct_antiglobulin_test", "indirect_antiglobulin_test",
            "protein_electrophoresis", "anti_dsdna", "anti_smith",
            "anti_ccp", "anti_gliadin", "anti_tpo",
            "beta_2_microglobulin", "haptoglobin", "ammonia",
        ],
    },

    # ===== LAB - MICRO =====
    "culture": {
        "source": [
            "blood", "urine", "sputum", "wound", "csf", "stool",
            "throat", "tissue", "abscess", "peritoneal_fluid",
            "pleural_fluid", "joint_fluid", "bronchoalveolar_lavage",
            "catheter_tip",
        ],
    },
    "serology": {
        "test": [
            "hepatitis_a", "hepatitis_b", "hepatitis_c", "hiv",
            "syphilis", "ebv", "cmv", "lyme", "autoimmune_panel",
            "toxoplasma", "rubella", "hsv", "varicella", "measles",
            "brucella", "dengue", "rickettsia", "aspergillus",
            "cryptococcus", "coxsackievirus",
        ],
    },
    "pcr_test": {
        "target": [
            "hiv", "hepatitis_b", "hepatitis_c", "cmv", "ebv",
            "hsv", "tuberculosis", "covid_19", "influenza",
            "mrsa", "clostridium_difficile", "hpv",
        ],
    },

    # ===== LAB - URINE =====
    "urinalysis": {
        "type": ["dipstick", "microscopy", "full_urinalysis"],
    },
    "urine_test": {
        "test": [
            "24h_urine_protein", "24h_urine_creatinine",
            "urine_electrolytes", "urine_osmolality",
            "urine_drug_screen", "urine_catecholamines",
            "urine_cortisol", "urine_porphyrins",
            "urine_organic_acids", "urine_amino_acids",
        ],
    },
    "urine_cytology": {
        "type": ["voided", "catheterized"],
    },

    # ===== LAB - OTHER =====
    "stool_test": {
        "test": [
            "occult_blood", "ova_and_parasites", "calprotectin",
            "elastase", "fat", "clostridium_difficile_toxin",
            "giardia_antigen", "reducing_substances",
        ],
    },

    # ===== PATHOLOGY =====
    "biopsy": {
        "site": [
            "skin", "liver", "kidney", "lung", "lymph_node", "breast",
            "thyroid", "gastrointestinal", "brain", "bone", "soft_tissue",
            "prostate", "cervix", "endometrium", "muscle", "nerve",
            "pleura", "peritoneum", "bone_marrow", "salivary_gland",
            "adrenal", "bladder", "pancreas", "spleen", "testis",
        ],
        "type": [
            "excisional", "incisional", "core_needle", "fine_needle",
            "punch", "shave", "open",
        ],
    },
    "histopathology": {
        "specimen": [
            "surgical_resection", "biopsy_specimen", "excised_lesion",
            "polyp", "tumor", "appendix", "gallbladder", "lymph_node",
            "skin", "bone", "soft_tissue",
        ],
    },
    "cytology": {
        "type": ["pap_smear", "fnac", "fluid_cytology", "brushing", "wash"],
        "site": [
            "cervix", "thyroid", "breast", "lung", "lymph_node",
            "pleural_fluid", "peritoneal_fluid", "csf",
        ],
    },
    "immunohistochemistry": {
        "markers": [
            "ki_67", "her2", "er_pr", "cd20", "cd3", "cd45",
            "ck7", "ck20", "ttf1", "p63", "s100", "sma",
            "desmin", "vimentin", "synaptophysin", "chromogranin",
            "multiple", "other",
        ],
    },
    "flow_cytometry": {
        "specimen": ["blood", "bone_marrow", "tissue", "csf", "pleural_fluid"],
    },

    # ===== PROCEDURES =====
    "bone_marrow_biopsy": {
        "type": ["aspiration", "biopsy", "both"],
    },
    "endoscopy": {
        "type": [
            "egd", "colonoscopy", "bronchoscopy", "cystoscopy",
            "sigmoidoscopy", "ercp", "eus", "laryngoscopy",
            "capsule_endoscopy", "rhinoscopy", "arthroscopy",
            "hysteroscopy", "colposcopy", "enteroscopy",
        ],
    },
    "arthrocentesis": {
        "joint": [
            "knee", "shoulder", "hip", "ankle", "wrist",
            "elbow", "finger", "toe", "sacroiliac",
        ],
    },

    # ===== NEURO =====
    "eeg": {
        "type": ["routine", "continuous", "video", "sleep_deprived", "ambulatory"],
    },
    "emg": {
        "region": [
            "upper_extremity", "lower_extremity", "facial",
            "paraspinal", "whole_body",
        ],
    },
    "evoked_potentials": {
        "type": ["visual", "auditory", "somatosensory"],
    },

    # ===== PHYSICAL EXAM =====
    "physical_examination": {
        "type": [
            "general", "neurological", "cardiovascular", "respiratory",
            "abdominal", "musculoskeletal", "dermatological",
            "ophthalmologic", "ent", "rectal", "breast",
            "genitourinary", "vital_signs", "psychiatric",
        ],
    },

    # ===== PULMONARY =====
    "pulmonary_function_test": {
        "type": [
            "spirometry", "full_pft", "dlco",
            "methacholine_challenge", "bronchoprovocation",
        ],
    },

    # ===== GENETIC =====
    "genetic_test": {
        "test": [
            "karyotyping", "fish", "gene_sequencing", "chromosomal_microarray",
            "mutation_analysis", "whole_exome_sequencing", "pcr_based",
            "methylation_analysis", "linkage_analysis",
        ],
    },
    "molecular_test": {
        "test": [
            "fish", "pcr", "sequencing", "microarray",
            "methylation", "gene_expression", "mutation_panel",
        ],
    },

    # ===== EYE =====
    "eye_exam": {
        "type": [
            "fundoscopy", "slit_lamp", "tonometry", "visual_acuity",
            "visual_field", "oct", "fluorescein_angiography",
            "gonioscopy", "color_vision", "refraction",
        ],
    },

    # ===== SKIN / ALLERGY =====
    "skin_test": {
        "type": [
            "tuberculin", "patch_test", "prick_test", "koh_prep",
            "woods_lamp", "dermoscopy", "skin_scraping", "tzanck_smear",
        ],
    },
    "allergy_test": {
        "type": ["skin_prick", "rast", "specific_ige", "patch_test"],
    },

    # ===== OTHER BLOOD-ADJACENT =====
    "viral_load": {
        "test": ["hiv", "hepatitis_b", "hepatitis_c", "cmv", "ebv"],
    },
}


# ---------------------------------------------------------------------------
# Alias mappings: (tool_name, param_name) -> {variant -> canonical}
# ---------------------------------------------------------------------------

# Shared region aliases
_REGION_ALIASES = {
    "brain": "head", "cranial": "head", "cranium": "head",
    "face": "head", "facial": "head", "skull": "head",
    "thorax": "chest", "thoracic": "chest", "lung": "chest", "lungs": "chest",
    "cardiac": "chest", "heart": "chest", "mediastinum": "chest",
    "abdominal": "abdomen", "hepatobiliary": "abdomen", "liver": "abdomen",
    "hepatic": "abdomen", "pancreas": "abdomen", "splenic": "abdomen",
    "spleen": "abdomen", "kidney": "abdomen", "renal": "abdomen",
    "adrenal": "abdomen", "gallbladder": "abdomen", "biliary": "abdomen",
    "gastrointestinal": "abdomen", "bowel": "abdomen", "intestinal": "abdomen",
    "pelvic": "pelvis", "hip": "pelvis", "bladder": "pelvis",
    "uterus": "pelvis", "uterine": "pelvis", "ovary": "pelvis", "ovarian": "pelvis",
    "prostate": "pelvis", "rectal": "pelvis", "sacrum": "pelvis",
    "cervical_spine": "spine", "thoracic_spine": "spine", "lumbar_spine": "spine",
    "lumbosacral": "spine", "cervical": "spine", "lumbar": "spine",
    "shoulder": "upper_extremity", "arm": "upper_extremity", "elbow": "upper_extremity",
    "wrist": "upper_extremity", "hand": "upper_extremity", "finger": "upper_extremity",
    "forearm": "upper_extremity", "humerus": "upper_extremity",
    "knee": "lower_extremity", "ankle": "lower_extremity", "foot": "lower_extremity",
    "leg": "lower_extremity", "femur": "lower_extremity", "tibia": "lower_extremity",
    "thigh": "lower_extremity", "toe": "lower_extremity", "calcaneus": "lower_extremity",
    "shin": "lower_extremity", "fibula": "lower_extremity",
    "whole body": "whole_body", "full body": "whole_body", "total body": "whole_body",
}

_CONTRAST_ALIASES = {
    "with": "yes", "with contrast": "yes", "enhanced": "yes", "contrast enhanced": "yes",
    "without": "no", "without contrast": "no", "non-contrast": "no",
    "non contrast": "no", "plain": "no", "unenhanced": "no",
}

_ECHO_TYPE_ALIASES = {
    "tte": "transthoracic", "tee": "transesophageal",
    "doppler": "transthoracic", "2d": "transthoracic",
}

TOOL_PARAM_ALIASES: dict[tuple[str, str], dict[str, str]] = {
    # Imaging regions
    ("xray", "region"): _REGION_ALIASES,
    ("ct", "region"): _REGION_ALIASES,
    ("mri", "region"): _REGION_ALIASES,
    ("ultrasound", "region"): {
        **_REGION_ALIASES,
        "carotid": "vascular", "doppler": "vascular", "venous": "vascular",
        "arterial": "vascular", "dvt": "vascular",
        "fetal": "obstetric", "pregnancy": "obstetric", "ob": "obstetric",
        "testicular": "scrotal", "testes": "scrotal",
        "echocardiogram": "cardiac", "echo": "cardiac",
    },
    ("dexa_scan", "region"): {
        "lumbar": "spine", "lumbar_spine": "spine",
        "femoral": "hip", "femoral_neck": "hip",
        "whole body": "whole_body", "total body": "whole_body",
        "radius": "forearm", "wrist": "forearm",
    },
    # Contrast
    ("ct", "contrast"): _CONTRAST_ALIASES,
    ("mri", "contrast"): _CONTRAST_ALIASES,
    ("echocardiogram", "contrast"): _CONTRAST_ALIASES,
    # Echo types
    ("echocardiogram", "type"): _ECHO_TYPE_ALIASES,
    ("echocardiography", "type"): _ECHO_TYPE_ALIASES,
    # Blood gas
    ("blood_gas", "type"): {
        "abg": "arterial", "vbg": "venous",
    },
    # Cardiac cath
    ("cardiac_catheterization", "type"): {
        "left": "left_heart", "right": "right_heart",
        "coronary": "coronary_angiography",
    },
    # Angiography
    ("angiography", "type"): {
        "cta": "ct_angiography", "mra": "mr_angiography",
        "dsa": "digital_subtraction",
    },
}


# ---------------------------------------------------------------------------
# Parameter normalization helpers
# ---------------------------------------------------------------------------

def normalize_param_value(tool_name: str, param_name: str, value: str) -> str | None:
    """Normalize a parameter value to its canonical enum form.

    Tries in order: direct match -> alias match -> substring containment.
    Returns the canonical value or None if no match.
    """
    if tool_name not in TOOL_PARAM_VALUES:
        return None
    allowed = TOOL_PARAM_VALUES[tool_name].get(param_name)
    if allowed is None:
        return None

    val = value.strip().lower().replace("-", "_").replace(" ", "_")

    # 1) Direct match
    if val in allowed:
        return val

    # 2) Alias match
    aliases = TOOL_PARAM_ALIASES.get((tool_name, param_name), {})
    if val in aliases:
        return aliases[val]
    # Also try original (un-normalized) value for aliases like "with contrast"
    val_orig = value.strip().lower()
    if val_orig in aliases:
        return aliases[val_orig]

    # 3) Substring containment: check if any allowed value is contained in the input
    for canon in allowed:
        if canon in val:
            return canon
    # Or if input is contained in any allowed value
    for canon in allowed:
        if val in canon:
            return canon

    # 4) Check aliases with substring matching
    for alias_key, alias_val in aliases.items():
        if alias_key in val or alias_key in val_orig:
            return alias_val

    return None


def rebuild_tool_call_string(func_name: str, arguments: dict[str, str]) -> str:
    """Reconstruct a tool call string from function name and arguments dict."""
    if not arguments:
        return f"{func_name}()"
    parts = [f'{k}="{v}"' for k, v in arguments.items()]
    return f"{func_name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Inject enums into TOOL_SCHEMAS at module load time
# ---------------------------------------------------------------------------
for _schema in TOOL_SCHEMAS:
    _func_name = _schema["function"]["name"]
    if _func_name in TOOL_PARAM_VALUES:
        _props = _schema["function"]["parameters"].get("properties", {})
        for _param_name, _allowed in TOOL_PARAM_VALUES[_func_name].items():
            if _param_name in _props:
                _props[_param_name]["enum"] = _allowed


# ---------------------------------------------------------------------------
# Tool selection for training data
# ---------------------------------------------------------------------------

def get_tool_info(tool_name: str) -> dict | None:
    """Get tool info (name, description, parameters) for a given tool name.

    Returns:
        {"name": ..., "description": ..., "parameters": {...}} or None if not found.
    """
    schema = TOOL_SCHEMA_MAP.get(tool_name)
    if schema is None:
        return None

    func = schema["function"]
    return {
        "name": func["name"],
        "description": func["description"],
        "parameters": func["parameters"],
    }


def select_tools_for_case(
    ground_truth_tools: list[str],
    num_distractors: int = 5,
    shuffle: bool = True,
) -> list[dict]:
    """Select tools for a case: ground truth tools + random distractors.

    Args:
        ground_truth_tools: List of tool names that are correct for this case.
        num_distractors: Number of distractor tools to add.
        shuffle: Whether to shuffle the final list.

    Returns:
        List of tool info dicts for the system prompt.
    """
    import random

    # Get ground truth tool infos (skip "other", deduplicate)
    selected = []
    gt_set = set()
    for name in ground_truth_tools:
        if name == "other" or name in gt_set:
            continue
        info = get_tool_info(name)
        if info:
            selected.append(info)
            gt_set.add(name)

    # Get all available tool names (excluding ground truth and "other")
    all_names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    distractor_pool = [n for n in all_names if n not in gt_set and n != "other"]

    # Sample distractors
    num_to_sample = min(num_distractors, len(distractor_pool))
    distractor_names = random.sample(distractor_pool, num_to_sample)

    for name in distractor_names:
        info = get_tool_info(name)
        if info:
            selected.append(info)

    # Shuffle if requested
    if shuffle:
        random.shuffle(selected)

    return selected


# ---------------------------------------------------------------------------
# Exam string parsing
# ---------------------------------------------------------------------------

def parse_exam_string(exam_str: str) -> dict | None:
    """Parse an exam string like 'xray(region="chest")' into a dict.

    Returns:
        {"name": "xray", "arguments": {"region": "chest"}}
        or None if parsing fails.
    """
    if not exam_str or not exam_str.strip():
        return None

    exam_str = exam_str.strip()

    # Match function_name(args...)
    m = re.match(r"(\w+)\((.*)\)$", exam_str, re.DOTALL)
    if not m:
        return None

    func_name = m.group(1)
    args_str = m.group(2).strip()

    arguments = {}
    if args_str:
        # Parse key="value" pairs - handles escaped quotes inside values
        for pm in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
            arguments[pm.group(1)] = pm.group(2)

        # Handle positional argument without key=value (e.g. blood_gas(arterial))
        if not arguments and re.match(r"^[a-zA-Z_]\w*$", args_str):
            # Map known positional patterns to proper parameter names
            _POSITIONAL_PARAM = {
                "blood_gas": "type",
            }
            param_name = _POSITIONAL_PARAM.get(func_name, "type")
            arguments[param_name] = args_str

    return {"name": func_name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Findings formatting
# ---------------------------------------------------------------------------

# Templates mapping exam function names to natural-language result phrasing
_FINDINGS_TEMPLATES = {
    "xray": "The {region} X-ray reveals: {findings}",
    "ct": "The CT scan of the {region} shows: {findings}",
    "mri": "The MRI of the {region} shows: {findings}",
    "ultrasound": "The ultrasound of the {region} shows: {findings}",
    "blood_test": "The {test} results: {findings}",
    "cbc": "The complete blood count results: {findings}",
    "bmp": "The basic metabolic panel results: {findings}",
    "cmp": "The comprehensive metabolic panel results: {findings}",
    "lipid_panel": "The lipid panel results: {findings}",
    "liver_function_test": "The liver function test results: {findings}",
    "renal_function_test": "The renal function test results: {findings}",
    "thyroid_function_test": "The thyroid function test results: {findings}",
    "coagulation_panel": "The coagulation study results: {findings}",
    "iron_studies": "The iron studies results: {findings}",
    "inflammatory_marker": "The inflammatory marker results: {findings}",
    "cardiac_enzyme": "The cardiac enzyme results: {findings}",
    "tumor_marker": "The tumor marker results: {findings}",
    "hormone_level": "The {hormone} level results: {findings}",
    "drug_level": "The {drug} drug level results: {findings}",
    "blood_gas": "The blood gas analysis results: {findings}",
    "ecg": "The ECG shows: {findings}",
    "echocardiogram": "The echocardiogram shows: {findings}",
    "echocardiography": "The echocardiography shows: {findings}",
    "holter_monitor": "The Holter monitor results: {findings}",
    "stress_test": "The stress test results: {findings}",
    "cardiac_catheterization": "The cardiac catheterization shows: {findings}",
    "eeg": "The EEG shows: {findings}",
    "emg": "The EMG/nerve conduction study shows: {findings}",
    "evoked_potentials": "The evoked potentials study shows: {findings}",
    "pulmonary_function_test": "The pulmonary function test results: {findings}",
    "endoscopy": "The endoscopy reveals: {findings}",
    "biopsy": "The biopsy of {site} shows: {findings}",
    "bone_marrow_biopsy": "The bone marrow biopsy shows: {findings}",
    "lumbar_puncture": "The lumbar puncture results: {findings}",
    "csf_analysis": "The CSF analysis results: {findings}",
    "arthrocentesis": "The joint aspiration of {joint} shows: {findings}",
    "paracentesis": "The paracentesis fluid analysis shows: {findings}",
    "thoracentesis": "The thoracentesis fluid analysis shows: {findings}",
    "amniocentesis": "The amniocentesis results: {findings}",
    "culture": "The culture from {source} shows: {findings}",
    "serology": "The {test} serology results: {findings}",
    "pcr_test": "The PCR test for {target} results: {findings}",
    "viral_load": "The viral load test results: {findings}",
    "genetic_test": "The genetic test ({test}) results: {findings}",
    "molecular_test": "The molecular test results: {findings}",
    "histopathology": "The histopathology of {specimen} shows: {findings}",
    "immunohistochemistry": "The immunohistochemistry results: {findings}",
    "cytology": "The cytology results: {findings}",
    "flow_cytometry": "The flow cytometry results: {findings}",
    "urinalysis": "The urinalysis results: {findings}",
    "urine_test": "The urine test ({test}) results: {findings}",
    "urine_cytology": "The urine cytology results: {findings}",
    "stool_test": "The stool test results: {findings}",
    "pet_scan": "The PET scan shows: {findings}",
    "nuclear_scan": "The nuclear scan shows: {findings}",
    "mammography": "The mammography shows: {findings}",
    "dexa_scan": "The DEXA scan results: {findings}",
    "fluoroscopy": "The fluoroscopic study shows: {findings}",
    "angiography": "The angiography of {region} shows: {findings}",
    "eye_exam": "The eye examination shows: {findings}",
    "physical_examination": "The physical examination reveals: {findings}",
    "allergy_test": "The allergy testing results: {findings}",
    "skin_test": "The skin test results: {findings}",
    "rheumatoid_factor": "The rheumatoid factor test results: {findings}",
    "hysterosalpingogram": "The hysterosalpingogram shows: {findings}",
    "other": "The {description} results: {findings}",
}


def format_findings(exam_name: str, exam_args: dict, raw_findings: str) -> str:
    """Convert raw findings into a natural-language sentence.

    Args:
        exam_name: Function name (e.g. "xray").
        exam_args: Arguments dict (e.g. {"region": "chest"}).
        raw_findings: Raw findings string from the dataset.

    Returns:
        Formatted findings string.
    """
    template = _FINDINGS_TEMPLATES.get(exam_name, "The {exam_name} results: {findings}")

    # Build substitution dict from args + findings + exam_name
    subs = {**exam_args, "findings": raw_findings, "exam_name": exam_name}

    try:
        return template.format_map(subs)
    except KeyError:
        # Fallback if template references a key not in args
        return f"The {exam_name} results: {raw_findings}"


def build_tool_call_message(exam_str: str, findings: str, tool_call_id: str) -> tuple[dict, dict] | None:
    """Build the assistant tool_call message and the tool response message.

    Args:
        exam_str: Raw exam string from dataset (e.g. 'xray(region="chest")').
        findings: Raw findings string from dataset.
        tool_call_id: Unique ID for this tool call.

    Returns:
        Tuple of (assistant_message, tool_message) or None if parsing fails.
    """
    parsed = parse_exam_string(exam_str)
    if parsed is None:
        return None

    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed["arguments"]),
                },
            }
        ],
    }

    formatted = format_findings(parsed["name"], parsed["arguments"], findings)
    tool_msg = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": formatted,
    }

    return assistant_msg, tool_msg
