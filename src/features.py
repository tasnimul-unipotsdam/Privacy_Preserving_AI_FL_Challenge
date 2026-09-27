"""A fixed, shared feature contract; no pooled vocabulary or fitted scaler."""

import numpy as np

from src.clinical import extract_clinical

SITES = ("BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE")
DIAGNOSES = (
    "atrial_fibrillation", "heart_failure", "hypertension", "type_2_diabetes",
    "chronic_kidney_disease", "coronary_artery_disease",
    "acute_coronary_syndrome", "pneumonia", "copd",
)
# Public, fixed reference values rather than statistics estimated across hospitals.
MEASUREMENTS = {
    "heart_rate_bpm": (80.0, 30.0),
    "systolic_bp_mmhg": (130.0, 30.0),
    "creatinine_mg_dl": (1.0, 1.0),
    "hemoglobin_g_dl": (13.0, 3.0),
    "lvef_percent": (55.0, 20.0),
}
FEATURE_NAMES = (
    ["intercept", "age_years", "male", "prior_admissions_12m",
     "length_of_stay_days", "emergency_admission"]
    + list(DIAGNOSES)
    + [name for field in MEASUREMENTS for name in (field, field + "_missing")]
    + ["smoking_current", "smoking_former", "smoking_missing"]
)


def feature_vector(record):
    """Use only input fields, including our extraction, never released gold labels."""
    structured = record["structured_features"]
    clinical = extract_clinical(record["note_text"])
    values = [
        1.0,
        (structured["age_years"] - 60.0) / 20.0,
        float(structured["sex"] == "male"),
        structured["prior_admissions_12m"] / 2.0,
        (structured["length_of_stay_days"] - 5.0) / 5.0,
        float(structured["emergency_admission"]),
    ]
    values.extend(float(name in clinical["diagnoses"]) for name in DIAGNOSES)
    for field, (center, scale) in MEASUREMENTS.items():
        value = clinical[field]
        values.extend((0.0 if value is None else (value - center) / scale,
                       float(value is None)))
    values.extend(float(clinical["smoking_status"] == status)
                  for status in ("current", "former", None))
    # Omit site indicators: an update coordinate supported by just one hospital
    # can reveal that hospital's contribution even through secure aggregation.
    result = np.asarray(values, dtype=np.float64)
    # Bound extreme inputs without estimating statistics on evaluation data.
    result[1:] = np.clip(result[1:], -5.0, 5.0)
    return result


def feature_matrix(records):
    if not records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([feature_vector(record) for record in records])
