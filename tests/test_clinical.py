"""Behavioural checks on invented notes, independent of benchmark labels."""
import pytest

from src.clinical import deidentify, extract_clinical


def test_pii_offsets_unicode_repeated_name_and_clinical_preservation():
    note = (
        "Patient: Zoë Müller | DOB: 1980-03-17 | Patient ID: ZZ/2026/91\n"
        "Address: Oak Lane 8, 10101 Testtown | Phone: +49 30 1234 5678\n"
        "Encounter date: 2026-05-01 | Consultant: Dr. Léa König\n"
        "Dx: AF. HR 105 bpm; BP 122/79. Creatinine 1.1 mg/dL.\n"
        "Signed by Dr. Léa König; lea.koenig@hospital.example."
    )
    spans, result = deidentify(note)
    labelled = [(note[s["start"]:s["end"]], s["label"]) for s in spans]
    assert ("Zoë Müller", "PATIENT_NAME") in labelled
    assert labelled.count(("Léa König", "CLINICIAN_NAME")) == 2
    assert ("Oak Lane 8, 10101 Testtown", "ADDRESS") in labelled
    assert ("ZZ/2026/91", "PATIENT_ID") in labelled
    assert ("1980-03-17", "DATE_OF_BIRTH") in labelled
    assert ("2026-05-01", "ENCOUNTER_DATE") in labelled
    assert ("+49 30 1234 5678", "PHONE_NUMBER") in labelled
    assert "Dr. [CLINICIAN_NAME]" in result
    assert "HR 105 bpm; BP 122/79. Creatinine 1.1 mg/dL." in result
    assert all(a["end"] <= b["start"] for a, b in zip(spans, spans[1:]))


def test_pii_bare_name_and_contextual_roles():
    note = (
        "SYNTHETIC RECORD\nAsha Test / MRN-26-12345 / born 03/04/1982 / seen 05/06/2026. "
        "Home: Flat 8-B, 90 Test Street, Sampletown 123456. "
        "Responsible doctor Nila Sample, email nila.sample@hospital.example."
    )
    spans, result = deidentify(note)
    assert {span["label"] for span in spans} == {
        "PATIENT_NAME", "PATIENT_ID", "DATE_OF_BIRTH", "ENCOUNTER_DATE", "ADDRESS", "CLINICIAN_NAME", "EMAIL"
    }
    assert "Flat 8-B" not in result
    assert "Asha Test" not in result


def test_no_clinical_number_or_home_medication_redaction():
    note = "Home medications: metoprolol 50 mg. HR 101. BP 120/80. Hb 123 g/L."
    assert deidentify(note) == ([], note)


def test_patient_prose_and_duration_are_not_names_or_addresses():
    note = (
        "Patient denies chest pain\nPatient reports no known allergies\n"
        "Recovering from 3 days of cough. Home BP 120/80."
    )
    assert deidentify(note) == ([], note)


def test_explicit_lowercase_name_is_retained_as_pii():
    note = "Patient: jane sample | DOB: 1980-01-02"
    spans, text = deidentify(note)
    assert any(s["label"] == "PATIENT_NAME" for s in spans)
    assert "jane sample" not in text


def test_aliases_and_units():
    result = extract_clinical(
        "Dx: HFrEF; Vorhofflimmern (AF); CKD-3; DM2; COAD; CAP. "
        "Rx: Eliquis; Lasix; ecosprin; frusemide; azithro. "
        "Ventricular rate 99 per minute; blood pressure 131 over 82. "
        "Serum creatinine 176,8 μmol/L; Hb 118 g/L. "
        "Left ventricular ejection fraction approximately 35 per cent. "
        "Smoking: ex-smoker. Allergy: contrast medium reaction."
    )
    assert result["diagnoses"] == ["atrial_fibrillation", "chronic_kidney_disease", "copd", "heart_failure", "pneumonia", "type_2_diabetes"]
    assert result["medications"] == ["apixaban", "aspirin", "azithromycin", "furosemide"]
    assert result["creatinine_mg_dl"] == pytest.approx(2.0)
    assert result["hemoglobin_g_dl"] == pytest.approx(11.8)
    assert result["heart_rate_bpm"] == 99
    assert result["systolic_bp_mmhg"] == 131
    assert result["lvef_percent"] == 35
    assert result["smoking_status"] == "former"
    assert result["allergy"] == "iodinated_contrast"


def test_negation_family_history_uncertainty_and_active_repeat():
    result = extract_clinical(
        "Family history of hypertension and heart failure. No COPD or pneumonia. "
        "Atrial fibrillation was considered but not confirmed. CAD ruled out. "
        "The patient denies a history of CKD. "
        "No hypertension but active AF. Confirmed type 2 diabetes."
    )
    assert result["diagnoses"] == ["atrial_fibrillation", "type_2_diabetes"]


def test_only_current_medications_and_word_boundaries():
    result = extract_clinical(
        "Warfarin discontinued. Hold metformin. Previously digoxin. "
        "No aspirin. Amiodarone was discussed but was not started. "
        "Rx: apixaban; metoprolol. Staff arranged follow-up."
    )
    assert result["medications"] == ["apixaban", "metoprolol"]
    assert result["diagnoses"] == []


def test_missing_values_do_not_borrow_other_numbers():
    result = extract_clinical(
        "HR not measured; BP not documented. Creatinine unavailable; Hb 13.2 g/dL. "
        "Ejection fraction not documented. Patient age 65."
    )
    for field in ("heart_rate_bpm", "systolic_bp_mmhg", "creatinine_mg_dl", "lvef_percent", "smoking_status", "allergy"):
        assert result[field] is None
    assert result["hemoglobin_g_dl"] == 13.2


def test_decimal_comma_and_unsupported_units():
    result = extract_clinical("Creatinine: 1,23 mg/dL; haemoglobin 12,8 g/dL; EF=51%.")
    assert result["creatinine_mg_dl"] == 1.23
    assert result["hemoglobin_g_dl"] == 12.8
    assert result["lvef_percent"] == 51
    assert extract_clinical("Creatinine 120 unknown_units.")["creatinine_mg_dl"] is None


def test_empty_note_returns_absent_values():
    assert deidentify("") == ([], "")
    result = extract_clinical("")
    assert result.pop("diagnoses") == []
    assert result.pop("medications") == []
    assert all(value is None for value in result.values())


def test_allergy_context_and_medication_safety():
    result = extract_clinical("Allergies: aspirin. Rx: metoprolol. Received penicillin previously.")
    assert result["medications"] == ["metoprolol"]
    assert result["allergy"] is None
    assert extract_clinical("No penicillin allergy.")["allergy"] is None
    assert extract_clinical("Drug allergy: ibuprofen.")["allergy"] == "nsaid"
    assert extract_clinical("Ibuprofen-associated urticaria.")["allergy"] == "nsaid"


def test_unstable_angina_is_acs_only_when_active():
    assert extract_clinical("Active diagnosis: unstable angina.")["diagnoses"] == ["acute_coronary_syndrome"]
    assert extract_clinical("Unstable angina was ruled out.")["diagnoses"] == []
    assert extract_clinical("Family history of unstable angina.")["diagnoses"] == []
