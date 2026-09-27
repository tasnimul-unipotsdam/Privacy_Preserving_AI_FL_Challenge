"""Offline, auditable rules for the synthetic challenge's clinical vocabulary.

Rules use the data dictionary and training notes plus public validation error
analysis; there is no case lookup. They are deliberately conservative around
absent values and unsupported units, and are not a clinical production
de-identifier.
"""

import re

from src.baseline import render_deidentified


DIAGNOSIS_ALIASES = {
    "atrial_fibrillation": (r"atrial fibrillation", r"AF", r"AFib", r"Vorhofflimmern"),
    "heart_failure": (
        r"heart failure", r"cardiac failure", r"HFrEF", r"HFpEF", r"CHF", r"CCF", r"LV failure",
    ),
    "hypertension": (r"hypertension", r"HTN", r"high blood pressure", r"hypertensive disease"),
    "type_2_diabetes": (
        r"type\s*(?:2|II)\s*(?:diabetes(?: mellitus)?|DM)",
        r"diabetes mellitus(?:\s*type\s*(?:2|II))?", r"T2DM", r"DM2",
    ),
    "chronic_kidney_disease": (r"chronic (?:kidney|renal) disease", r"CKD", r"chronic renal dysfunction"),
    "coronary_artery_disease": (r"coronary (?:artery )?disease", r"CAD", r"IHD", r"isch[ae]mic heart disease"),
    "acute_coronary_syndrome": (r"acute coronary syndrome", r"ACS", r"NSTEMI", r"STEMI", r"unstable angina"),
    "pneumonia": (r"pneumonia", r"bronchopneumonia", r"CAP", r"infective consolidation"),
    "copd": (
        r"COPD", r"COAD", r"chronic obstructive (?:pulmonary|airway) disease",
        r"chronic airway obstruction", r"obstructive lung disease",
    ),
}

MEDICATION_ALIASES = {
    "apixaban": (r"apixaban", r"Eliquis", r"APX"),
    "rivaroxaban": (r"rivaroxaban", r"Xarelto"),
    "warfarin": (r"warfarin",),
    "metoprolol": (r"metoprolol",),
    "bisoprolol": (r"bisoprolol",),
    "furosemide": (r"furosemide", r"frusemide", r"Lasix"),
    "ramipril": (r"ramipril",),
    "amlodipine": (r"amlodipine",),
    "metformin": (r"metformin",),
    "insulin": (r"insulin",),
    "atorvastatin": (r"atorvastatin", r"atorva"),
    "aspirin": (r"aspirin", r"ASA", r"acetylsalicylic acid", r"ecosprin"),
    "clopidogrel": (r"clopidogrel",),
    "amiodarone": (r"amiodarone",),
    "digoxin": (r"digoxin",),
    "azithromycin": (r"azithromycin", r"azithro"),
}

DATE = r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{4})"
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
KNOWN_ID = re.compile(r"\b(?:B-\d{6}|CHN-\d{7}|HYD\d{6}|BER/\d{4}/\d{2}|UHID/\d{5}/\d{2}|MRN-\d{2}-\d{5})\b")
NAME = r"[^\W\d_]+(?:['’.-][^\W\d_]+)*(?:[ \t]+[^\W\d_]+(?:['’.-][^\W\d_]+)*){1,4}"
TITLE = r"(?:Dr\.?[ \t]+)?"
SEPARATOR = r"[ \t]*[:=\-–—]?[ \t]*"
NUMBER = r"(?P<value>\d+(?:[.,]\d+)?)"


def _add_span(spans, start, end, label):
    """Keep exact original offsets and reject overlaps deterministically."""
    if start >= end or any(start < span["end"] and end > span["start"] for span in spans):
        return
    spans.append({"start": start, "end": end, "label": label})


def deidentify(note):
    """Return nonoverlapping PII spans and their placeholder rendering."""
    spans = []

    for match in EMAIL.finditer(note):
        _add_span(spans, *match.span(), "EMAIL")
    for match in KNOWN_ID.finditer(note):
        _add_span(spans, *match.span(), "PATIENT_ID")
    # Context permits unseen ID formats without treating clinical numbers as IDs.
    for match in re.finditer(
        r"\b(?:MRN|UHID|case\s*ID|patient\s*ID|record\s*ID)\s*[:=]\s*([\w/-]+)", note, re.I
    ):
        _add_span(spans, *match.span(1), "PATIENT_ID")

    for match in re.finditer(r"\b" + DATE + r"\b", note):
        before = note[max(0, match.start() - 40):match.start()]
        birth = re.search(r"(?:DOB|D\.O\.B\.|date\s+of\s+birth|birth\s*date|born)\s*[:=]?\s*$", before, re.I)
        _add_span(spans, *match.span(), "DATE_OF_BIRTH" if birth else "ENCOUNTER_DATE")

    # Anchoring phones to a contact cue avoids matching BP, dates and lab values.
    phone_pattern = (
        r"\b(?:telephone|phone|tel|mobile|contact(?:\s+number)?|ph)\b"
        + SEPARATOR + r"(?P<value>\+?\d[\d ()/-]*\d)"
    )
    for match in re.finditer(phone_pattern, note, re.I):
        if sum(ch.isdigit() for ch in match["value"]) >= 8:
            _add_span(spans, *match.span("value"), "PHONE_NUMBER")

    address_pattern = (
        r"\b(?P<cue>address|residence|home|residing\s+at|from)\b" + SEPARATOR
        + r"(?P<value>[^\n|;]+?)(?=\s*[|;\n]|\.(?:\s|$)|$)"
    )
    for match in re.finditer(address_pattern, note, re.I):
        value = match["value"]
        if match["cue"].lower() in {"from", "home"} and not re.search(
            r"\b(?:flat|apartment|house|street|road|lane|avenue)\b|\b\d{5,6}\b", value, re.I
        ):
            continue
        # All released addresses have a house/flat number. Avoid prose such as
        # "home medications" or "from pneumonia" being treated as an address.
        if re.search(r"\d", value) and not re.search(r"\b(?:medications?|medicines?|dose)\b|:", value, re.I):
            start, end = match.span("value")
            end -= len(value) - len(value.rstrip())
            _add_span(spans, start, end, "ADDRESS")

    name_rules = [
        ("PATIENT_NAME", r"\b(?:patient(?:\s+name)?|name|pt)" + SEPARATOR + TITLE + r"(?P<value>" + NAME + r")(?=\s*(?:[|\n;(\[/]|,\s*(?:born|DOB)|$))"),
        ("PATIENT_NAME", r"(?m)^" + TITLE + r"(?P<value>" + NAME + r")(?=,\s*born\b|\s*/\s*[A-Z0-9/-]+\s*/\s*born\b)"),
        ("CLINICIAN_NAME", r"\b(?:attending(?:\s+physician)?|treating\s+clinician|consultant|clinician|responsible\s+doctor|reviewed\s+by|author|(?:electronically\s+)?signed(?:\s+by)?)" + SEPARATOR + TITLE + r"(?P<value>" + NAME + r")(?=\s*(?:[|\n;(<,]|$))"),
    ]
    for label, pattern in name_rules:
        for match in re.finditer(pattern, note, re.I):
            # Bare "Patient reports ..." is ordinary prose. Explicit fields
            # accept lowercase names; ambiguous prose cues require a capital.
            prefix = note[match.start():match.start("value")]
            if not re.search(r"[:=]", prefix) and not match["value"][0].isupper():
                continue
            _add_span(spans, *match.span("value"), label)

    # A name established from a role cue remains PII in an unlabelled signature.
    names = [(note[s["start"]:s["end"]], s["label"]) for s in spans if s["label"].endswith("_NAME")]
    for name, label in names:
        for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", note):
            _add_span(spans, *match.span(), label)

    spans.sort(key=lambda span: (span["start"], span["end"]))
    return spans, render_deidentified(note, spans)


def _affirmed(note, start, end, *, medication=False):
    """Apply local forward/backward negation, uncertainty and experiencer cues."""
    before = re.split(r"[.;!?\n]|\b(?:but|however|yet)\b", note[:start], flags=re.I)[-1]
    after = re.split(r"[.;!?\n,]|\b(?:but|however|yet)\b", note[end:], flags=re.I)[0]
    excluded_before = (
        r"\b(?:no|not|denies|denied|without|negative\s+for|rule[ds]?\s+out|"
        r"exclude[ds]?|suspected|possible|consider(?:ed)?|family\s+history|"
        r"mother|father|sibling)\b"
    )
    excluded_after = r"\b(?:(?:was\s+)?(?:ruled\s+out|excluded|denied|considered)|not\s+(?:confirmed|present|diagnosed))\b"
    if medication:
        excluded_before += r"|\b(?:stop(?:ped)?|discontinued|held|withheld|hold|previously|past\s+medications|allergic\s+to|allerg(?:y|ies))\b"
        excluded_after += r"|\b(?:(?:was|is)\s+)?(?:stopped|discontinued|held|withheld|discussed|not\s+started)\b"
    return not (re.search(excluded_before, before, re.I) or re.search(excluded_after, after, re.I))


def _concepts(note, aliases, *, medication=False):
    found = []
    for canonical, alternatives in aliases.items():
        pattern = r"\b(?:" + "|".join(alternatives) + r")\b"
        if any(_affirmed(note, *match.span(), medication=medication) for match in re.finditer(pattern, note, re.I)):
            found.append(canonical)
    return sorted(found)


def _measurement(note, names, units=None):
    """Require a value next to its label; never borrow another field's number."""
    pattern = r"\b(?:" + names + r")\b\s*(?:(?:is|of|approximately|approx\.?)\s*)?[=:~]?\s*" + NUMBER
    if units:
        pattern += r"\s*(?P<unit>" + units + r")\b"
    match = re.search(pattern, note, re.I)
    if not match:
        return None, ""
    return float(match["value"].replace(",", ".")), match["unit"].lower().replace(" ", "") if units else ""


def _allergy(note):
    """A drug mention alone does not establish an allergy."""
    no_allergy = re.search(
        r"\b(?:NKDA|NKA|no\s+known\s+(?:drug\s+)?allergies|no\s+medication\s+allergy(?:\s+documented)?)\b",
        note, re.I,
    )
    for clause in re.split(r"[.;!?\n]", note):
        if not re.search(r"\ballerg(?:y|ies|ic)\b|\b(?:reaction|urticaria)\b", clause, re.I):
            continue
        for category, pattern in [
            ("penicillin", r"\bpenicillin\b"),
            ("nsaid", r"\b(?:NSAIDs?|ibuprofen|non[- ]steroidal\s+anti[- ]inflammatory)\b"),
            ("iodinated_contrast", r"\bcontrast\b"),
        ]:
            match = re.search(pattern, clause, re.I)
            if match and _affirmed(clause, *match.span()):
                return category
    return "none" if no_allergy else None


def extract_clinical(note):
    """Extract canonical active diagnoses, current medicines and observations."""
    heart_rate, _ = _measurement(note, r"HR|heart\s+rate|pulse|ventricular\s+rate")
    systolic, _ = _measurement(note, r"BP|blood\s+pressure|systolic(?:\s+BP|\s+blood\s+pressure)?|SBP")
    creatinine, creatinine_unit = _measurement(note, r"(?:serum\s+)?creatinine|creat|SCr", r"mg\s*/\s*d[lL]|[µμu]mol\s*/\s*[lL]")
    hemoglobin, hemoglobin_unit = _measurement(note, r"h(?:ae|e)moglobin|Hb|Hgb", r"g\s*/\s*d[lL]|g\s*/\s*[lL]")
    lvef, _ = _measurement(note, r"LVEF|EF|(?:left\s+ventricular\s+)?ejection\s+fraction")
    if creatinine is not None and creatinine_unit != "mg/dl":
        creatinine = round(creatinine / 88.4, 4)
    if hemoglobin is not None and hemoglobin_unit == "g/l":
        hemoglobin = round(hemoglobin / 10, 4)

    smoking = None
    for category, pattern in [
        ("former", r"\b(?:former\s+smoker|ex[- ]smoker|stopped\s+smoking|quit\s+smoking)\b"),
        ("never", r"\b(?:never\s+smoked|never\s+smoker|non[- ]?smoker|no\s+tobacco\s+use)\b"),
        ("current", r"\b(?:current\s+smoker|actively\s+smokes|ongoing\s+tobacco\s+use)\b"),
    ]:
        if re.search(pattern, note, re.I):
            smoking = category
            break

    return {
        "diagnoses": _concepts(note, DIAGNOSIS_ALIASES),
        "medications": _concepts(note, MEDICATION_ALIASES, medication=True),
        "heart_rate_bpm": heart_rate,
        "systolic_bp_mmhg": systolic,
        "creatinine_mg_dl": creatinine,
        "hemoglobin_g_dl": hemoglobin,
        "lvef_percent": lvef,
        "smoking_status": smoking,
        "allergy": _allergy(note),
    }
