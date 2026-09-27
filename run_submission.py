#!/usr/bin/env python3
"""Train the synthetic challenge solution and predict without evaluation labels."""

import os

# Apply before importing NumPy. Fixed threads reduce machine-to-machine variation.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.clinical import deidentify, extract_clinical
from src.features import SITES, feature_matrix
from src.federated import run_experiments, sigmoid
from src.privacy import privacy_summary


def read_jsonl(path, *, training=False):
    records, seen = [], set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                case_id = record.get("case_id")
                if not isinstance(case_id, str) or not case_id or case_id in seen:
                    raise ValueError("missing or duplicate case_id")
                if record.get("hospital_id") not in SITES:
                    raise ValueError("unknown hospital_id")
                if not isinstance(record.get("note_text"), str) or not record["note_text"].strip():
                    raise ValueError("note_text must be nonempty text")
                features = record["structured_features"]
                for name in ("age_years", "prior_admissions_12m", "length_of_stay_days"):
                    value = features[name]
                    if type(value) is not int or value < 0:
                        raise ValueError(f"{name} must be a nonnegative integer")
                if features["age_years"] > 120:
                    raise ValueError("age_years must be at most 120")
                if features["sex"] not in ("female", "male") or type(features["emergency_admission"]) is not bool:
                    raise ValueError("invalid sex or emergency_admission")
                if training and (type(record["labels"]["readmission_30d"]) is not int
                                 or record["labels"]["readmission_30d"] not in (0, 1)):
                    raise ValueError("readmission_30d must be 0 or 1")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
            records.append(record)
            seen.add(case_id)
    if training and not records:
        raise ValueError("Training file is empty")
    return records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    try:
        sources = {args.train.resolve(), args.input.resolve()}
        destinations = [args.output.resolve(),
                        (args.artifacts_dir / "experiment_summary.json").resolve(),
                        (args.artifacts_dir / "privacy_summary.json").resolve()]
        if sources.intersection(destinations) or len(set(destinations)) != len(destinations):
            raise ValueError("Output and artifact paths must be distinct from each other and from input files")
        train_records = read_jsonl(args.train, training=True)
        evaluation_records = read_jsonl(args.input)
        overlap = {record["case_id"] for record in train_records}.intersection(
            record["case_id"] for record in evaluation_records)
        if overlap:
            raise ValueError("Training and evaluation case IDs overlap")
        weights, experiment, comparison = run_experiments(train_records)
        probabilities = sigmoid(feature_matrix(evaluation_records) @ weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for record, probability in zip(evaluation_records, probabilities):
                spans, rendered = deidentify(record["note_text"])
                prediction = {
                    "case_id": record["case_id"], "pii_entities": spans,
                    "deidentified_text": rendered,
                    "extracted_clinical_data": extract_clinical(record["note_text"]),
                    "readmission_probability": float(probability),
                }
                handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
        privacy = privacy_summary()
        privacy["empirical_comparison"] = comparison
        privacy["additional_disclosures"] = experiment["communication"]["evaluation_disclosure"]
        privacy["training_diagnostics"] = "Scalar training objectives and site aggregate statistics are released without DP and are outside the masking guarantee."
        experiment["total_runtime_seconds"] = perf_counter() - started
        write_json(args.artifacts_dir / "experiment_summary.json", experiment)
        write_json(args.artifacts_dir / "privacy_summary.json", privacy)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Submission failed: {exc}\n")
    print(f"Wrote {len(evaluation_records)} predictions to {args.output} in {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
