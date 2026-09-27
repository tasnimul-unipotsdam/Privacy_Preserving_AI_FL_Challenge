"""Exercise the reviewers' CLI with labels-free inputs outside the repository."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/prediction.schema.json").read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def run_cli(directory, train_path, input_path):
    output_path = directory / "nested output" / "predictions.jsonl"
    artifact_path = directory / "artifacts"
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_submission.py"),
            "--train", str(train_path),
            "--input", str(input_path),
            "--output", str(output_path),
            "--artifacts-dir", str(artifact_path),
        ],
        cwd=directory,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return read_jsonl(output_path), artifact_path


@pytest.fixture(scope="module")
def submission_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("submission outside repository")
    train_path = directory / "training records.jsonl"
    shutil.copyfile(ROOT / "data/train.jsonl", train_path)
    records = read_jsonl(ROOT / "data/validation_inputs.jsonl")[:3]
    for index, record in enumerate(records):
        record["case_id"] = f"unseen-case-{index}"
        # Unicode before every entity catches byte offsets used as character offsets.
        record["note_text"] = "Synthetic note — α\n" + record["note_text"]
        assert "labels" not in record

    first_directory = directory / "first run"
    first_directory.mkdir()
    input_path = first_directory / "unlabelled records.jsonl"
    write_jsonl(input_path, records)
    first, artifacts = run_cli(first_directory, train_path, input_path)

    second_directory = directory / "second run"
    second_directory.mkdir()
    reordered_path = second_directory / "different inputs.jsonl"
    write_jsonl(reordered_path, list(reversed(records)))
    second, _ = run_cli(second_directory, train_path, reordered_path)
    return records, first, second, artifacts


def test_cli_predictions_follow_the_submission_contract(submission_runs):
    inputs, predictions, _, _ = submission_runs
    indexed_inputs = {record["case_id"]: record for record in inputs}
    assert len(predictions) == len(inputs)
    assert {record["case_id"] for record in predictions} == set(indexed_inputs)
    clinical_schema = SCHEMA["properties"]["extracted_clinical_data"]
    pii_labels = SCHEMA["properties"]["pii_entities"]["items"]["properties"]["label"]["enum"]

    for prediction in predictions:
        assert set(prediction) == set(SCHEMA["required"])
        probability = prediction["readmission_probability"]
        assert type(probability) in (int, float)
        assert math.isfinite(probability) and 0.0 <= probability <= 1.0

        note = indexed_inputs[prediction["case_id"]]["note_text"]
        spans = sorted(prediction["pii_entities"], key=lambda span: span["start"])
        previous_end = 0
        for span in spans:
            assert type(span["start"]) is int and type(span["end"]) is int
            assert previous_end <= span["start"] < span["end"] <= len(note)
            assert span["label"] in pii_labels
            previous_end = span["end"]
        rendered = note
        for span in reversed(spans):
            rendered = rendered[:span["start"]] + f"[{span['label']}]" + rendered[span["end"]:]
        assert prediction["deidentified_text"] == rendered

        extracted = prediction["extracted_clinical_data"]
        assert set(extracted) == set(clinical_schema["required"])
        for field, constraints in clinical_schema["properties"].items():
            value = extracted[field]
            if field in ("diagnoses", "medications"):
                assert isinstance(value, list) and len(value) == len(set(value))
                assert set(value).issubset(constraints["items"]["enum"])
            elif "enum" in constraints:
                assert value in constraints["enum"]
            else:
                assert value is None or (type(value) in (int, float) and math.isfinite(value))


def test_prediction_is_reproducible_and_independent_of_input_order(submission_runs):
    _, first, second, _ = submission_runs
    indexed_second = {record["case_id"]: record for record in second}
    for record in first:
        repeated = indexed_second[record["case_id"]]
        # BLAS may choose a different summation order for reversed matrix rows.
        assert record["readmission_probability"] == pytest.approx(
            repeated["readmission_probability"], abs=1e-12, rel=0.0)
        assert {key: value for key, value in record.items() if key != "readmission_probability"} == {
            key: value for key, value in repeated.items() if key != "readmission_probability"
        }


def test_cli_writes_implemented_experiment_and_privacy_summaries(submission_runs):
    _, _, _, artifacts = submission_runs
    experiment = json.loads((artifacts / "experiment_summary.json").read_text(encoding="utf-8"))
    privacy = json.loads((artifacts / "privacy_summary.json").read_text(encoding="utf-8"))
    for summary in (experiment, privacy):
        assert isinstance(summary, dict) and summary
        assert "TODO" not in json.dumps(summary)
        assert "starter_baseline" not in json.dumps(summary)
    for name in ("local_models", "federated_model", "centralized_model"):
        assert experiment.get(name), f"The experiment must report {name}"
        validation = experiment[name]["validation"]
        assert set(validation["by_site"]) == {"BERLIN_NODE", "CHENNAI_NODE", "HYDERABAD_NODE"}
        assert validation["overall"]["n"] == sum(site["n"] for site in validation["by_site"].values())
        assert 0.0 <= validation["overall"]["brier_score"] <= 1.0
        assert validation["overall"]["log_loss"] >= 0.0
    assert experiment["validation_source"]
    assert experiment["seed_runs"]
    assert experiment["communication"]["raw_rows_transferred_in_federated_training"] is False
    assert privacy.get("mechanism"), "The additional privacy mechanism must be identified"


@pytest.mark.parametrize("raw, message", [
    ("{broken json}\n", "line 1"),
    ("[]\n", "expected a JSON object"),
    ("null\n", "expected a JSON object"),
])
def test_invalid_jsonl_has_an_actionable_error(tmp_path, raw, message):
    from run_submission import read_jsonl as read_submission_inputs

    path = tmp_path / "invalid.jsonl"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_submission_inputs(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    from run_submission import read_jsonl as read_submission_inputs

    record = read_jsonl(ROOT / "data/validation_inputs.jsonl")[0]
    path = tmp_path / "duplicate.jsonl"
    write_jsonl(path, [record, record])
    with pytest.raises(ValueError, match="line 2.*duplicate case_id"):
        read_submission_inputs(path)


def test_invalid_numeric_features_and_training_labels_are_rejected(tmp_path):
    from run_submission import read_jsonl as read_submission_inputs

    record = read_jsonl(ROOT / "data/train.jsonl")[0]
    path = tmp_path / "invalid-training.jsonl"
    record["labels"]["readmission_30d"] = True
    write_jsonl(path, [record])
    with pytest.raises(ValueError, match="readmission_30d must be 0 or 1"):
        read_submission_inputs(path, training=True)
    record["labels"]["readmission_30d"] = 0
    record["structured_features"]["age_years"] = float("nan")
    write_jsonl(path, [record])
    with pytest.raises(ValueError, match="age_years must be a nonnegative integer"):
        read_submission_inputs(path, training=True)
