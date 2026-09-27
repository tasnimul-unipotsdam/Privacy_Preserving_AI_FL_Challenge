#!/usr/bin/env python3
"""Compare fixed models on public labels after fitting, outside the submission CLI."""
import argparse
import json
from pathlib import Path
import sys

# Running `python scripts/analyze_validation.py` should work from any directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_submission import read_jsonl, write_json
from evaluator.evaluate import index_by_case, read_jsonl as read_truth_jsonl
from src.features import SITES, feature_matrix
from src.federated import (
    Client, Config, gradient_steps, initial_weights, metrics, partition_records,
    sigmoid, train_federated,
)

import numpy as np


def calibration_bins(y, probabilities):
    """Five fixed bins; empty bins remain visible and no calibrator is fitted."""
    bins = []
    edges = np.linspace(0.0, 1.0, 6)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == 4 else probabilities < upper)
        count = int(selected.sum())
        bins.append({
            "lower": float(lower), "upper": float(upper), "n": count,
            "mean_probability": float(probabilities[selected].mean()) if count else None,
            "observed_fraction": float(y[selected].mean()) if count else None,
        })
    return bins


def summarize(y, probabilities, sites):
    overall = metrics(y, probabilities)
    overall["mean_probability"] = float(probabilities.mean())
    return {
        "overall": overall,
        "by_site": {site: metrics(y[sites == site], probabilities[sites == site]) for site in SITES},
        "calibration_bins": calibration_bins(y, probabilities),
    }


def analyze(train_path, inputs_path, truth_path):
    training = read_jsonl(train_path, training=True)
    inputs = read_jsonl(inputs_path)
    if not inputs:
        raise ValueError("Validation inputs must contain at least one case")
    if {record["case_id"] for record in training} & {record["case_id"] for record in inputs}:
        raise ValueError("Training and validation case IDs overlap")

    config, seed = Config(), 7
    groups = partition_records(training)
    clients = [Client(site, groups[site]) for site in SITES]
    initial = initial_weights(seed)
    local_weights = {client.site: client.train_local(initial, config) for client in clients}
    federated, _, plain_seconds = train_federated(clients, seed, config)
    secure, _, secure_seconds = train_federated(clients, seed, config, secure=True)

    # Pooling is confined to the explicitly named centralized reference.
    pooled = [record for site in SITES for record in groups[site]]
    train_x = feature_matrix(pooled)
    train_y = np.asarray([record["labels"]["readmission_30d"] for record in pooled], dtype=float)
    centralized = gradient_steps(train_x, train_y, initial,
                                 config.rounds * config.local_epochs, config)

    input_x = feature_matrix(inputs)
    sites = np.asarray([record["hospital_id"] for record in inputs])
    local_probabilities = np.empty(len(inputs), dtype=float)
    for site in SITES:
        selected = sites == site
        local_probabilities[selected] = sigmoid(input_x[selected] @ local_weights[site])
    probabilities = {
        "local": local_probabilities,
        "federated": sigmoid(input_x @ federated),
        "secure_federated": sigmoid(input_x @ secure),
        "centralized": sigmoid(input_x @ centralized),
        "prevalence_baseline": np.full(len(inputs), float(train_y.mean())),
    }

    # Public labels are opened only after every model and prediction is fixed.
    truth = index_by_case(read_truth_jsonl(truth_path), "public validation ground truth")
    if set(truth) != {record["case_id"] for record in inputs}:
        raise ValueError("Validation input and ground-truth case IDs must match exactly")
    outcomes = []
    for record in inputs:
        label_record = truth[record["case_id"]]
        outcome = label_record.get("readmission_30d")
        if label_record.get("hospital_id") != record["hospital_id"]:
            raise ValueError(f"Hospital mismatch for {record['case_id']}")
        if type(outcome) is not int or outcome not in (0, 1):
            raise ValueError(f"Invalid binary outcome for {record['case_id']}")
        outcomes.append(outcome)
    y = np.asarray(outcomes, dtype=float)

    return {
        "purpose": "Offline public validation comparison; this script is not used by run_submission.py.",
        "selection": "Fixed default configuration and seed 7; labels read only after all models and predictions are fitted. No tuning or calibration fitting.",
        "training_cases": len(training), "validation_cases": len(inputs),
        "seed": seed, "hyperparameters": config._asdict(),
        "training_by_site": {client.site: client.statistics() for client in clients},
        "local_routing": "Each input is predicted by the local model trained at its own site.",
        "centralized_fairness": "Same training cases, features, objective, initialization, learning rate and number of gradient steps; only pooled versus local updates differ.",
        "models": {name: summarize(y, values, sites) for name, values in probabilities.items()},
        "privacy_comparison": {
            "max_weight_difference": float(np.max(np.abs(secure - federated))),
            "max_probability_difference": float(np.max(np.abs(probabilities["secure_federated"] - probabilities["federated"]))),
            "plain_training_seconds": plain_seconds, "secure_training_seconds": secure_seconds,
        },
        "limitations": [
            "Small public sample and few positives make site AUC and calibration bins unstable.",
            "Five calibration bins are descriptive; no recalibration was performed.",
            "Public metrics are exploratory estimates, not a hidden-test or clinical performance guarantee.",
            "Plaintext outcome-based diagnostics are outside the secure-aggregation privacy claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.jsonl")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/validation_inputs.jsonl")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "data/validation_ground_truth.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model_comparison.json")
    args = parser.parse_args()
    try:
        if args.output.resolve() in {path.resolve() for path in (args.train, args.inputs, args.ground_truth)}:
            raise ValueError("Output path must differ from the input files")
        result = analyze(args.train, args.inputs, args.ground_truth)
        write_json(args.output, result)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Validation analysis failed: {exc}\n")
    print(json.dumps({name: value["overall"] for name, value in result["models"].items()}, indent=2))
    print(f"Wrote public validation comparison to {args.output}")


if __name__ == "__main__":
    main()
