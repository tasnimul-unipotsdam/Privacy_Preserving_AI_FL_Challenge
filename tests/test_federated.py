"""Check FedAvg math, weighting, leakage boundaries, and privacy integration."""
import numpy as np
import pytest

from src.features import FEATURE_NAMES, feature_vector
from src.federated import Client, Config, gradient_steps, initial_weights, metrics, train_federated


def record(site, age, label):
    return {"case_id": f"{site}-{age}", "hospital_id": site,
            "note_text": "Diagnoses: hypertension. HR 80 bpm. No known drug allergies.",
            "structured_features": {"age_years": age, "sex": "female",
                                    "prior_admissions_12m": 1, "length_of_stay_days": 4,
                                    "emergency_admission": False},
            "labels": {"readmission_30d": label}}


def test_one_local_step_equals_pooled_gradient_with_unequal_clients():
    groups = [[record("BERLIN_NODE", 40, 0)],
              [record("CHENNAI_NODE", 50 + i, i % 2) for i in range(3)],
              [record("HYDERABAD_NODE", 60 + i, 1) for i in range(2)]]
    clients = [Client(group[0]["hospital_id"], group) for group in groups]
    rows = sum(groups, [])
    x = np.vstack([feature_vector(row) for row in rows])
    y = np.array([row["labels"]["readmission_30d"] for row in rows])
    config = Config(rounds=1, local_epochs=1)
    expected = gradient_steps(x, y, initial_weights(7), 1, config)
    actual, _, _ = train_federated(clients, 7, config)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    secure, _, _ = train_federated(clients, 7, config, secure=True)
    np.testing.assert_allclose(secure, expected, atol=1e-8)


def test_features_never_use_gold_labels():
    row = record("BERLIN_NODE", 50, 1)
    expected = feature_vector(row)
    row["labels"] = {"readmission_30d": 0, "extracted_clinical_data": {"heart_rate_bpm": 300}}
    np.testing.assert_array_equal(feature_vector(row), expected)
    assert len(expected) == len(FEATURE_NAMES)


def test_single_class_metrics_and_empty_client():
    assert metrics(np.zeros(3), np.full(3, 0.2))["roc_auc"] is None
    assert metrics(np.array([]), np.array([]))["n"] == 0
    with pytest.raises(ValueError, match="No training cases"):
        Client("BERLIN_NODE", [])


def test_local_updates_do_not_change_another_client():
    a = Client("BERLIN_NODE", [record("BERLIN_NODE", 30, 0)])
    b = Client("CHENNAI_NODE", [record("CHENNAI_NODE", 60, 1)])
    start = initial_weights(7)
    before = b.update(start, Config())
    a.train_local(start, Config(rounds=3))
    np.testing.assert_array_equal(b.update(start, Config()), before)
    np.testing.assert_array_equal(start, initial_weights(7))
