"""Arithmetic correctness and failure boundaries of the privacy prototype."""

import json
from fractions import Fraction

import numpy as np
import pytest

from src.privacy import (
    MODULUS,
    aggregate_masked_updates,
    create_client_masks,
    mask_weighted_update,
    privacy_summary,
    secure_weighted_average,
)


def test_integer_masks_cancel_exactly_even_near_modulus(monkeypatch):
    monkeypatch.setattr("src.privacy.secrets.randbelow", lambda upper: upper - 1)
    masks = create_client_masks(3, (2,))
    total = sum((mask.astype(object) for mask in masks)) % MODULUS
    np.testing.assert_array_equal(total, [0, 0])
    updates = [np.array([-1.25, 0.5]), np.array([0.25, -2.0]), np.array([1.5, 2.0])]
    counts = [2, 3, 5]
    messages = [mask_weighted_update(x, n, m) for x, n, m in zip(updates, counts, masks)]
    actual = aggregate_masked_updates(messages, counts)
    np.testing.assert_allclose(actual, np.average(updates, axis=0, weights=counts), atol=1e-14)


def test_nonuniform_client_weighting_and_quantization_bound():
    updates = [
        np.array([[0.123456789, -0.25], [0.00001, 1.0]]),
        np.array([[0.654321987, -0.50], [0.00009, 2.0]]),
        np.array([[-0.456789123, 0.75], [-0.00004, 3.0]]),
    ]
    result, diagnostics = secure_weighted_average(updates, [7, 13, 19], scale=100)
    expected = np.average(updates, axis=0, weights=[7, 13, 19])
    bound = diagnostics["quantization_error_bound"] + diagnostics["float64_roundoff_allowance"]
    assert np.max(np.abs(result - expected)) <= bound
    assert diagnostics["client_to_server_vector_bytes"] == 3 * 4 * 8
    assert diagnostics["ideal_pairwise_mask_bytes"] == 3 * 4 * 8
    assert diagnostics["secure_aggregation_seconds"] >= 0


def test_fresh_masks_change_messages_but_not_aggregate():
    updates = [np.array([0.125, -0.2, 0.3])] * 3
    first_masks = create_client_masks(3, (3,))
    second_masks = create_client_masks(3, (3,))
    first = [mask_weighted_update(x, 1, m) for x, m in zip(updates, first_masks)]
    second = [mask_weighted_update(x, 1, m) for x, m in zip(updates, second_masks)]
    assert not np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(aggregate_masked_updates(first, [1] * 3), aggregate_masked_updates(second, [1] * 3))


def test_negative_rounding_and_ties_match_exact_rational_reference():
    values = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 0.1])
    zero_mask = np.zeros(values.shape, dtype=np.uint64)
    message = mask_weighted_update(values, 1, zero_mask, scale=2)
    expected = [round(Fraction(float(value)) * 2) % MODULUS for value in values]
    assert message.tolist() == expected


def test_server_needs_only_uint64_messages_and_public_counts():
    masks = create_client_masks(3, (2,))
    messages = [mask_weighted_update(np.array([i, -i]), 2, masks[i]) for i in range(3)]
    del masks
    # A byte serialization roundtrip represents the server's entire input.
    received = [np.frombuffer(message.tobytes(), dtype=np.uint64) for message in messages]
    np.testing.assert_allclose(aggregate_masked_updates(received, [2, 2, 2]), [1.0, -1.0])


@pytest.mark.parametrize("bad_count", [0, -1, 1.5, True])
def test_invalid_client_counts_are_rejected(bad_count):
    with pytest.raises(ValueError, match="positive integer"):
        mask_weighted_update(np.array([1.0]), bad_count, np.zeros(1, dtype=np.uint64))


@pytest.mark.parametrize("bad_update", [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_nonfinite_and_empty_updates_are_rejected(bad_update):
    with pytest.raises(ValueError, match="nonempty finite"):
        mask_weighted_update(bad_update, 1, np.zeros(bad_update.shape, dtype=np.uint64))


def test_public_bound_is_enforced_without_silent_clipping():
    with pytest.raises(ValueError, match="exceeds"):
        mask_weighted_update(np.array([1.01]), 1, np.zeros(1, dtype=np.uint64), max_abs_update=1)
    with pytest.raises(ValueError, match="wraparound"):
        aggregate_masked_updates([np.zeros(1, dtype=np.uint64)] * 3, [10**9] * 3)


def test_dropouts_and_malformed_messages_are_rejected():
    messages = [np.zeros(1, dtype=np.uint64)] * 3
    with pytest.raises(ValueError, match="three"):
        aggregate_masked_updates(messages[:2], [1, 1])
    with pytest.raises(ValueError, match="one count"):
        aggregate_masked_updates(messages, [1, 1])
    with pytest.raises(ValueError, match="matching shapes"):
        aggregate_masked_updates(messages[:2] + [np.zeros(2, dtype=np.uint64)], [1] * 3)
    with pytest.raises(ValueError, match="uint64"):
        aggregate_masked_updates([np.zeros(1)] * 3, [1] * 3)
    with pytest.raises(ValueError, match="residues"):
        aggregate_masked_updates([np.array([MODULUS], dtype=np.uint64)] * 3, [1] * 3)


def test_privacy_summary_is_json_and_explicit_about_scope():
    summary = privacy_summary()
    json.dumps(summary, allow_nan=False)
    assert summary["configuration"]["differential_privacy_epsilon"] is None
    assert "one process" in summary["implementation_status"]
    assert "not a deployment guarantee" in summary["privacy_claim"]
