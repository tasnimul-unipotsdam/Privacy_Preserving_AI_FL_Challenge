"""Pairwise additive masking: a secure-aggregation arithmetic prototype.

This module separates simulated client masking from server aggregation. Setup
stands in for private, authenticated client-to-client channels; it is NOT a
deployed cryptographic protocol or a security boundary within this Python process.
All selected clients must participate, and masks must be fresh each round.
"""

import math
import secrets
import time
from numbers import Integral

import numpy as np


MODULUS = (1 << 61) - 1
DEFAULT_SCALE = 100_000_000
DEFAULT_MAX_ABS_UPDATE = 1000.0


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_bounds(counts, scale, max_abs_update):
    """Check a public worst-case bound, without inspecting client updates."""
    scale = _positive_integer(scale, "scale")
    counts = [_positive_integer(count, "client count") for count in counts]
    if not counts:
        raise ValueError("At least one client count is required")
    if not math.isfinite(max_abs_update) or max_abs_update <= 0:
        raise ValueError("max_abs_update must be finite and positive")
    numerator, denominator = float(max_abs_update).as_integer_ratio()
    # Ceil the public bound using integers; allow one rounding unit per client.
    bound = (numerator * sum(counts) * scale + denominator - 1) // denominator
    if bound + len(counts) >= MODULUS // 2:
        raise ValueError("Public update bound risks modular wraparound; reduce scale or bound")
    return counts, scale


def _check_message(message, shape):
    if message.shape != shape or message.dtype != np.uint64:
        raise ValueError("Masks and messages must have matching shapes and uint64 dtype")
    if np.any(message >= MODULUS):
        raise ValueError("Masks and messages must contain residues below the modulus")


def create_client_masks(num_clients, shape):
    """Simulate private pairwise setup, returning only each client's net mask.

    For every pair i < j, a fresh uniform vector is added at i and subtracted at
    j. Masks sum to zero modulo MODULUS. In a deployment each pair would create
    its shared randomness privately; the server must never receive these masks.
    This trusted simulator holds all masks and therefore must be outside the
    stated adversary's view. No secret seeds or masks are written to artifacts.
    """
    num_clients = _positive_integer(num_clients, "num_clients")
    if num_clients < 3:
        raise ValueError("This prototype requires at least three participating clients")
    if not shape or any(_positive_integer(size, "dimension") != size for size in shape):
        raise ValueError("A nonempty array shape is required")
    masks = [np.zeros(shape, dtype=np.uint64) for _ in range(num_clients)]
    size = math.prod(shape)
    for left in range(num_clients):
        for right in range(left + 1, num_clients):
            pair_mask = np.fromiter(
                (secrets.randbelow(MODULUS) for _ in range(size)), dtype=np.uint64, count=size
            ).reshape(shape)
            # Every intermediate is below 2 * MODULUS, so uint64 cannot overflow.
            masks[left] = (masks[left] + pair_mask) % MODULUS
            masks[right] = (masks[right] + MODULUS - pair_mask) % MODULUS
    return masks


def _round_weighted(value, weight):
    """Round a float's exact rational value times weight, with ties to even."""
    numerator, denominator = float(value).as_integer_ratio()
    quotient, remainder = divmod(numerator * weight, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2):
        quotient += 1
    return quotient


def mask_weighted_update(
    update,
    count,
    mask,
    *,
    scale=DEFAULT_SCALE,
    max_abs_update=DEFAULT_MAX_ABS_UPDATE,
):
    """Client operation: quantize n_i * update_i, then apply its private mask.

    The bound is checked, never silently clipped. It prevents arithmetic overflow
    and is not a differential-privacy clipping parameter. Only the returned uint64
    vector and public sample count should be sent to the server.
    """
    counts, scale = _check_bounds([count], scale, max_abs_update)
    update = np.asarray(update, dtype=np.float64)
    if update.size == 0 or update.ndim == 0 or not np.all(np.isfinite(update)):
        raise ValueError("Client updates must be nonempty finite arrays")
    if np.any(np.abs(update) > max_abs_update):
        raise ValueError("Client update exceeds the public max_abs_update bound")
    _check_message(mask, update.shape)
    weight = counts[0] * scale
    encoded = np.fromiter(
        (_round_weighted(value, weight) % MODULUS for value in update.flat),
        dtype=np.uint64,
        count=update.size,
    ).reshape(update.shape)
    return (encoded + mask) % MODULUS


def aggregate_masked_updates(
    messages,
    counts,
    *,
    scale=DEFAULT_SCALE,
    max_abs_update=DEFAULT_MAX_ABS_UPDATE,
):
    """Server operation: recover a weighted mean from masked messages only.

    The coordinator must supply exactly the roster used during setup, with one
    message per client. There is no dropout recovery, authentication, or replay
    defense here. Public checks cannot prove that a malicious client honored the
    bound or supplied its true count.
    """
    if len(messages) < 3 or len(messages) != len(counts):
        raise ValueError("At least three messages and one count per message are required")
    counts, scale = _check_bounds(counts, scale, max_abs_update)
    shape = messages[0].shape
    if not shape or messages[0].size == 0:
        raise ValueError("Messages must be nonempty arrays")
    total = np.zeros(shape, dtype=np.uint64)
    for message in messages:
        _check_message(message, shape)
        total = (total + message) % MODULUS
    signed = total.astype(np.int64)
    signed[total > MODULUS // 2] -= MODULUS
    return signed.astype(np.float64) / (scale * sum(counts))


def secure_weighted_average(
    updates,
    counts,
    *,
    scale=DEFAULT_SCALE,
    max_abs_update=DEFAULT_MAX_ABS_UPDATE,
):
    """Test/benchmark harness with access to clear updates; not the FL server."""
    if len(updates) < 3 or len(updates) != len(counts):
        raise ValueError("At least three updates and one count per update are required")
    counts, scale = _check_bounds(counts, scale, max_abs_update)
    started = time.perf_counter()
    arrays = [np.asarray(update, dtype=np.float64) for update in updates]
    masks = create_client_masks(len(arrays), arrays[0].shape)
    messages = [
        mask_weighted_update(update, count, mask, scale=scale, max_abs_update=max_abs_update)
        for update, count, mask in zip(arrays, counts, masks)
    ]
    result = aggregate_masked_updates(messages, counts, scale=scale, max_abs_update=max_abs_update)
    secure_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference = np.average(np.stack(arrays), axis=0, weights=counts)
    plain_seconds = time.perf_counter() - started
    quantization_bound = len(arrays) / (2 * scale * sum(counts))
    return result, {
        "secure_aggregation_seconds": secure_seconds,
        "plain_aggregation_seconds": plain_seconds,
        "max_absolute_error_vs_plain": float(np.max(np.abs(result - reference))),
        "quantization_error_bound": quantization_bound,
        "float64_roundoff_allowance": 4 * np.finfo(np.float64).eps * max_abs_update,
        "client_to_server_vector_bytes": sum(message.nbytes for message in messages),
        "ideal_pairwise_mask_bytes": len(arrays) * (len(arrays) - 1) // 2 * arrays[0].size * 8,
        "client_count": len(arrays),
        "total_sample_count": sum(counts),
        "scale": scale,
        "modulus": MODULUS,
    }


def privacy_summary():
    """Static threat model; the experiment adds its measured utility and cost."""
    return {
        "mechanism": "Pairwise additive masking of fixed-point weighted client updates",
        "implementation_status": "Working arithmetic prototype with idealized private setup, in one process",
        "protected_asset": "Each hospital's quantized model update within a single aggregation round",
        "adversary": "Honest-but-curious aggregation server, optionally colluding with one of three clients",
        "trust_assumptions": [
            "At least two participating clients do not collude with the server",
            "Private authenticated pairwise setup and uncompromised client randomness are assumed",
            "The simulator holding setup masks is trusted and outside the adversary's view",
            "Every selected client participates and follows the protocol with correct public counts",
            "A fresh independent mask is used for every pair, coordinate, and round",
        ],
        "configuration": {
            "modulus": MODULUS,
            "fixed_point_scale": DEFAULT_SCALE,
            "public_max_absolute_update": DEFAULT_MAX_ABS_UPDATE,
            "randomness": "secrets.randbelow, backed by operating-system cryptographic randomness",
            "numeric_encoding": "round(scale * client_count * update), with exact rational rounding",
            "message_dtype": "uint64",
            "differential_privacy_epsilon": None,
            "differential_privacy_delta": None,
        },
        "privacy_claim": (
            "Under ideal private uniform pairwise masks, masked messages reveal no additional "
            "information about individual encoded updates beyond their aggregate and the colluders' "
            "own information. This is a conditional arithmetic claim, not a deployment guarantee."
        ),
        "utility_and_cost": {
            "quantization_error_bound_per_coordinate": "num_clients / (2 * scale * total_sample_count), plus float64 decoding roundoff",
            "complexity": "O(num_clients^2 * num_parameters) setup; O(num_clients * num_parameters) aggregation",
            "message_cost": "8 bytes per model coordinate per client, plus public counts and setup costs",
            "empirical_comparison": "The training experiment adds measured plain-versus-masked model and runtime comparisons",
        },
        "remaining_attack_surface": [
            "No process isolation: a user controlling this simulator can read all client state and masks",
            "No key exchange, authenticated transport, secret erasure, replay defense, or dropout recovery",
            "No protection against a malicious server, malicious clients, poisoning, or false sample counts",
            "Server collusion with all but one client reveals the remaining update from the aggregate",
            "Aggregates and final predictions may still permit membership or attribute inference",
            "Repeated aggregates, auxiliary knowledge, and very small silos can expose information",
            "Hospital sample counts and model dimensions are public metadata",
            "No differential privacy, patient-level privacy guarantee, or clinical certification",
        ],
        "reproducibility": "Masks and timings vary; exact modular cancellation makes predictions independent of masks",
        "reference": {
            "title": "Bonawitz et al., Practical Secure Aggregation for Privacy-Preserving Machine Learning, CCS 2017",
            "url": "https://acmccs.github.io/papers/p1175-bonawitzA.pdf",
            "relationship": "Lemma 6.1 motivates the masking arithmetic; the full paper's cryptographic and dropout protocol is not implemented",
        },
    }
