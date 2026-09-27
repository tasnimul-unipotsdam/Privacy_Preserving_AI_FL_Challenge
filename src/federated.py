"""Small cross-silo FedAvg simulation with explicit client/server boundaries.

Clients retain their own feature rows and labels. The orchestration process is
trusted: Python objects are not process or security isolation. Only the separate
centralized reference deliberately pools training data.
"""
from collections import namedtuple
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, SITES, feature_matrix
from src.privacy import aggregate_masked_updates, create_client_masks, mask_weighted_update


Config = namedtuple(
    "Config", ["rounds", "local_epochs", "learning_rate", "l2"],
    defaults=[200, 2, 0.15, 0.05],
)


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def objective(x, y, weights, l2):
    scores = x @ weights
    return float(np.mean(np.logaddexp(0.0, scores) - y * scores)
                 + 0.5 * l2 * np.dot(weights[1:], weights[1:]))


def gradient_steps(x, y, weights, steps, config):
    result = weights.copy()
    for _ in range(steps):
        gradient = x.T @ (sigmoid(x @ result) - y) / len(y)
        gradient[1:] += config.l2 * result[1:]
        result -= config.learning_rate * gradient
    return result


def metrics(y, probabilities):
    if not len(y):
        return {"n": 0, "prevalence": None, "roc_auc": None,
                "average_precision": None, "brier_score": None, "log_loss": None}
    return {
        "n": len(y), "prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)) if np.any(y) else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


class Client:
    def __init__(self, site, records, validation=None):
        self.site = site
        self._x = feature_matrix(records)
        self._y = np.array([record["labels"]["readmission_30d"] for record in records], dtype=float)
        self._validation_x = feature_matrix(validation or [])
        self._validation_y = np.array(
            [record["labels"]["readmission_30d"] for record in validation or []], dtype=float)
        self.count = len(records)
        if not self.count:
            raise ValueError(f"No training cases for {site}")

    def update(self, weights, config, mask=None):
        delta = gradient_steps(self._x, self._y, weights, config.local_epochs, config) - weights
        if mask is not None:
            return mask_weighted_update(delta, self.count, mask)
        return delta

    def train_local(self, initial, config):
        return gradient_steps(self._x, self._y, initial,
                              config.rounds * config.local_epochs, config)

    def loss(self, weights, config):
        return objective(self._x, self._y, weights, config.l2)

    def evaluate(self, weights):
        # For offline evaluation ONLY, labels and predictions leave each client.
        # This diagnostic channel is not part of the privacy claim.
        return self._validation_y.copy(), sigmoid(self._validation_x @ weights)

    def statistics(self):
        return {"n": self.count, "positive": int(self._y.sum()),
                "prevalence": float(self._y.mean()),
                "mean_age_years": float(self._x[:, 1].mean() * 20 + 60),
                "mean_prior_admissions_12m": float(self._x[:, 3].mean() * 2),
                "emergency_fraction": float(self._x[:, 5].mean())}


def initial_weights(seed):
    return np.random.default_rng(seed).normal(0.0, 0.01, len(FEATURE_NAMES))


def train_federated(clients, seed, config, secure=False):
    weights = initial_weights(seed)
    counts = [client.count for client in clients]
    history = []
    started = perf_counter()
    for round_number in range(config.rounds + 1):
        if round_number % 20 == 0 or round_number == config.rounds:
            # Aggregate scalar diagnostics; never aggregate feature rows.
            loss = sum(client.count * client.loss(weights, config) for client in clients) / sum(counts)
            history.append({"round": round_number, "training_objective": loss})
        if round_number == config.rounds:
            break
        if secure:
            # Ideal private setup stands in for pairwise authenticated channels.
            # The simulated server below receives masked updates and public counts.
            masks = create_client_masks(len(clients), weights.shape)
            messages = [client.update(weights, config, mask)
                        for client, mask in zip(clients, masks)]
            del masks
            delta = aggregate_masked_updates(messages, counts)
        else:
            deltas = [client.update(weights, config) for client in clients]
            delta = np.average(deltas, axis=0, weights=counts)
        weights += delta
    return weights, history, perf_counter() - started


def evaluate_clients(clients, weights):
    by_site, all_y, all_probabilities = {}, [], []
    for client in clients:
        model = weights[client.site] if isinstance(weights, dict) else weights
        y, probabilities = client.evaluate(model)
        by_site[client.site] = metrics(y, probabilities)
        all_y.extend(y)
        all_probabilities.extend(probabilities)
    return {"overall": metrics(np.asarray(all_y), np.asarray(all_probabilities)), "by_site": by_site}


def partition_records(records):
    groups = {site: [] for site in SITES}
    for record in sorted(records, key=lambda item: item["case_id"]):
        groups[record["hospital_id"]].append(record)
    if any(not group for group in groups.values()):
        raise ValueError("Training requires cases from all three hospital nodes")
    return groups


def holdout_split(groups):
    fitting, validation = {}, {}
    for site, records in groups.items():
        y = [record["labels"]["readmission_30d"] for record in records]
        # Small custom datasets can have only one class; never invent an AUC.
        counts = np.bincount(y, minlength=2)
        stratify = y if min(counts) >= 2 and len(records) >= 8 else None
        if len(records) < 2:
            fitting[site], validation[site] = records, []
        else:
            fitting[site], validation[site] = train_test_split(
                records, test_size=0.25, random_state=2026, stratify=stratify)
    return fitting, validation


def run_experiments(records, config=Config()):
    groups = partition_records(records)
    fitting, validation = holdout_split(groups)
    clients = [Client(site, fitting[site], validation[site]) for site in SITES]
    seeds = [7, 19, 43]
    runs, privacy_runs = [], []
    # Pooling is intentional ONLY for this reference experiment.
    pooled = [record for site in SITES for record in fitting[site]]
    central_x = feature_matrix(pooled)
    central_y = np.asarray([record["labels"]["readmission_30d"] for record in pooled], dtype=float)
    for seed in seeds:
        local = {client.site: client.train_local(initial_weights(seed), config) for client in clients}
        federated, history, plain_seconds = train_federated(clients, seed, config)
        centralized = gradient_steps(central_x, central_y, initial_weights(seed),
                                     config.rounds * config.local_epochs, config)
        secure, _, secure_seconds = train_federated(clients, seed, config, secure=True)
        runs.append({"seed": seed, "local": evaluate_clients(clients, local),
                     "federated": evaluate_clients(clients, federated),
                     "centralized": evaluate_clients(clients, centralized),
                     "secure_federated": evaluate_clients(clients, secure), "convergence": history})
        plain_p = np.concatenate([client.evaluate(federated)[1] for client in clients])
        secure_p = np.concatenate([client.evaluate(secure)[1] for client in clients])
        privacy_runs.append({"seed": seed, "plain_training_seconds": plain_seconds,
                             "secure_training_seconds": secure_seconds,
                             "max_weight_difference": float(np.max(np.abs(federated - secure))),
                             "max_probability_difference": float(np.max(np.abs(plain_p - secure_p)))
                             if len(plain_p) else 0.0})

    # Final inference uses secure FedAvg, retrained on every released training row.
    final_clients = [Client(site, groups[site]) for site in SITES]
    final_weights, final_history, final_seconds = train_federated(final_clients, seeds[0], config, secure=True)
    first = runs[0]
    summary = {
        "implementation": "sample_weighted_fedavg_logistic_regression",
        "random_seeds": seeds, "features": FEATURE_NAMES,
        "preprocessing": "Fixed scales and vocabulary; missing numeric values map to center plus missing indicator; no fitted pooled preprocessing.",
        "hyperparameters": config._asdict(),
        "selection": "Optimizer hyperparameters fixed before public validation scoring. Site indicators removed after privacy review; terminology refined through public error analysis. No validation-label access in this entry point.",
        "validation_source": "25% site-stratified holdout from --train; random_state=2026. Public validation labels are not read.",
        "split_sizes": {site: {"train": len(fitting[site]), "validation": len(validation[site])} for site in SITES},
        "local_models": {"algorithm": "L2 logistic regression per site", "validation": first["local"],
                         "routing": "Each validation case uses its own site's local model."},
        "federated_model": {"algorithm": "FedAvg", "rounds": config.rounds,
                            "local_epochs": config.local_epochs, "optimizer": "full-batch gradient descent",
                            "client_weighting": "number of local training cases / total training cases",
                            "validation": first["federated"]},
        "centralized_model": {"algorithm": "L2 logistic regression", "validation": first["centralized"],
                              "fairness": f"Same features, split, objective, seed, learning rate and {config.rounds * config.local_epochs} gradient steps; pooled gradient differs from multi-step local FedAvg."},
        "seed_runs": runs,
        "seed_stability": {model: {
            metric: {"mean": float(np.mean([run[model]["overall"][metric] for run in runs])),
                     "std": float(np.std([run[model]["overall"][metric] for run in runs]))}
            for metric in ("brier_score", "log_loss") if first[model]["overall"][metric] is not None}
            for model in ("local", "federated", "centralized")},
        "communication": {
            "raw_rows_transferred_in_federated_training": False,
            "plain_client_to_server": "Float64 model delta and sample count; scalar objective diagnostics.",
            "secure_client_to_server": "Masked fixed-point model delta and public sample count; scalar objective diagnostics.",
            "server_to_client": "Global coefficient vector each round.",
            "parameters": len(FEATURE_NAMES),
            "vector_bytes_per_client_per_round": 8 * len(FEATURE_NAMES),
            "all_client_upload_bytes": config.rounds * len(SITES) * 8 * len(FEATURE_NAMES),
            "evaluation_disclosure": "Offline diagnostics collect held-out labels/probabilities centrally for exact AUC/AP. This is outside the privacy claim and would need a separate protected evaluation protocol in deployment.",
        },
        "non_iid_analysis": {"site_statistics": {client.site: client.statistics() for client in final_clients},
                             "observation": "Site prevalence and covariates differ. Small per-site holdouts make AUC and worst-site comparisons unstable."},
        "final_model": {"method": "secure FedAvg", "seed": seeds[0], "n_train": len(records),
                        "convergence": final_history, "training_seconds": final_seconds},
        "limitations": ["Single-process simulation, not a distributed security boundary.",
                        "No formal patient-level privacy guarantee.",
                        "Initialization seeds measure optimizer stability, not split or population uncertainty.",
                        "Only 120 synthetic training cases; no clinical validity; unseen text formats can fail."],
    }
    privacy_comparison = {"seed_runs": privacy_runs,
                          "plain_validation": first["federated"],
                          "secure_validation": first["secure_federated"],
                          "final_secure_training_seconds": final_seconds}
    return final_weights, summary, privacy_comparison
