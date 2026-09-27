# Privacy-Preserving Clinical AI and Federated Learning

An offline solution for the supplied synthetic challenge: contextual de-identification,
canonical clinical extraction, local/FedAvg/centralized logistic regression, and a
pairwise masking secure aggregation prototype. Final predictions use secure FedAvg.
This is a synthetic benchmark, not a clinically validated system.

The original challenge instructions are preserved in [CHALLENGE.md](CHALLENGE.md).
See [REPORT.md](REPORT.md) for results and limitations and [AI_USAGE.md](AI_USAGE.md)
for the development disclosure.

## Setup

Python 3.11 and CPU only are sufficient. No model downloads, credentials, or network
access are required at runtime. Dependencies are pinned to the versions tested in
the supplied `privacy-fl` environment and Linux Docker; no additional libraries
were needed beyond its existing dependency set.

```bash
conda activate privacy-fl
python -m pip install -r requirements.txt
```

For a fresh environment, run `conda create -n privacy-fl python=3.11` first.
Alternatively, create a Python 3.11 virtual environment and install the same file.

## Standard submission command

```bash
python run_submission.py --train data/train.jsonl --input data/validation_inputs.jsonl --output outputs/validation_predictions.jsonl --artifacts-dir outputs/artifacts
```

This command writes one JSON object per input case and the required
`experiment_summary.json` and `privacy_summary.json`. Paths are supplied explicitly;
hidden files need only follow the input schema. Evaluation labels are never read
by this entry point. The experiment summary compares models on a 25% holdout from
the supplied training file, then the final secure model is fitted on all 120
training cases. Empty evaluation files produce empty predictions; malformed records,
duplicate IDs, overlapping train/evaluation IDs, missing hospital training data,
and output/input path collisions produce a nonzero exit and an explanation.

## Validation and tests

```bash
python evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions outputs/validation_predictions.jsonl --report outputs/validation_report.json
python -m pytest -q
python scripts/analyze_validation.py --train data/train.jsonl --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --output outputs/model_comparison.json
```

The optional analysis script uses public labels only after fitting all comparison
models. It is separate from submission inference and reports local, federated,
secure federated, centralized and training-prevalence baseline metrics.

On systems with GNU Make, `make evaluate` runs the submission and official evaluator,
and `make test` runs the tests. The Python commands above work in PowerShell as well.
Committed aggregate evidence is in [results/](results/); generated predictions and
runtime artifacts under `outputs/` are ignored by Git.

## Docker

Start Docker Desktop (Linux containers) or the Docker engine, then build:

```bash
docker build -t privacy-fl-challenge .
```

Run without network access on Linux/macOS:

```bash
mkdir -p outputs
docker run --rm --network none --cpus 4 --memory 16g -v "$PWD/outputs:/outputs" privacy-fl-challenge --train data/train.jsonl --input data/validation_inputs.jsonl --output /outputs/validation_predictions.jsonl --artifacts-dir /outputs/artifacts
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
$outputPath = (Resolve-Path outputs).Path
docker run --rm --network none --cpus 4 --memory 16g --mount "type=bind,source=$outputPath,target=/outputs" privacy-fl-challenge --train data/train.jsonl --input data/validation_inputs.jsonl --output /outputs/validation_predictions.jsonl --artifacts-dir /outputs/artifacts
```

Public synthetic data are bundled. To evaluate another file, mount its containing
directory read-only at `/evaluation` and pass `--input /evaluation/hidden_inputs.jsonl`.
All required training and prediction operations run offline inside the image.

Tests and evaluator can also run inside the image:

```bash
docker run --rm --network none --entrypoint python privacy-fl-challenge -m pytest -q
docker run --rm --network none -v "$PWD/outputs:/outputs" --entrypoint python privacy-fl-challenge evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions /outputs/validation_predictions.jsonl --report /outputs/validation_report.json
```

Installation/image build needs access to package/image registries. Runtime does
not. The tested image is approximately 0.58 GB, well below the 8 GB limit.
