# Privacy-Preserving Clinical AI and Federated Learning



## 1. Executive summary

The solution finds personal information in synthetic clinical notes, replaces it
with placeholders, extracts clinical facts, and predicts 30-day readmission.
It compares local, federated and centralized models. The final predictions come
from a federated model with a secure aggregation prototype.

On 30 public validation cases, the solution scored **36.97 out of 40**. Both
de-identification and clinical extraction scored **1.0000**. The evaluator's
readmission score was **0.6968**, with AUROC **0.7292** and Brier score **0.1498**.

All **43 tests passed** locally and in Docker. The system runs on a CPU without
runtime internet access. These results apply to a small synthetic dataset. Some
text rules were improved after checking public validation errors, and the privacy
prototype is not ready for real hospital deployment.

## 2. System architecture

The pipeline has four main steps:

1. Find personal information and replace it with labels such as `[PATIENT_NAME]`.
2. Extract diagnoses, medicines, measurements, smoking status and allergies.
3. Combine selected extracted facts with the supplied structured features.
4. Use the trained model to produce a readmission probability.

The model has 28 coefficients, including an intercept. Its inputs cover age,
sex, previous admissions, length of stay, emergency admission, diagnoses,
measurements, smoking status and indicators for missing values. Names, IDs,
medications and hospital identifiers are not prediction features.

Scaling uses fixed reference values. Missing measurements use a fixed replacement
value and a separate flag showing that the measurement was missing. Gold clinical
labels are not used as model inputs.

Each simulated hospital keeps its own training rows. The federated server combines
model updates. A separate centralized experiment pools rows for comparison.
Everything runs in one Python process, so this is a simulation of separate hospitals.

## 3. De-identification

The code uses text patterns and nearby words to find all eight required types of
personal information: patient name, date of birth, encounter date, address, phone
number, patient ID, clinician name and email.

Each result includes the exact start and end character positions. Repeated names
are also detected. The original rendering function replaces each span with its
label, and submitted spans cannot overlap.

Context helps preserve useful clinical information. For example, phone detection
looks for a contact-related word so it does not mistake blood pressure for a phone
number. Tests exposed unnecessary removal of phrases such as “Patient denies chest
pain”; the rules were tightened to handle those cases.

All **255 annotated entities** in public validation were matched exactly, with no
missed or extra spans. Unseen names and note formats can still cause errors.

For scanned documents, a future version would use OCR to read text and retain its
position on each page. Detected information would then be removed from the image,
hidden text layers and metadata. This extension has not been implemented.

## 4. Structured extraction and standardization

The extraction rules map different expressions to the required standard terms.
For example, AF maps to `atrial_fibrillation`, Eliquis to `apixaban`, and Lasix
to `furosemide`.

The rules check nearby context to exclude ruled-out diagnoses, family history,
uncertain conditions and stopped medicines. They also handle decimal commas,
missing values and unit conversion:

- Creatinine in µmol/L is divided by 88.4 to obtain mg/dL.
- Hemoglobin in g/L is divided by 10 to obtain g/dL.
- Missing measurements and undocumented categories are returned as `null`.

Diagnosis and medication F1 scores were both **1.0000**. All numeric values were
within the evaluator's allowed errors. Smoking and allergy results were correct
for all 30 cases.

Public error analysis revealed a missing `unstable angina` synonym. It was added
as an acute coronary syndrome term. The final extraction score therefore reflects
development on the released data, not an untouched test result. Complex sentences
and changes in a patient's condition over time remain difficult for these rules.

## 5. Federated-learning experiment

Three approaches use the same logistic regression model:

- **Local:** each hospital trains a model using only its own cases.
- **Federated:** hospitals train locally and combine their updates using FedAvg.
- **Centralized:** one model trains on all hospitals' pooled cases as a reference.

For internal validation, 25% of each hospital's training cases are held out.
The fitting/holdout counts are 31/11 for Berlin and 29/10 each for Chennai and
Hyderabad. The split seed is 2026. Final training uses all 120 cases: 42 from
Berlin and 39 each from Chennai and Hyderabad.

FedAvg runs for **200 rounds**, with **two local updates per round**. Each local
update uses all of that hospital's fitting cases. The learning rate is 0.15 and
the L2 regularization strength is 0.05. The server weights updates by the number
of training cases, so larger hospitals contribute proportionally more.

Local and centralized models also receive 400 update steps. They use the same
features, data splits and initial weights. Two local steps between aggregations
can still produce a different result from two centralized steps.

On the 31-case internal holdout, AUROC was 0.7000 for local models, 0.7238 for
FedAvg and 0.7286 for the centralized model. The final public comparison was:

| Model | AUROC ↑ | Average precision ↑ | Brier score ↓ |
|---|---:|---:|---:|
| Local | 0.7361 | 0.5524 | 0.1404 |
| FedAvg | 0.7292 | 0.5027 | 0.1498 |
| Secure FedAvg, submitted model | 0.7292 | 0.5027 | 0.1498 |
| Centralized | 0.7292 | 0.5027 | 0.1498 |

AUROC measures ranking quality; average precision summarizes performance on
positive cases. Brier score measures probability error. Arrows show which
direction is better. Local models performed slightly better on public validation,
so these results do not show that federation always improves performance.

The hospitals have different data distributions. Training outcome rates are
35.7% in Berlin, 43.6% in Chennai and 23.1% in Hyderabad. Public AUROC for the
submitted model is 0.7619, 1.0000 and 0.1111 respectively. Hyderabad performs
poorly, but its validation set contains only one positive case among ten.

Predicted risks are also too high on average: the mean probability is 0.324,
while the observed outcome rate is 0.200. No public-label recalibration was used.

The full-data training loss decreased from 0.6894 to 0.5260. Initialization seeds
7, 19 and 43 gave very similar results; FedAvg's internal Brier standard deviation
was about 0.000009. This shows stable optimization, not reliable performance on
every possible data split.

The standard submission command never reads public validation labels. Its model
comparisons use the training holdout. Text rules were developed using all released
training annotations, so that holdout is not an independent test of text extraction.

## 6. Privacy extension and threat model

The additional mechanism is a **secure aggregation prototype**. It aims to hide
each hospital's individual model update from a server that follows the protocol
but tries to learn from the messages it receives.

Each hospital adds random masks to its weighted update. Hospitals use matching
masks with opposite signs, so the masks cancel when the server adds the messages.
The server can then recover the combined update.

The prototype assumes private mask setup, correct participation by all three
hospitals, and at least two hospitals that do not share their secrets with the
server. It uses fresh operating-system randomness, a scale of `100,000,000` for
integer encoding, and arithmetic modulo `2^61 - 1`. Update bounds are checked to
prevent overflow. The masking idea follows [Bonawitz et al., 2017](https://acmccs.github.io/papers/p1175-bonawitzA.pdf);
the full protocol from that paper is not implemented.

In the recorded comparison, ordinary FedAvg took about **0.030 seconds** and
secure FedAvg about **0.100 seconds**. The largest prediction difference was only
**1.37 × 10⁻¹⁰**. Each client uploads a 224-byte update vector per round, excluding
setup and other messages.

The protection has important limits:

- The trusted simulation can inspect every client's memory and masks.
- Real key exchange, authenticated communication and dropout recovery are absent.
- Aggregates and final predictions may still reveal information.
- Training statistics and offline evaluation labels/probabilities are outside
  the masking protection.
- It provides **no differential privacy or patient-level privacy guarantee**.

Hospital-specific prediction features were removed because their aggregate
updates could reveal one hospital's contribution. Rare clinical features can
still create similar risks. Keeping data at hospitals alone does not guarantee
privacy.

## 7. Reproducibility and testing

The solution uses Python 3.11, pinned package versions, fixed training seeds and
one numerical processing thread. It needs no GPU, downloaded model or runtime
network connection. Random masks change between runs but cancel during aggregation.
Small floating-point and timing differences can occur across machines.

Recorded checks include:

- **43 tests passed** locally and inside Docker.
- The candidate independently reran the local tests: **43 passed in 7.54 seconds**.
- The candidate ran submission twice, producing 30 predictions in **1.23 and
  1.19 seconds**, then reproduced **36.97/40** with the official evaluator.
- During development, Docker inference and evaluation passed with networking disabled, a four-CPU
  limit and a 16 GB memory limit. The image size was **0.576 GB**.
- Windows and Docker predictions matched exactly in the recorded comparison.

Submission timings exclude interpreter and import startup. Tests cover extraction,
character offsets, preservation of clinical content,
federated weighting, privacy arithmetic, invalid inputs and command-line output.
The CLI also handles empty evaluation files and rejects duplicate IDs, missing
training hospitals and unsafe output paths.

See [README.md](README.md) for commands, [verification.json](results/verification.json)
for recorded checks, and [AI_USAGE.md](AI_USAGE.md) for contribution details.

## 8. Limitations and next steps

The dataset is small and synthetic, with only six positive outcomes in public
validation. Its repeated note styles are simpler than real clinical records.
High scores here do not establish clinical safety or hidden-test performance.

The implementation can miss unfamiliar wording and complex negation. Its linear
prediction model has uneven site performance and imperfect probability estimates.
The privacy mechanism is a working arithmetic prototype with a simulated trust
boundary.

Next steps are to test unseen note formats, evaluate more data splits, investigate
Hyderabad errors, and assess calibration using training data only. Real deployment
would also require separate hospital services, a complete secure aggregation
protocol, protected evaluation and a patient-level privacy assessment. Scanned
documents and images would need their own implementation and tests.
