# LLM Abstention Evaluation

Can a small language model identify which of its own multiple-choice answers are likely to be wrong? This Sequential Inference experiment compares two confidence signals on the same answers.

## Run

From PowerShell with Python 3.12:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate.py --limit 200 --output results/pilot.json
```

The first run downloads SmolLM2-135M-Instruct and MMLU from Hugging Face. CPU is the default. The pilot samples up to 200 rows independently from each official split using a fixed seed. For a published run, record package versions and pass immutable Hugging Face commit IDs to both revision flags. Use `--limit 0` for the full splits.

## Protocol

1. Choose the answer by normalizing next-token scores over A–D. The highest option probability is the first confidence signal.
2. Present that same answer back to the model and elicit a confidence bucket using a second prompt. Choose the highest scoring bucket from 25%, 50%, 75%, and 95%. This is constrained elicitation.
3. Fit a separate logistic mapping from each confidence signal to answer correctness on the official validation split. Score both raw signals and both fitted mappings on the official test split. The constant reference predicts the validation correctness rate.
4. Report accuracy, Brier score, log loss, and accepted error at 25%, 50%, 75%, and 100% coverage. Ties use seeded sample order. A large tied bucket may straddle the cutoff.

The matched-coverage comparisons use retrospective rankings. They measure how well scores sort examples and do not choose a deployment threshold. A positive-slope logistic map preserves ranking, so better probability scores need not reduce accepted error at matched coverage. Accuracy is identical across signals because the answers are fixed. Confidence values are conditional on this prompt and its offered options. Public benchmark exposure and sensitivity to prompt wording limit interpretation.

## Pilot results

SmolLM2-135M-Instruct answered 39 of 200 held-out MMLU questions
correctly (19.5%). Calibrating the answer-option probabilities improved
Brier score and log loss relative to the raw scores. The constant
validation-rate baseline scored slightly better on both metrics.
When prompted for confidence, the model selected the 25% bucket on
198 of 200 test questions.

[View the pilot configuration, predictions, and metrics](artifacts/pilot.json).

The run used seed 42 and 200 questions per split. The model and dataset
revisions were both set to `main`.

Companion project: [Calibration Decision Demo](https://github.com/SeqInference/calibration-decision-demo).

## Cross-model follow-up

Compare SmolLM2-135M-Instruct and SmolLM2-1.7B-Instruct with the same prompt
templates, MMLU revision, seed, ordered question IDs, and coverage levels.
The default samples 1,000 questions per split, with a separate calibrator
for each confidence signal and model fit on validation only.

From the repository folder in PowerShell, using the existing environment:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" compare_models.py --limit 1000 --output results/cross-model.json
```

Use `--device cuda` with a CUDA-enabled PyTorch installation. The script
resolves Hugging Face revisions to immutable commit IDs before inference and
records them with package versions. To repeat a run, pass the recorded IDs
to `--first-revision`, `--second-revision`, and `--data-revision`.

The JSON contains per-question predictions, paired test correctness counts,
confidence bucket counts by split, Brier score, log loss, and accepted error
at matched coverage. The main comparison is the share of test questions
assigned to the 25% confidence bucket. The original pilot is linked above.
