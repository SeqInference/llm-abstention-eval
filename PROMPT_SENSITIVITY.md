# Prompt sensitivity: frozen calibration transfer

Status: implemented; model-backed results pending. Original experiments remain unchanged.

Question: does a correctness calibrator learned under one answer instruction transfer to two predefined alternative instructions?

## Frozen protocol

- Same pinned SmolLM2-135M-Instruct and 1.7B-Instruct checkpoints and MMLU revision as the published cross-model run; exact SHAs are recorded by the runner.
- Seed 42; independently shuffle official validation/test indices as in evaluate.py. Sample order is shared across variants and models for the same limit.
- Original: `Choose the best answer.`
- Direct: `Select the correct option.`
- Careful: `Read the question carefully and select the best option.`
- Only the answer instruction changes. Question, options, A-D order and confidence elicitation prompt stay fixed. Each variant chooses its own answer and correctness is scored against the original label.
- Original validation only: fit separate logistic maps for answer-option probability and elicited confidence. Freeze coefficients for all three test variants. Do not refit or select prompts using test performance.
- Constant reference is original-validation correctness rate, fixed across test variants for a given model.
- Report binary Brier/log loss on correctness confidence, fixed-answer accuracy per variant, error at 25/50/75/100% coverage, answer-change rate and mean absolute raw-signal changes relative to original test.
- Primary comparisons: frozen-calibrated answer-option Brier difference between each alternative and original test; accepted-error differences at 25% coverage are secondary/descriptive. Report all variants, including negative findings.

## PowerShell

Use the existing home virtual environment. No new packages are required beyond the original requirements.

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" -m unittest test_prompt_sensitivity -v
& "$env:USERPROFILE\.venv\Scripts\python.exe" prompt_sensitivity.py --model 135m --limit 20 --output results/prompt-smoke-135m.json
```

A very small validation sample may contain one correctness class, preventing calibration. Predictions remain in the checkpoint; use a new output path and a larger sample if that occurs.

After checking execution, run both 200-row pilots separately:

```powershell
& "$env:USERPROFILE\.venv\Scripts\python.exe" prompt_sensitivity.py --model 135m --limit 200 --output results/prompt-pilot-135m.json
& "$env:USERPROFILE\.venv\Scripts\python.exe" prompt_sensitivity.py --model 1.7b --limit 200 --output results/prompt-pilot-1.7b.json
```

To resume an interrupted pilot, repeat the exact command with `--resume`. Each output has a sibling `.checkpoint.json`. Model, code, packages, seed, device and sample size must match; do not edit the checkpoint. Each checkpoint write uses a temporary file and atomic replacement. Only one process may use an output/checkpoint pair. Do not start simultaneous runs on this CPU.

Final outputs are never overwritten. A completed run can be read without rerunning. Do not upload intermediate checkpoints as completed results.

## Interpretation

Raw prompts are used, not a chat template. This is a controlled instruction sensitivity check, not a comprehensive prompt robustness study. Confidence buckets create ties broken by seeded sample order. Rankings select each method's own examples retrospectively, not a deployable threshold. Negative logistic slopes reverse rankings and must be reported. Alternative prompts may change answers and hence correctness labels. One seed and two related checkpoints do not establish general robustness or a causal size effect. A lower Brier score alone does not establish better calibration independently of resolution.

The original 1,000-question test sample has already been inspected in earlier work. A larger follow-up on those questions is a descriptive extension, not a fresh confirmatory holdout. Pilot examples also overlap a later larger run with the same seed. Keep prompts/protocol fixed after the pilot, or explicitly document any changes.
