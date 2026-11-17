"""Compare two models on aligned validation and test samples."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import re

from evaluate import predict, summarize


def compare_runs(runs):
    for split in ("validation", "test"):
        orders = [[(row["index"], row["tie_key"]) for row in run[split]]
                  for run in runs]
        if not orders[0] or orders[0] != orders[1]:
            raise ValueError(f"Models must use the same ordered {split} sample")

    results = []
    for run in runs:
        validation, test = run["validation"], run["test"]
        if len({row["correct"] for row in validation}) < 2:
            raise ValueError("Calibration requires correct and incorrect validation answers")
        results.append({
            "metrics": summarize(validation, test),
            "confidence_bucket_counts": {
                split: dict(Counter(row["elicited_confidence"] for row in rows))
                for split, rows in run.items()
            },
            "predictions": run,
        })
    first, second = (run["test"] for run in runs)
    agreement = Counter((a["correct"], b["correct"]) for a, b in zip(first, second))
    return results, {
        "both_correct": agreement[1, 1],
        "first_only_correct": agreement[1, 0],
        "second_only_correct": agreement[0, 1],
        "both_incorrect": agreement[0, 0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--second-model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    for flag in ("first-revision", "second-revision", "data-revision"):
        parser.add_argument(f"--{flag}", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/cross-model.json"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.output.exists():
        parser.error("Output already exists. Choose a new path.")

    from huggingface_hub import HfApi

    hub = HfApi()
    data_revision = hub.dataset_info("cais/mmlu", revision=args.data_revision).sha
    models = [
        (args.first_model, hub.model_info(args.first_model, revision=args.first_revision).sha),
        (args.second_model, hub.model_info(args.second_model, revision=args.second_revision).sha),
    ]
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha or "")
           for sha in [data_revision, *(sha for _, sha in models)]):
        raise RuntimeError("Could not resolve immutable revisions")
    print(f"MMLU revision: {data_revision}", flush=True)
    runs = []
    for model, revision in models:
        print(f"Evaluating {model} at {revision}", flush=True)
        runs.append(predict(argparse.Namespace(
            model=model, model_revision=revision, data_revision=data_revision,
            subject="all", device=args.device, seed=args.seed, limit=args.limit,
        )))
    results, paired = compare_runs(runs)
    for result, (model, revision) in zip(results, models):
        result.update(model=model, revision=revision)
    report = {
        "dataset": "cais/mmlu", "data_revision": data_revision,
        "subject": "all", "seed": args.seed, "limit": args.limit,
        "device": args.device, "models": results, "paired_test_counts": paired,
        "package_versions": {p: version(p) for p in
                             ("torch", "transformers", "datasets", "scikit-learn", "huggingface-hub")},
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(payload)
    print(args.output)


if __name__ == "__main__":
    main()
