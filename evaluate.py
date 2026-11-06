"""Measure whether two confidence signals rank the same LLM answers by correctness."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics


def score(records):
    ordered = sorted(records, key=lambda row: (-row[0], row[2]))
    if not ordered:
        raise ValueError("No predictions to evaluate")
    probabilities = [p for p, _, _ in ordered]
    labels = [y for _, y, _ in ordered]
    losses = [-math.log(max(1e-15, p if y else 1 - p))
              for p, y, _ in ordered]
    selective = []
    for fraction in (.25, .50, .75, 1.0):
        count = math.ceil(len(ordered) * fraction)
        selective.append({
            "coverage": count / len(ordered), "accepted": count,
            "error": 1 - statistics.mean(labels[:count]),
            "cutoff_confidence": probabilities[count - 1],
        })
    return {
        "accuracy": statistics.mean(labels),
        "brier": statistics.mean((p - y)**2 for p, y, _ in ordered),
        "log_loss": statistics.mean(losses),
        "selective": selective,
    }


def summarize(calibration, test):
    from sklearn.linear_model import LogisticRegression

    report = {}
    for signal in ("option_probability", "elicited_confidence"):
        estimator = LogisticRegression()
        estimator.fit([[row[signal]] for row in calibration],
                      [row["correct"] for row in calibration])
        fitted = estimator.predict_proba([[row[signal]] for row in test])[:, 1]
        report[signal] = {
            "raw": score([(r[signal], r["correct"], r["tie_key"]) for r in test]),
            "calibrated": score([(float(p), r["correct"], r["tie_key"])
                                 for p, r in zip(fitted, test)]),
            "calibrator": {"slope": float(estimator.coef_[0, 0]),
                           "intercept": float(estimator.intercept_[0])},
        }
    rate = statistics.mean(row["correct"] for row in calibration)
    report["constant"] = score([(rate, r["correct"], r["tie_key"]) for r in test])
    return report


def predict(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype="auto"
    ).to(args.device).eval()
    labels = "ABCD"

    def choices(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        prefix = inputs.input_ids[0].tolist()
        suffixes = [tokenizer(prompt + " " + letter)["input_ids"] for letter in labels]
        if any(ids[:-1] != prefix for ids in suffixes):
            raise ValueError("The tokenizer must encode each option as one appended token")
        token_ids = [ids[-1] for ids in suffixes]
        if len(set(token_ids)) != len(labels):
            raise ValueError("Answer labels must map to distinct tokens")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, token_ids].float()
        return logits.softmax(dim=0).tolist()

    outputs = {}
    for split in ("validation", "test"):
        data = load_dataset("cais/mmlu", args.subject, split=split,
                            revision=args.data_revision)
        indices = list(range(len(data)))
        random.Random(args.seed).shuffle(indices)
        if args.limit:
            indices = indices[:args.limit]
        rows = []
        for tie_key, index in enumerate(indices):
            example = data[index]
            options = "\n".join(f"{letter}. {option}" for letter, option in
                                zip(labels, example["choices"]))
            question = f"{example['question']}\n{options}"
            probabilities = choices(f"{question}\nChoose the best answer.\nAnswer:")
            selected = max(range(4), key=probabilities.__getitem__)
            buckets = (.25, .50, .75, .95)
            confidence = choices(
                f"{question}\nSelected answer: {labels[selected]}\n"
                "How likely is the selected answer to be correct?\n"
                "A. 25%\nB. 50%\nC. 75%\nD. 95%\nConfidence category:"
            )
            rows.append({
                "index": index, "tie_key": tie_key, "selected": selected,
                "correct": int(selected == example["answer"]),
                "option_probability": probabilities[selected],
                "elicited_confidence": buckets[max(range(4), key=confidence.__getitem__)],
            })
        outputs[split] = rows
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--data-revision", default="main")
    parser.add_argument("--subject", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=200, help="Rows per split; 0 uses all")
    parser.add_argument("--output", type=Path, default=Path("results/run.json"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    predictions = predict(args)
    if len({row["correct"] for row in predictions["validation"]}) < 2:
        raise ValueError("Calibration needs both correct and incorrect predictions; increase --limit")
    result = {
        "configuration": {**vars(args), "output": str(args.output)},
        "predictions": predictions,
        "metrics": summarize(predictions["validation"], predictions["test"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
