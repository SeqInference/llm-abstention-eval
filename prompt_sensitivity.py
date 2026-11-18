"""Frozen calibration transfer across three predefined answer instructions."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import statistics

from evaluate import score

PROMPTS = {
    'original': 'Choose the best answer.',
    'direct': 'Select the correct option.',
    'careful': 'Read the question carefully and select the best option.',
}
SIGNALS = ('option_probability', 'elicited_confidence')
MODELS = {
    '135m': ('HuggingFaceTB/SmolLM2-135M-Instruct', '12fd25f77366fa6b3b4b768ec3050bf629380bac'),
    '1.7b': ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '31b70e2e869a7173562077fd711b654946d38674'),
}
DATA_REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def indices(size, seed, limit):
    result = list(range(size))
    random.Random(seed).shuffle(result)
    return result[:limit] if limit else result


def collect(data, order, instruction, choices, rows, checkpoint, progress_every=10):
    """Resume a validated prefix; checkpoint after every completed example."""
    if len(rows) > len(order):
        raise ValueError('Checkpoint exceeds sample size')
    for i, row in enumerate(rows):
        if row['index'] != order[i] or row['tie_key'] != i:
            raise ValueError('Checkpoint is not a prefix of this sample')
    for key in range(len(rows), len(order)):
        index = order[key]
        example = data[index]
        options = '\n'.join(f'{letter}. {option}' for letter, option in zip('ABCD', example['choices']))
        question = f"{example['question']}\n{options}"
        probabilities = choices(f'{question}\n{instruction}\nAnswer:')
        selected = max(range(4), key=probabilities.__getitem__)
        buckets = choices(
            f"{question}\nSelected answer: {'ABCD'[selected]}\n"
            'How likely is the selected answer to be correct?\n'
            'A. 25%\nB. 50%\nC. 75%\nD. 95%\nConfidence category:')
        rows.append(dict(index=index, tie_key=key, selected=selected,
                         correct=int(selected == example['answer']),
                         option_probability=probabilities[selected],
                         elicited_confidence=(.25, .5, .75, .95)[max(range(4), key=buckets.__getitem__)]))
        checkpoint()
        if (key + 1) % progress_every == 0 or key + 1 == len(order):
            print(f'  {key + 1}/{len(order)} completed', flush=True)


def analyze(validation, variants):
    from sklearn.linear_model import LogisticRegression
    if len({r['correct'] for r in validation}) != 2:
        raise ValueError('Original validation requires correct and incorrect answers; increase --limit')
    original = variants['original']
    if not original:
        raise ValueError('Empty test sample')
    fits = {}
    for signal in SIGNALS:
        estimator = LogisticRegression().fit([[r[signal]] for r in validation], [r['correct'] for r in validation])
        fits[signal] = dict(slope=float(estimator.coef_[0, 0]), intercept=float(estimator.intercept_[0]))
    prior = statistics.mean(r['correct'] for r in validation)
    report = {}
    for name, rows in variants.items():
        if [(r['index'], r['tie_key']) for r in rows] != [(r['index'], r['tie_key']) for r in original]:
            raise ValueError('Test variants must have identical ordered samples')
        metrics = {}
        for signal, fit in fits.items():
            probabilities = [1 / (1 + math.exp(-(fit['slope'] * r[signal] + fit['intercept']))) for r in rows]
            metrics[signal] = {
                'raw': score([(r[signal], r['correct'], r['tie_key']) for r in rows]),
                'frozen_calibrated': score([(p, r['correct'], r['tie_key']) for p, r in zip(probabilities, rows)]),
                'mean_absolute_signal_change': statistics.mean(abs(r[signal] - b[signal]) for r, b in zip(rows, original)),
            }
        metrics['constant'] = score([(prior, r['correct'], r['tie_key']) for r in rows])
        metrics['answer_change_rate'] = statistics.mean(r['selected'] != b['selected'] for r, b in zip(rows, original))
        report[name] = metrics
    return dict(calibrators=fits, original_validation_rate=prior, variants=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='135m')
    parser.add_argument('--limit', type=int, default=200, help='Rows per split; 0 uses all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.output.exists():
        parser.error('Final output already exists; choose another path')
    checkpoint_path = args.output.with_suffix('.checkpoint.json')
    packages = {p: version(p) for p in ('torch', 'transformers', 'datasets', 'scikit-learn', 'huggingface-hub')}
    model_id, revision = MODELS[args.model]
    config = dict(protocol=1, model=model_id, model_revision=revision, dataset='cais/mmlu',
                  data_revision=DATA_REVISION, subject='all', limit=args.limit, seed=args.seed,
                  device=args.device, prompts=PROMPTS, package_versions=packages,
                  script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    if checkpoint_path.exists():
        if not args.resume:
            parser.error('Checkpoint exists; use --resume or choose another output path')
        state = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if state['configuration'] != config:
            parser.error('Checkpoint configuration, code, or package versions differ')
    else:
        if args.resume:
            parser.error('No checkpoint to resume')
        state = dict(configuration=config, validation=[], test={name: [] for name in PROMPTS})
        save(checkpoint_path, state)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    print(f'Loading {model_id} at {revision}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype='auto').to(args.device).eval()

    def choices(prompt):
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        prefix = inputs.input_ids[0].tolist()
        suffixes = [tokenizer(prompt + ' ' + label)['input_ids'] for label in 'ABCD']
        if any(ids[:-1] != prefix for ids in suffixes) or len({ids[-1] for ids in suffixes}) != 4:
            raise ValueError('Answer labels must be distinct single appended tokens')
        with torch.inference_mode():
            return model(**inputs).logits[0, -1, [ids[-1] for ids in suffixes]].float().softmax(dim=0).tolist()

    for split in ('validation', 'test'):
        data = load_dataset('cais/mmlu', 'all', split=split, revision=DATA_REVISION)
        order = indices(len(data), args.seed, args.limit)
        variants = {'original': state['validation']} if split == 'validation' else state['test']
        for name, rows in variants.items():
            print(f'{split} / {name}: resuming at {len(rows)}/{len(order)}', flush=True)
            collect(data, order, PROMPTS[name], choices, rows, lambda: save(checkpoint_path, state))
    state['metrics'] = analyze(state['validation'], state['test'])
    save(args.output, state)
    print(f'Finished: {args.output}', flush=True)


if __name__ == '__main__':
    main()
