import copy
import tempfile
import unittest
from pathlib import Path
from prompt_sensitivity import analyze, collect, indices, save, PROMPTS


def rows():
    return [dict(index=i, tie_key=i, selected=i % 4, correct=i % 2,
                 option_probability=.3 + .05 * i, elicited_confidence=.25 if i % 2 == 0 else .75)
            for i in range(8)]


class Tests(unittest.TestCase):
    def test_sampling(self):
        self.assertEqual(indices(30, 42, 8), indices(30, 42, 0)[:8])
        self.assertEqual(len(set(indices(30, 42, 8))), 8)

    def test_identity_and_freeze(self):
        source = rows(); original = copy.deepcopy(source)
        result = analyze(source, {k: copy.deepcopy(source) for k in PROMPTS})
        self.assertEqual(source, original)
        self.assertEqual(result['variants']['original'], result['variants']['direct'])
        self.assertEqual(result['variants']['original']['answer_change_rate'], 0)

    def test_alignment(self):
        other = rows()[::-1]
        with self.assertRaises(ValueError): analyze(rows(), dict(original=rows(), direct=other))

    def test_single_class(self):
        val = rows()
        for r in val: r['correct'] = 0
        with self.assertRaises(ValueError): analyze(val, dict(original=rows()))

    def test_changed_answers_keep_their_own_labels(self):
        other = rows()
        for r in other: r['selected'] = (r['selected'] + 1) % 4; r['correct'] = 0
        result = analyze(rows(), dict(original=rows(), direct=other))['variants']['direct']
        self.assertEqual(result['answer_change_rate'], 1)
        self.assertEqual(result['option_probability']['raw']['accuracy'], 0)

    def test_resume_and_checkpoint(self):
        data = [dict(question='Q', choices=['x']*4, answer=0) for _ in range(3)]
        calls = []; snapshots = []; collected = []
        def choices(p): calls.append(p); return [.7, .1, .1, .1]
        collect(data, [2, 0], PROMPTS['original'], choices, collected, lambda: snapshots.append(copy.deepcopy(collected)))
        count = len(calls)
        collect(data, [2, 0], PROMPTS['original'], choices, collected, lambda: None)
        self.assertEqual(len(calls), count)
        self.assertEqual([len(s) for s in snapshots], [1, 2])
        self.assertEqual(collected[0]['correct'], 1)
        self.assertIn('Choose the best answer.', calls[0])
        with self.assertRaises(ValueError): collect(data, [0, 2], '', choices, collected, lambda: None)

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'state.json'; save(p, {'n':1}); save(p, {'n':2})
            self.assertIn('2', p.read_text()); self.assertFalse(p.with_suffix('.json.tmp').exists())

if __name__ == '__main__': unittest.main()
