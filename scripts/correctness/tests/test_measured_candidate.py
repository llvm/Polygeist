import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'select_measured_candidate.py'
spec = importlib.util.spec_from_file_location('measured_candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MeasuredCandidateTest(unittest.TestCase):
    def profile(self, samples, correct=True):
        return dict(context={'input': 'arbitrary-shape', 'device': 'test-device'},
                    samples_ms=samples, correctness_passed=correct)

    def test_profitable_and_correct(self):
        result = module.choose(self.profile([10] * 5), self.profile([7, 8, 8, 8, 9]))
        self.assertEqual(result['selected'], 'candidate')

    def test_faster_but_incorrect_keeps_original(self):
        result = module.choose(self.profile([10] * 5), self.profile([1] * 5, False))
        self.assertEqual(result['selected'], 'baseline')

    def test_one_fast_outlier_does_not_authorize_change(self):
        result = module.choose(self.profile([10] * 5), self.profile([1, 11, 11, 11, 11]))
        self.assertEqual(result['selected'], 'baseline')

    def test_small_or_negative_benefit_keeps_original(self):
        for time in [9.9, 10, 12]:
            result = module.choose(self.profile([10] * 5), self.profile([time] * 5))
            self.assertEqual(result['selected'], 'baseline')

    def test_incomparable_or_incomplete_measurements_rejected(self):
        original = self.profile([10] * 5)
        candidate = self.profile([1] * 5)
        candidate['context'] = {'device': 'different'}
        with self.assertRaises(ValueError):
            module.choose(original, candidate)
        with self.assertRaises(ValueError):
            module.choose(original, self.profile([1]))
        with self.assertRaises(ValueError):
            module.choose(original, self.profile([float('nan')] * 5))

    def test_unvalidated_baseline_cannot_authorize_selection(self):
        for invalid in [False, None, 'false', 1]:
            with self.assertRaises(ValueError):
                module.choose(self.profile([10] * 5, invalid), self.profile([1] * 5))

    def test_boolean_is_not_a_runtime_measurement(self):
        with self.assertRaises(ValueError):
            module.choose(self.profile([10] * 5), self.profile([True] * 5))


if __name__ == '__main__':
    unittest.main()
