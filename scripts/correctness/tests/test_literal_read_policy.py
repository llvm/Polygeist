"""Zero captures must not enable read-deleting arithmetic specializations."""
from pathlib import Path
import importlib.util
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'lib_v2'))
import mlir_lite as ML
loader = importlib.util.spec_from_file_location('literal_read_policy', HERE / 'region_match/literal_policy.py')
P = importlib.util.module_from_spec(loader)
loader.loader.exec_module(P)


class LiteralReadPolicy(unittest.TestCase):
    def uses(self, body):
        text = ('module {\n func.func @f(%x: f32, %parameter: f32) -> f32 {\n' + body + '\n }\n}')
        return ML.Module(text).funcs[0].ops

    def test_selected_zero_keeps_unselected_nonfinite_values_unobserved(self):
        ops = self.uses('''  %p = arith.cmpf ole, %x, %parameter : f32
  %r = arith.select %p, %parameter, %x : f32
  return %r : f32''')
        safe = P.zero_preserves_reads(ops, '%parameter')
        self.assertTrue(safe)
        self.assertTrue(P.literal_ok({}, 'parameter', 0.0,
                                    zero_read_preserving=safe)[0])
        self.assertFalse(P.literal_ok({'assumptions': ['parameter != 0']},
                                     'parameter', 0.0, zero_read_preserving=safe)[0])

    def test_multiplicative_zero_still_cannot_discard_nan_or_infinite_reads(self):
        ops = self.uses('''  %r = arith.mulf %parameter, %x : f32
  return %r : f32''')
        safe = P.zero_preserves_reads(ops, '%parameter')
        self.assertFalse(safe)
        self.assertFalse(P.literal_ok({}, '%parameter', 0.0,
                                     zero_read_preserving=safe)[0])

    def test_unknown_and_unused_parameters_fail_closed(self):
        ops = self.uses('''  %r = func.call @unknown(%parameter) : (f32) -> f32
  return %r : f32''')
        self.assertFalse(P.zero_preserves_reads(ops, '%parameter'))
        self.assertFalse(P.zero_preserves_reads(ops, '%absent'))
        self.assertFalse(P.literal_ok({}, 'parameter', 0.0)[0])


if __name__ == '__main__':
    unittest.main()
