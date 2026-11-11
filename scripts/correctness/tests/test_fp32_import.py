"""Generic fixes exercised by the FP32 reference import (no vendor execution)."""
import os
import re
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import kernel_match as K
import legalize_tensor_submaps as L
import match_v2 as M
import mlir_lite as ML
import recipe as R


def comparison(predicate, reverse=False):
    a, b = ('%out', '%in') if reverse else ('%in', '%out')
    return K.GenericBody(ins_arg_names=['%in'], outs_arg_names=['%out'],
                         body_lines=[f'%v = arith.cmpf {predicate}, {a}, {b} : f32'],
                         yield_values=['%v'], captures=[], indexing_maps=[],
                         iterator_types=['parallel'], constants={},
                         block_arg_types=['f32', 'f32'])


MULTI = '''module {
  func.func @test(%x: tensor<?xf32>, %y: tensor<?xf32>, %sum: tensor<f32>, %n: index) {
    %v = polygeist.submap(%sum, %n) {map = affine_map<(d0) -> ()>} : (tensor<f32>, index) -> tensor<?xf32>
    %r:2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["reduction"]} ins(%x : tensor<?xf32>) outs(%y, %v : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32, %acc: f32):
      %square = arith.mulf %in, %in : f32
      %total = arith.addf %acc, %square : f32
      linalg.yield %square, %total : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    %w = polygeist.submapInverse(%sum, %r#1, %n) {map = affine_map<(d0) -> ()>} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
    return
  }
}
'''


class GenericImportFixes(unittest.TestCase):
    def test_comparison_reversal_is_exact_in_strict_mode(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            a = K.encode_body(comparison('ogt'))
            self.assertTrue(K.equivalent(a, K.encode_body(comparison('olt', True))))
            self.assertFalse(K.equivalent(a, K.encode_body(comparison('ugt'))))
            self.assertFalse(K.equivalent(a, K.encode_body(comparison('oge'))))

    def test_split_independent_mixed_outputs(self):
        text, count = L.split_independent_outputs(MULTI)
        self.assertEqual(count, 1)
        ops = [g for g in ML.Module(text).ops if g.name == 'linalg.generic']
        self.assertEqual([g.iters for g in ops], [['parallel'], ['reduction']])
        self.assertEqual([len(g.outs) for g in ops], [1, 1])
        self.assertNotIn('%acc', ops[0].text)

    def test_cross_output_recurrence_cannot_split(self):
        text = MULTI.replace('%in, %in', '%in, %acc')
        self.assertEqual(L.split_independent_outputs(text), (text, 0))

    def test_effectful_body_cannot_split(self):
        text = MULTI.replace('arith.mulf %in, %in : f32',
                             'func.call @side_effect(%in) : (f32) -> f32')
        self.assertEqual(L.split_independent_outputs(text), (text, 0))


class ImportedSoftmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ROOT / 'generic_solver/kernel_library_v3.mlir'
        source = ROOT / 'third_party/cnn-extracted/llama_forward_ops.c'
        if not source.is_file() or '@aten_softmax_f32(' not in cls.lib.read_text():
            raise unittest.SkipTest('run the FP32 reference import first')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        steps = R.raise_c(R.PRODUCTION, str(source), 'kernel_llama_attention_softmax',
                          str(Path(cls.temp.name) / 'application'), [])
        if not steps.get('debuf') or not steps['debuf'].ok:
            raise AssertionError('fresh softmax raising failed: ' + repr(steps))
        cls.app = Path(steps['paths']['debuf'])

    def matches(self, text):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            return M.run(text, self.lib, region=False, allowed_specs={'aten_softmax_f32'})[1]

    def test_complete_fp32_reference_matches(self):
        matches = self.matches(self.app.read_text())
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(matches[0]['generics']), 3)

    def test_different_precision_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('f32', 'f64')))

    def test_inconsistent_reduction_bound_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('constant 31 : index',
                                                                  'constant 30 : index')))

    def test_incorrect_computation_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('arith.divf', 'arith.mulf')))

    def test_incorrect_producer_result_does_not_match(self):
        text = self.app.read_text()
        result = re.search(r'(%[\w.-]+)#1', text)
        self.assertIsNotNone(result)
        self.assertFalse(self.matches(text.replace(result[0], result[1]+'#0')))

    def test_changed_reduction_seed_does_not_match(self):
        self.assertFalse(self.matches(self.app.read_text().replace('0.000000e+00 : f32',
                                                                  '1.000000e+00 : f32')))


if __name__ == '__main__':
    unittest.main()
