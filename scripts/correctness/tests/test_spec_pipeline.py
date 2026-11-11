#!/usr/bin/env python3
"""Regressions for imported matching, semantic preservation, and evidence reuse.
Run after build_lib_v2.py with /usr/bin/python3 -m unittest discover
-s scripts/correctness/tests -p test_spec_pipeline.py.
"""
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import kernel_match as KM
import spec_match as SM
from region_match.mlir_parse import parse_module
from region_match.symexec import execute, EUn, ERead, Unsupported


def scalar(lines, scalar_type='f64'):
    return KM.GenericBody(ins_arg_names=['%in'], outs_arg_names=['%out'],
                          body_lines=lines, yield_values=['%v'], captures=[],
                          indexing_maps=[], iterator_types=['parallel'], constants={},
                          block_arg_types=[scalar_type, scalar_type])


def program(body, argtype='memref<64xf64>'):
    text = f'''module {{
  func.func @test(%a: {argtype}, %b: memref<1xf64>) {{
    %zero = arith.constant 0 : index
    {body}
    return
  }}
}}'''
    funcs, aliases = parse_module(text)
    return execute(funcs[0], aliases)


class ScalarSemantics(unittest.TestCase):
    def test_rounding_is_not_identity(self):
        rounded = KM.encode_body(scalar([
            '%x = arith.truncf %in : f64 to f32',
            '%v = arith.extf %x : f32 to f64']))
        for model in ('strict', 'relaxed'):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL=model):
                self.assertFalse(KM.equivalent(rounded, KM.Term.In(0)))
                self.assertTrue(KM.equivalent(rounded, rounded))

    def test_conversion_signedness_is_preserved(self):
        signed = KM.encode_body(scalar(['%v = arith.sitofp %in : i32 to f64'], 'i32'))
        unsigned = KM.encode_body(scalar(['%v = arith.uitofp %in : i32 to f64'], 'i32'))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertFalse(KM.equivalent(signed, unsigned))

    def test_unknown_operation_is_not_a_wildcard(self):
        with self.assertRaises(ValueError):
            KM.encode_body(scalar(['%v = custom.operation %in : f64']))

    def test_reassociation_requires_opt_in(self):
        a, b, c = (KM.Term.In(i) for i in range(3))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            self.assertFalse(KM.equivalent((a + b) + c, a + (b + c)))
            self.assertFalse(KM.equivalent(a * KM.Term.Lit(0.0), KM.Term.Lit(0.0)))
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(KM.equivalent((a + b) + c, a + (b + c)))


class MemorySemantics(unittest.TestCase):
    def test_composed_subviews_keep_offsets_and_strides(self):
        p = program('''%s = memref.subview %a[3] [8] [2] : memref<64xf64> to memref<8xf64, strided<[2], offset: 3>>
    %t = memref.subview %s[1] [3] [2] : memref<8xf64, strided<[2], offset: 3>> to memref<3xf64, strided<[4], offset: 5>>
    %one = arith.constant 1 : index
    %v = memref.load %t[%one] : memref<3xf64, strided<[4], offset: 5>>
    memref.store %v, %b[%zero] : memref<1xf64>''')
        read = p.stmts[-1].value
        self.assertIsInstance(read, ERead)
        self.assertEqual(read.idx[0].const, 9)

    def test_reinterpret_cast_keeps_offset_and_stride(self):
        p = program('''%s = memref.reinterpret_cast %a to offset: [5], sizes: [3], strides: [2] : memref<64xf64> to memref<3xf64, strided<[2], offset: 5>>
    %one = arith.constant 1 : index
    %v = memref.load %s[%one] : memref<3xf64, strided<[2], offset: 5>>
    memref.store %v, %b[%zero] : memref<1xf64>''')
        self.assertEqual(p.stmts[-1].value.idx[0].const, 7)

    def test_reinterpret_of_offset_view_is_rejected(self):
        with self.assertRaises(Unsupported):
            program('''%s = memref.subview %a[3] [8] [2] : memref<64xf64> to memref<8xf64, strided<[2], offset: 3>>
    %t = memref.reinterpret_cast %s to offset: [0], sizes: [3], strides: [1] : memref<8xf64, strided<[2], offset: 3>> to memref<3xf64>''')

    def test_log1p_remains_log1p(self):
        p = program('''%x = memref.load %a[%zero] : memref<64xf64>
    %v = math.log1p %x : f64
    memref.store %v, %b[%zero] : memref<1xf64>''')
        self.assertIsInstance(p.stmts[-1].value, EUn)
        self.assertEqual(p.stmts[-1].value.op, 'log1p')

    def test_region_keeps_rounding(self):
        p = program('''%x = memref.load %a[%zero] : memref<64xf64>
    %y = arith.truncf %x : f64 to f32
    %v = arith.extf %y : f32 to f64
    memref.store %v, %b[%zero] : memref<1xf64>''')
        self.assertEqual(p.stmts[-1].value.op, 'arith.extf:f32 to f64')
        self.assertEqual(p.stmts[-1].value.a.op, 'arith.truncf:f64 to f32')

    def test_narrowing_index_conversion_needs_a_range_proof(self):
        with self.assertRaises(Unsupported):
            program('''%wide = arith.constant 257 : i64
    %small = arith.trunci %wide : i64 to i8
    %index = arith.index_cast %small : i8 to index
    %v = memref.load %a[%index] : memref<64xf64>
    memref.store %v, %b[%zero] : memref<1xf64>''')


class BackendSelection(unittest.TestCase):
    def test_layout_rewrite_cannot_bypass_backend_filter(self):
        from region_match import matcher as R
        emitter = SimpleNamespace(allowed_specs={'reference'})
        spec = SimpleNamespace(name='reference', entry={'target': {}})
        binding = R.Binding({'%a': ('%x', None, (1, 0))}, {}, {})
        replacement = SimpleNamespace(name='unavailable')
        with patch.object(R, 'transposed_equivalent', return_value=(replacement, binding, 'proved')):
            with patch.object(R, '_launch', side_effect=AssertionError('must not lower')):
                with self.assertRaises(R.Reject):
                    R._emit_one(emitter, spec, binding)


class ReductionRules(unittest.TestCase):
    def test_sum_combination_requires_equal_ranges_and_relaxed_model(self):
        from region_match.egg import RmIx, rm_sum, rules
        r, lo, hi = RmIx('test_r'), RmIx.lit(0), RmIx.lit(5)
        a, b = KM.Term.In(0), KM.Term.In(1)
        split = rm_sum(r, lo, hi, a) + rm_sum(r, lo, hi, b)
        combined = rm_sum(r, lo, hi, a + b)
        wrong = rm_sum(r, lo, hi, a) + rm_sum(r, lo, RmIx.lit(6), b)
        for model in ('strict', 'relaxed'):
            with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL=model, EQSAT_ABLATION='full'):
                eg = KM.EGraph()
                eg.register(split, combined, wrong)
                eg.run(rules() * 8)
                if model == 'relaxed':
                    eg.check(split == combined)
                else:
                    with self.assertRaises(Exception):
                        eg.check(split == combined)
                with self.assertRaises(Exception):
                    eg.check(wrong == combined)


class RaisingRecovery(unittest.TestCase):
    def test_failed_tensor_stage_does_not_validate_or_hide_an_affine_stage(self):
        from lib_v2.spec_provenance import validated_stages
        row = {'numerical_validation': {'affine': 'EXACT', 'defn': 'UNSOUND'}}
        self.assertEqual(validated_stages(row), ['cgeist-affine'])
        row['numerical_validation']['affine'] = 'LOWER_FAIL'
        self.assertEqual(validated_stages(row), [])

    def test_failed_optional_pass_preserves_its_input_and_failure(self):
        from lib_v2 import recipe
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'in.mlir', Path(directory) / 'out.mlir'
            source.write_text('module {}\n')
            cmd = [sys.executable, '-c',
                   'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("broken"); sys.exit(7)',
                   str(output)]
            result = recipe.run_with_legal_input(cmd, source, output, 10)
            self.assertTrue(result.ok)
            self.assertEqual(result.recovery['rc'], 7)
            self.assertEqual(result.recovery['cmd'], cmd)
            self.assertEqual(output.read_text(), source.read_text())

    def test_missing_checkpoint_does_not_report_success(self):
        from lib_v2 import recipe
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = recipe.run_with_legal_input([sys.executable, '-c', 'exit(7)'],
                                                 root / 'missing', root / 'out', 10)
            self.assertFalse(result.ok)


class ImportedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed')
        cls.env.start()
        cls.input = (ROOT / 'generic_solver/lib_v3/raised/openblas_dgemm_nn_debuf.mlir').read_text()
        cls.output, cls.proof = SM.match(cls.input, backend='cpu',
                                        only=['openblas_dgemm_nn'], region=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def test_imported_positive_and_evidence(self):
        self.assertEqual(self.proof['emitted'], ['openblas_dgemm_nn'])
        SM.verify(self.output, self.proof, backend='cpu')

    def test_wrong_computation_is_not_matched(self):
        bad = self.input.replace('arith.mulf', 'arith.subf')
        self.assertNotEqual(bad, self.input)
        _, proof = SM.match(bad, backend='cpu', only=['openblas_dgemm_nn'], region=False)
        self.assertEqual(proof['emitted'], [])

    def test_wrong_precision_is_not_matched(self):
        _, proof = SM.match(self.input.replace('f64', 'f32'), backend='cpu',
                            only=['openblas_dgemm_nn'], region=False)
        self.assertEqual(proof['emitted'], [])

    def test_empty_backend_has_no_fallback(self):
        output, proof = SM.match(self.input, backend='pva')
        self.assertEqual(output, self.input)
        self.assertEqual(proof['emitted'], [])

    def test_modified_ir_cannot_reuse_evidence(self):
        with self.assertRaises(SM.SpecificationError):
            SM.verify(self.output + '\n// edited', self.proof)

    def test_configuration_change_cannot_reuse_evidence(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            with self.assertRaises(SM.SpecificationError):
                SM.verify(self.output, self.proof)
        with self.assertRaises(SM.SpecificationError):
            SM.verify(self.output, self.proof, backend='cuda')

    def test_engine_change_cannot_reuse_evidence(self):
        with patch.object(SM, 'semantic_inputs', return_value={}):
            with self.assertRaises(SM.SpecificationError):
                SM.verify(self.output, self.proof)

    def test_input_with_existing_replacement_is_rejected(self):
        with self.assertRaises(SM.SpecificationError):
            SM.match(self.output, backend='cpu')


if __name__ == '__main__':
    unittest.main()
