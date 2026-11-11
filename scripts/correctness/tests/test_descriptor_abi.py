"""Descriptor binding tests; computational recognition stays in common matchers."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import abi_v2 as A
import descriptor_abi as D
import gen_v2_runtime as G
import lower_v2 as L
import spec_match as S


class DescriptorBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls, cls.specs, errors = A.build_table(['aten_softmax_f32'])
        if errors:
            raise AssertionError(errors)
        cls.sa = cls.specs['aten_softmax_f32']
        cls.ca = cls.calls['cudnnSoftmaxForward']
        cls.target = A.manifest_entry(cls.sa.spec)[1]['target']
        cls.proto = A.resolve_proto('cuDNN', 'cudnnSoftmaxForward')

    def check_contract(self, target=None, args=None):
        return D.contract(A, target or self.target, args or self.sa.args, self.proto)

    def test_typed_bindings_and_backend(self):
        self.assertEqual(self.ca.backends, ['cuda'])
        self.assertEqual(self.sa.binding['x'], ('arg', 1))
        self.assertEqual(self.sa.binding['y'], ('arg', 2))
        self.assertEqual([a.ctype for a in self.sa.args], ['int', 'float', 'float'])

    def test_vendor_call_and_descriptor_are_generated(self):
        cuda = G.gen_cuda(self.calls)
        self.assertIn('CUDNN_DATA_FLOAT, 1, 1, 1, n', cuda)
        self.assertIn('CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE', cuda)
        self.assertIn('incx != 1', cuda)
        self.assertIn('polygeist_v2_overlaps(x, x_bytes, y, y_bytes)', cuda)
        self.assertNotIn('cudnnSoftmaxForward', G.gen_cpu(self.calls))

    def test_cpu_capability_is_rejected(self):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed'):
            rows = S.registry()
            cuda, _ = S.capabilities(rows, 'cuda', only=['aten_softmax_f32'])
            cpu, why = S.capabilities(rows, 'cpu', only=['aten_softmax_f32'])
        self.assertEqual(cuda, {'aten_softmax_f32'})
        self.assertFalse(cpu)
        self.assertIn('unavailable', why['aten_softmax_f32'])

    def test_reference_precision_must_match_contract(self):
        args = copy.deepcopy(self.sa.args)
        args[1].ctype = 'double'
        with self.assertRaises(A.AbiError):
            self.check_contract(args=args)

    def test_unknown_dimensions_are_rejected(self):
        target = copy.deepcopy(self.target)
        target['descriptor_adapter']['vendor_arguments']['xDesc']['dimensions'][-1] = 'missing'
        with self.assertRaises(A.AbiError):
            self.check_contract(target)

    def test_incorrect_vendor_enum_type_is_rejected(self):
        target = copy.deepcopy(self.target)
        target['descriptor_adapter']['vendor_arguments']['algo']['value'] = 'CUDNN_DATA_FLOAT'
        with self.assertRaises(A.AbiError):
            self.check_contract(target)

    def test_unbound_vendor_argument_is_rejected(self):
        target = copy.deepcopy(self.target)
        del target['descriptor_adapter']['vendor_arguments']['beta']
        with self.assertRaises(A.AbiError):
            self.check_contract(target)

    def test_cpu_advertising_is_rejected(self):
        target = copy.deepcopy(self.target)
        target['backends'] = ['cpu', 'cuda']
        with self.assertRaises(A.AbiError):
            self.check_contract(target)

    def lower(self, ty):
        text = f'    kernel.launch @aten_softmax_f32(%n, %x, %y) : (i32, {ty}, {ty}) -> ()'
        m = L.LAUNCH_RE.match(text)
        self.assertIsNotNone(m)
        return L.lower_launch(m, self.sa, self.ca, 'test')

    def test_lowering_keeps_float_and_actual_stride(self):
        body, decl = self.lower('memref<?xf32>')
        self.assertIn('polygeist_v2_cudnnSoftmaxForward', decl)
        self.assertIn('memref<?xf32>', '\n'.join(body))
        self.assertNotIn('f64', '\n'.join(body))

    def test_lowering_rejects_wrong_type_or_rank(self):
        for ty in ('memref<?xf64>', 'memref<?x?xf32>'):
            with self.subTest(ty=ty), self.assertRaises(L.LowerError):
                self.lower(ty)


if __name__ == '__main__':
    unittest.main()
