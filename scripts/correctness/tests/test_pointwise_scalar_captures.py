"""Reject unsafe pointwise bindings; preserve caller-supplied scalar values."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import abi_v2 as A
import gen_v2_runtime as G
import runtime_adapter as R


class PointwiseScalarCaptures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls, cls.specs, errors = A.build_table(['aten_threshold_f32', 'aten_scaled_add_f32'])
        if errors:
            raise AssertionError(errors)

    def test_runtime_receives_scalar_parameters_without_specializing_them(self):
        source = G.gen_cuda(self.calls)
        for name in ['alpha', 'threshold', 'value']:
            self.assertIn('float ' + name, source)
        for call in self.calls.values():
            graph = call.runtime_adapter['graph']
            supplied = set(graph['scalar_arguments'])
            expected = {a.name for a in self.specs[call.specs[0]].args if a.kind == 'scalar'}
            self.assertEqual(supplied, expected)
            self.assertEqual(len(set(graph['scalar_arguments'].values())), len(expected))
            body = G.gen_runtime_adapter_cuda(call)
            graph_call = next(line for line in body.splitlines()
                              if line.strip().startswith('polygeist_cudnn_pointwise_graph_f32('))
            for name in supplied:
                self.assertIn(', ' + name + ',', graph_call)

    def test_fp64_scalar_cannot_bind_to_fp32_reference(self):
        spec = self.specs['aten_scaled_add_f32']
        args = copy.deepcopy(spec.args)
        next(a for a in args if a.kind == 'scalar').ctype = 'double'
        target = A.manifest_entry(spec.spec)[1]['target']
        with self.assertRaises(A.AbiError):
            R.contract(A, spec.spec, target, spec.ret, args)

    def test_unordered_predicate_cannot_be_silently_lowered_as_ordered(self):
        spec = self.specs['aten_threshold_f32']
        text = (A.SPEC_DIR.parent / 'kernel_library_v3.mlir').read_text()
        original = 'arith.cmpf ole,'
        self.assertIn(original, text)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'kernel_library_v3.mlir').write_text(text.replace(original, 'arith.cmpf ule,'))
            provider = types.SimpleNamespace(REPO=A.REPO, SPEC_DIR=root / 'lib_v3',
                                              AbiError=A.AbiError, enum_of_constant=A.enum_of_constant)
            with self.assertRaisesRegex(A.AbiError, 'unsupported floating comparison predicate'):
                R.pointwise_graph(provider, spec.spec, spec.args, 'n')


if __name__ == '__main__':
    unittest.main()
