"""Tensor-read shape specialization must retain precision and indices."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import match_v2 as M
import mlir_lite as ML


def read_module(ty, index=0, indices='%i'):
    return ML.Module(f'''module {{
  func.func @read(%a: {ty}) {{
    %i = arith.constant {index} : index
    %r = tensor.extract %a[{indices}] : {ty}
    return
  }}
}}''')


def matches(sty, aty, index=0, indices='%i'):
    m = M.Matcher.__new__(M.Matcher)
    m.L, m.A = read_module(sty), read_module(aty, index, indices)
    m.spec = SimpleNamespace(params={'%a': sty}, yield_op=m.L.ops[-1])
    a = next(op for op in m.A.ops if op.name == 'tensor.extract')
    return bool(list(m.match_value('%r', ('op', a.idx, 0), M.Env())))


class TensorReadSpecialization(unittest.TestCase):
    def test_dynamic_extent_accepts_static_source(self):
        self.assertTrue(matches('tensor<?xf32>', 'tensor<2048xf32>'))

    def test_precision_is_retained(self):
        self.assertFalse(matches('tensor<?xf32>', 'tensor<2048xf64>'))

    def test_rank_is_retained(self):
        self.assertFalse(matches('tensor<?xf32>', 'tensor<32x64xf32>', indices='%i, %i'))

    def test_index_is_retained(self):
        self.assertFalse(matches('tensor<?xf32>', 'tensor<2048xf32>', index=1))

    def test_static_reference_extent_is_retained(self):
        self.assertFalse(matches('tensor<32xf32>', 'tensor<2048xf32>'))


if __name__ == '__main__':
    unittest.main()
