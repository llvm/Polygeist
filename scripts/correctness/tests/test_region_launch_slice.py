"""Opaque calls on tensor slices retain coordinates and invalidate old values."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_match.affine import Aff
from region_match.mlir_parse import parse_module
from region_match.symexec import Executor, Unsupported


class LaunchSliceTests(unittest.TestCase):
    def program(self, source="%restored", row=1):
        return """module {
 func.func @f(%out: memref<3x4xf32>, %readback: memref<1xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %base = bufferization.to_tensor %out restrict writable : memref<3x4xf32>
  %row = tensor.extract_slice %base[1, 0] [1, 4] [1, 1] : tensor<3x4xf32> to tensor<4xf32>
  %updated = kernel.launch @opaque(%row) : (tensor<4xf32>) -> tensor<4xf32>
  %restored = polygeist.submapInverse(%base, %updated, %four) {map = affine_map<(d0) -> (ROW, d0)>} : (tensor<3x4xf32>, tensor<4xf32>, index) -> tensor<3x4xf32>
  %value = tensor.extract SOURCE[%one, %zero] : tensor<3x4xf32>
  memref.store %value, %readback[%zero] : memref<1xf32>
  return
 }
}
""".replace("SOURCE", source).replace("ROW", str(row))

    def execute(self, text, resolver=lambda _: [0]):
        functions, aliases = parse_module(text)
        executor = Executor(functions[0], aliases, None, resolver)
        executor.run()
        return executor

    def test_result_keeps_slice_and_parent_is_restored(self):
        ex = self.execute(self.program())
        updated, restored = ex.env["%updated"], ex.env["%restored"]
        self.assertEqual(updated.view.index, [Aff.cst(1), Aff.var("@v0")])
        self.assertEqual(updated.view.sizes, [Aff.cst(4)])
        self.assertIsNone(restored.view)
        self.assertEqual(restored.ver, 1)
        self.assertEqual(len(ex.stmts), 1)  # No fictitious copy for same-view scatter.
        self.assertEqual(ex.stmts[0].value.idx, (Aff.cst(1), Aff.cst(0)))

    def test_old_parent_remains_stale(self):
        with self.assertRaisesRegex(Unsupported, "stale tensor"):
            self.execute(self.program(source="%base"))

    def test_different_row_requires_real_copy(self):
        ex = self.execute(self.program(row=2))
        self.assertEqual(len(ex.stmts), 2)
        self.assertEqual(ex.stmts[0].idx[0], Aff.cst(2))
        self.assertEqual(ex.stmts[0].value.idx[0], Aff.cst(1))

    def test_unknown_call_is_not_ignored(self):
        with self.assertRaisesRegex(Unsupported, "unknown specification"):
            self.execute(self.program(), resolver=lambda _: None)


if __name__ == "__main__":
    unittest.main()
