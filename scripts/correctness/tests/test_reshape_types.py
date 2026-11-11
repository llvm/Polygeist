from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import mlir_lite as ML


class ReshapeTypesTest(unittest.TestCase):
    def test_collapse_uses_result_descriptor_rank(self):
        text = '%v = memref.collapse_shape %a [[0, 1], [2]] : memref<3x5x7xf32> into memref<15x7xf32>'
        self.assertEqual(ML._result_types(text, 'memref.collapse_shape', 1),
                         ['memref<15x7xf32>'])

    def test_expand_uses_result_tensor_rank(self):
        text = '%v = tensor.expand_shape %a [[0, 1]] : tensor<15xf32> into tensor<3x5xf32>'
        self.assertEqual(ML._result_types(text, 'tensor.expand_shape', 1),
                         ['tensor<3x5xf32>'])

    def test_collapsed_strides_are_preserved(self):
        text = '%v = memref.collapse_shape %a [[0, 1], [2]] : memref<3x5x7xf32, strided<[70, 14, 2]>> into memref<15x7xf32, strided<[14, 2]>>'
        self.assertEqual(ML._result_types(text, 'memref.collapse_shape', 1),
                         ['memref<15x7xf32, strided<[14, 2]>>'])


if __name__ == '__main__':
    unittest.main()
