"""An overwritten temporary must not replace the enclosing write destination."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'lib_v2')]
import match_v2 as M


def application(extra='', value='%sum', bound='%n'):
    return '''module {
  func.func @combine(%a: memref<?xf32>, %b: memref<?xf32>) {
    %n = arith.constant 8 : index
    %short = arith.constant 7 : index
    %x = bufferization.to_tensor %a : memref<?xf32>
    %y = bufferization.to_tensor %b : memref<?xf32>
    %av = polygeist.submap(%x, %n) {map = affine_map<(d0)->(d0)>} : (tensor<?xf32>, index) -> tensor<8xf32>
    %bv = polygeist.submap(%y, %n) {map = affine_map<(d0)->(d0)>} : (tensor<?xf32>, index) -> tensor<8xf32>
    %empty = tensor.empty() : tensor<8xf32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%av, %bv : tensor<8xf32>, tensor<8xf32>) outs(%empty : tensor<8xf32>) {
    ^bb0(%xv: f32, %yv: f32, %old: f32):
      %sum = arith.addf %xv, %yv : f32
      EXTRA
      linalg.yield VALUE : f32
    } -> tensor<8xf32>
    %updated = polygeist.submapInverse(%y, %result, BOUND) {map = affine_map<(d0)->(d0)>} : (tensor<?xf32>, tensor<8xf32>, index) -> tensor<?xf32>
    %dest = bufferization.to_memref %updated : memref<?xf32>
    memref.copy %dest, %b : memref<?xf32> to memref<?xf32>
    return
  }
}'''.replace('EXTRA', extra).replace('VALUE', value).replace('BOUND', bound)


class UnusedOutputDestination(unittest.TestCase):
    def matched(self, **kwargs):
        with patch.dict(os.environ, POLYGEIST_NUMERIC_MODEL='strict'):
            output, matches, _ = M.run(application(**kwargs), M.DEFAULT_LIB,
                                       region=False, diag=True, allowed_specs={'aten_add_f32'})
        return output, matches

    def test_output_binds_the_final_destination(self):
        output, matches = self.matched()
        self.assertEqual(len(matches), 1)
        self.assertIn('kernel.launch @aten_add_f32', output)
        self.assertEqual(matches[0]['binding']['%output'], '%y')

    def test_reading_the_initial_output_cannot_be_ignored(self):
        _, matches = self.matched(extra='%with_old = arith.addf %sum, %old : f32', value='%with_old')
        self.assertFalse(matches)

    def test_inverse_write_extent_is_still_checked(self):
        _, matches = self.matched(bound='%short')
        self.assertFalse(matches)


if __name__ == '__main__':
    unittest.main()
