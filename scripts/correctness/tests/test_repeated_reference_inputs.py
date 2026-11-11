"""Several reference inputs may bind one tensor only with matching reads and types."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import match_v2 as M


def application(ty='f32',predicate='oge',input_map='affine_map<(d0)->(d0)>',old=False):
    return '''module {
  func.func @filter(%input: memref<?xTYPE>, %limit: TYPE, %output: memref<?xTYPE>) {
    %n = arith.constant 8 : index
    %zero = arith.constant 0.0 : TYPE
    %negative = arith.negf %limit : TYPE
    %x = bufferization.to_tensor %input : memref<?xTYPE>
    %out = bufferization.to_tensor %output : memref<?xTYPE>
    %xv = polygeist.submap(%x, %n) {map = affine_map<(d0)->(d0)>} : (tensor<?xTYPE>, index) -> tensor<8xTYPE>
    %ov = polygeist.submap(%out, %n) {map = affine_map<(d0)->(d0)>} : (tensor<?xTYPE>, index) -> tensor<8xTYPE>
    %r = linalg.generic {indexing_maps = [INPUT_MAP, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%xv : tensor<8xTYPE>) outs(%ov : tensor<8xTYPE>) {
    ^bb0(%value: TYPE, %old: TYPE):
      %lower = arith.cmpf PREDICATE, %value, %negative : TYPE
      %upper = arith.cmpf ole, %value, %limit : TYPE
      %inside = arith.andi %lower, %upper : i1
      %selected = arith.select %inside, %zero, %value : TYPE
      EXTRA
      linalg.yield ANSWER : TYPE
    } -> tensor<8xTYPE>
    %updated = polygeist.submapInverse(%out, %r, %n) {map = affine_map<(d0)->(d0)>} : (tensor<?xTYPE>, tensor<8xTYPE>, index) -> tensor<?xTYPE>
    %dest = bufferization.to_memref %updated : memref<?xTYPE>
    memref.copy %dest, %output : memref<?xTYPE> to memref<?xTYPE>
    return
  }
}'''.replace('EXTRA','%with_old = arith.addf %selected, %old : TYPE' if old else '').replace('ANSWER','%with_old' if old else '%selected').replace('TYPE',ty).replace('PREDICATE',predicate).replace('INPUT_MAP',input_map)


class RepeatedReferenceInputs(unittest.TestCase):
    def matched(self,**kwargs):
        with patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='strict'):
            return M.run(application(**kwargs),M.DEFAULT_LIB,region=False,diag=True,
                         allowed_specs={'aten_shrink_backward_f32'})[1]

    def test_two_formal_reads_bind_the_same_input(self):
        matches=self.matched()
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['binding']['%grad_val'],'%x')
        self.assertEqual(matches[0]['binding']['%self_val'],'%x')
        self.assertEqual(matches[0]['binding']['%lambd_val'],'%limit')

    def test_boundary_comparison_is_not_relaxed(self):
        self.assertFalse(self.matched(predicate='ogt'))

    def test_repeated_inputs_do_not_erase_precision(self):
        self.assertFalse(self.matched(ty='f64'))

    def test_broadcast_read_does_not_become_elementwise(self):
        self.assertFalse(self.matched(input_map='affine_map<(d0)->(0)>'))

    def test_initial_output_read_is_not_discarded(self):
        self.assertFalse(self.matched(old=True))


if __name__=='__main__':unittest.main()
