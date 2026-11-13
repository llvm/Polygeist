"""Returned tensor slices must preserve their parent and all dataflow coordinates."""
from pathlib import Path
import os,sys,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import match_v2 as M
TEXT=(Path(__file__).resolve().parent/'fixtures/returned_slice.mlir').read_text()
class ReturnedSlice(unittest.TestCase):
 def match(self,text):
  with patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='strict'):
   return M.run(text,M.DEFAULT_LIB,region=False,diag=True,allowed_specs={'aten_softmax_f32'})[:2]
 def mutate(self,a,b):
  self.assertEqual(TEXT.count(a),1);return TEXT.replace(a,b)
 def test_full_row_pipeline(self):
  out,m=self.match(TEXT);self.assertEqual(len(m),1);self.assertEqual(len(m[0]['generics']),3);self.assertIn('tensor.insert_slice',out)
 def test_initial_read_cannot_change_row(self):
  text=self.mutate('%extracted = tensor.extract %1[%arg2, %c0]', '%extracted = tensor.extract %1[%c0, %c0]')
  self.assertFalse(self.match(text)[1])
 def test_final_writeback_cannot_change_row(self):
  text=self.mutate('polygeist.submapInverse(%11, %13, %arg2, %c5)', 'polygeist.submapInverse(%11, %13, %c0, %c5)')
  self.assertFalse(self.match(text)[1])
 def test_inverse_extent_cannot_shrink(self):
  text=self.mutate('polygeist.submapInverse(%11, %13, %arg2, %c5)', 'polygeist.submapInverse(%11, %13, %arg2, %c4)')
  self.assertFalse(self.match(text)[1])
 def test_captured_reduction_cannot_change(self):
  text=self.mutate('%extracted_3 = tensor.extract %10#1[]','%extracted_3 = tensor.extract %6[]')
  self.assertFalse(self.match(text)[1])
 def test_signed_or_wider_type_not_equivalent(self):
  for ty in ['f64']:
   with self.subTest(ty=ty):self.assertFalse(self.match(TEXT.replace('f32',ty))[1])
 def test_wrong_initial_element(self):
  text=self.mutate('%extracted = tensor.extract %1[%arg2, %c0]', '%extracted = tensor.extract %1[%arg2, %c4]')
  self.assertFalse(self.match(text)[1])
if __name__=='__main__':unittest.main()
