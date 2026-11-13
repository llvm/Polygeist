"""Temporary kernel definitions must not shadow application symbols."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import lower_v2 as L
class LaunchSymbols(unittest.TestCase):
 def test_existing_and_generated_names_remain_distinct(self):
  arg=A.SpecArg('input','array',rank=1,intent='in',ctype='float')
  spec=A.SpecAbi('value','vendor_value','float',[arg],{'input':('arg',0)},{})
  text='''module {
  func.func private @value__v2sig1()
  func.func @value(%input: tensor<8xf32>) -> f32 {
    %r = kernel.launch @value(%input) : (tensor<8xf32>) -> f32
    %s = kernel.launch @value(%input) : (tensor<8xf32>) -> f32
    %sum = arith.addf %r, %s : f32
    return %sum : f32
  }
}'''
  prepared,names=L.prepare(text,{'value':spec})
  self.assertEqual(names,{'value__v2sig2':'value'})
  self.assertEqual(prepared.count('kernel.launch @value__v2sig2('),2)
  self.assertIn('func.func @value(',prepared)
  L.run_opt([],prepared,[])
 def test_quoted_symbols_are_reserved(self):
  arg=A.SpecArg('input','array',rank=1,intent='in',ctype='float')
  spec=A.SpecAbi('value','vendor_value','float',[arg],{'input':('arg',0)},{})
  text='''module {
  func.func private @"value"()
  func.func @application(%input: tensor<8xf32>) -> f32 {
    %r = kernel.launch @value(%input) : (tensor<8xf32>) -> f32
    return %r : f32
  }
}'''
  prepared,names=L.prepare(text,{'value':spec});self.assertEqual(names,{'value__v2sig1':'value'})
  L.run_opt([],prepared,[])
if __name__=='__main__':unittest.main()
