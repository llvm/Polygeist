from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import lower_v2 as L
class ScalarLaunchPreparation(unittest.TestCase):
 def prepare(self,ret='float',result='f32',intent='in'):
  args=[A.SpecArg('input','array',rank=1,intent=intent,ctype=ret)]
  spec=A.SpecAbi('value','vendor_value',ret,args,{'input':('arg',0)},{})
  text='''module {
  func.func @application(%input: tensor<8xf32>) -> RESULT {
    %r = kernel.launch @value(%input) : (tensor<8xf32>) -> RESULT
    return %r : RESULT
  }
}'''.replace('RESULT',result)
  return L.prepare(text,{'value':spec})
 def test_tensor_inputs_to_scalar_result_prepare(self):
  text,names=self.prepare();self.assertIn('kernel.defn @value',text);self.assertEqual(names,{'value':'value'})
 def test_mismatched_or_mutating_return_contract_is_rejected(self):
  for kwargs in [{'result':'f64'},{'ret':'int'},{'intent':'out'},{'intent':'inout'}]:
   with self.subTest(kwargs=kwargs),self.assertRaises(L.LowerError):self.prepare(**kwargs)
if __name__=='__main__':unittest.main()
