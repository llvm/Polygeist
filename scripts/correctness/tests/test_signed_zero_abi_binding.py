"""A proved literal must retain its sign when rebound to a launch operand."""
import os,sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import match_v2 as M
class SignedZeroABI(unittest.TestCase):
 def matcher(self):
  m=object.__new__(M.Matcher);m.spec=SimpleNamespace(params={'%scale':'f32'})
  m.literal_allowed=lambda *a:True;m.app_gbs={0:SimpleNamespace(constants={'%positive':0.0,'%negative':-0.0})}
  m.akey=lambda name,op:('named',name);return m
 def test_launch_operand_uses_the_proved_zero_sign(self):
  m=self.matcher();op=SimpleNamespace(idx=0)
  for value,name in [(0.0,'%positive'),(-0.0,'%negative')]:
   result=list(m.match_cap('%scale',('Lit',value),op,M.Env()))
   self.assertEqual(len(result),1);self.assertEqual(result[0].vals['%scale'][3],('named',name))
 def test_repeated_parameter_cannot_change_zero_sign(self):
  m=self.matcher();op=SimpleNamespace(idx=0)
  for first,second in [(0.0,-0.0),(-0.0,0.0)]:
   env=list(m.match_cap('%scale',('Lit',first),op,M.Env()))[0]
   self.assertEqual(list(m.match_cap('%scale',('Lit',second),op,env)),[])
 def test_fixed_reference_constant_retains_sign(self):
  m=self.matcher();m.spec.params={};op=SimpleNamespace(idx=0)
  m.sdef=lambda _: (SimpleNamespace(name='arith.constant',text='%c = arith.constant -0.000000e+00 : f32'),0)
  self.assertEqual(list(m.match_cap('%c',('Lit',0.0),op,M.Env())),[])
  self.assertEqual(len(list(m.match_cap('%c',('Lit',-0.0),op,M.Env()))),1)
if __name__=='__main__':unittest.main()
