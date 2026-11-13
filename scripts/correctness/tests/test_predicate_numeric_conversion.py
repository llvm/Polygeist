"""Boolean conversions preserve signedness and reject unproved integer casts."""
from pathlib import Path
import importlib.util,itertools,os,sys,unittest
from unittest.mock import patch
ROOT=Path(os.environ.get('POLYGEIST_ROOT',Path(__file__).resolve().parents[3]))
sys.path[:0]=[str(ROOT/'scripts/correctness/tests'),str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import abi_v2 as A
from test_pointwise_scalar_closure import fixture,evaluate
if os.environ.get('STAGED_BOOLEAN_ADAPTER'):
 s=importlib.util.spec_from_file_location('staged_runtime_adapter',os.environ['STAGED_BOOLEAN_ADAPTER']);R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
else:import runtime_adapter as R
class PredicateConversionTests(unittest.TestCase):
 def compile(self,body,scalar=False):
  args=[type('Arg',(),dict(kind='scalar',ctype='float',name='limit'))()] if scalar else []
  with patch.object(R,'reference_region',return_value=fixture(body,scalar=scalar)):
   return R.pointwise_graph(A,'test',args,'n')
 def test_bool_float_casts_and_extensions(self):
  prefix='%p = arith.cmpf ogt, %x, %zero : f32\n'
  cases=[('arith.uitofp %p : i1 to f32',1),('arith.sitofp %p : i1 to f32',-1)]
  for extension,conversion,value in [('extui','uitofp',1),('extui','sitofp',1),('extsi','sitofp',-1)]:
   for width in [8,16,32,64]:
    body=prefix+f'%v = arith.{extension} %p : i1 to i{width}\n%answer = arith.{conversion} %v : i{width} to f32'
    graph=self.compile(body)
    for x in [-1.,0.,1.,float('nan')]:self.assertEqual(evaluate(graph,[x,0.,0.]),float(value) if x>0 else 0.)
  for conversion,value in cases:
   graph=self.compile(prefix+'%answer = '+conversion)
   for x in [-1.,0.,1.,float('nan')]:self.assertEqual(evaluate(graph,[x,0.,0.]),float(value) if x>0 else 0.)
 def test_boolean_equality_xor_truth_tables(self):
  for operation in ['arith.xori','arith.cmpi eq,','arith.cmpi ne,']:
   for right in ['%q','%true','%false']:
    body='%p = arith.cmpf ogt, %x, %zero : f32\n%q = arith.cmpf ogt, %y, %zero : f32\n'+f'%z0 = {operation} %p, {right} : i1\n%answer = arith.uitofp %z0 : i1 to f32'
    graph=self.compile(body)
    for p,q in itertools.product([False,True],repeat=2):
     r={'%q':q,'%true':True,'%false':False}[right];want=(p==r) if 'eq' in operation else p!=r
     self.assertEqual(evaluate(graph,[float(p),float(q),0.]),float(want))
 def test_dynamic_parameter_is_not_reused_as_zero_literal(self):
  graph=self.compile('%p = arith.cmpf ogt, %x, %limit : f32\n%answer = arith.uitofp %p : i1 to f32',scalar=True)
  self.assertEqual(evaluate(graph,[0.,0.,0.],{'limit':.25}),0.)
  self.assertEqual(evaluate(graph,[1.,0.,0.],{'limit':.25}),1.)
 def test_unsupported_casts_and_integer_predicates_rejected(self):
  prefix='%p = arith.cmpf ogt, %x, %zero : f32\n'
  bad=['%answer = arith.sitofp %x : i32 to f32','%v = arith.extsi %p : i1 to i32\n%answer = arith.uitofp %v : i32 to f32','%v = arith.extui %p : i1 to i1\n%answer = arith.uitofp %v : i1 to f32','%v = arith.extui %p : i1 to i128\n%answer = arith.uitofp %v : i128 to f32','%q = arith.cmpi slt, %p, %p : i1\n%answer = arith.uitofp %q : i1 to f32','%q = arith.xori %x, %y : f32\n%answer = arith.uitofp %q : i1 to f32']
  for body in bad:
   with self.subTest(body=body),self.assertRaises(A.AbiError):self.compile(prefix+body)
if __name__=='__main__':unittest.main()
