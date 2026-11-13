"""Exact typed integer expressions may use FP32 only with proved finite bounds."""
from pathlib import Path
import importlib.util,sys,itertools,struct,unittest,re
from unittest.mock import patch
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts/correctness/tests'),str(R/'scripts/correctness'),str(R/'scripts/correctness/lib_v2')]
import abi_v2 as A
import runtime_adapter as B
from test_pointwise_scalar_closure import fixture
MODES={int(k):v for k,v in re.findall(r'case (\d+): \*mode = CUDNN_POINTWISE_(\w+); return true;',(R/'runtime/polygeist_cublas_rt_cuda.c').read_text())}
def evaluate(g,x,y):
 v={0:x,1:y,2:0.};v.update({4+i:k for i,k in enumerate(g['scalars'])})
 for i in range(g['nodes']):
  word=(g['words'][i//2]>>(32*(i%2)))&0xffffffff;mode=MODES[word>>24];a=v[(word>>16)&255];b=v.get((word>>8)&255);c=v.get(word&255)
  if mode=='CMP_GT':z=a>b
  elif mode=='BINARY_SELECT':z=b if a else c
  elif mode=='ADD':z=a+b
  elif mode=='SUB':z=a-b
  else:raise AssertionError(mode)
  v[12+i]=struct.unpack('f',struct.pack('f',z))[0]
 return v[11+g['nodes']]
class IntegerGraph(unittest.TestCase):
 def compile(self,body,outer=''):
  with patch.object(B,'reference_region',return_value=fixture(body,outer)):
   return B.pointwise_graph(A,'test',[],'n')
 def body(self,operation='subi',width=32,ext='extui',conversion='sitofp'):
  return f'''%p = arith.cmpf ogt, %x, %zero : f32
%q = arith.cmpf ogt, %y, %zero : f32
%a = arith.{ext} %p : i1 to i{width}
%b = arith.{ext} %q : i1 to i{width}
%r = arith.{operation} %a, %b : i{width}
%answer = arith.{conversion} %r : i{width} to f32'''
 def test_truth_tables_and_positive_zero(self):
  for width,ext,op in itertools.product([8,16,32,64],['extui','extsi'],['addi','subi']):
   g=self.compile(self.body(op,width,ext))
   for x,y in itertools.product([-1.,-0.,0.,1.,float('nan'),float('inf')],repeat=2):
    a=int(x>0)*(1 if ext=='extui'else-1);b=int(y>0)*(1 if ext=='extui'else-1);want=float(a+b if op=='addi'else a-b)
    self.assertEqual(struct.pack('f',evaluate(g,x,y)),struct.pack('f',want))
 def test_nested_integer_chain_and_unsigned_nonnegative(self):
  s=self.body('addi').replace('%answer = arith.sitofp %r : i32 to f32','%s = arith.addi %r, %a : i32\n%answer = arith.uitofp %s : i32 to f32');g=self.compile(s)
  for x,y in itertools.product([-1.,1.],repeat=2):self.assertEqual(evaluate(g,x,y),2*int(x>0)+int(y>0))
 def test_exact_boundary_constant(self):
  s=self.body().replace('%r = arith.subi %a, %b : i32','%r = arith.subi %limit, %a : i32');g=self.compile(s,'%limit = arith.constant 16777216 : i32')
  self.assertEqual(evaluate(g,1.,0.),16777215.);self.assertEqual(evaluate(g,0.,0.),16777216.)
 def test_overflow_precision_signedness_and_types_rejected(self):
  bad=[(self.body('addi',2),''),(self.body(conversion='uitofp'),''),(self.body().replace('%r = arith.subi %a, %b : i32','%r = arith.addi %limit, %a : i32'),'%limit = arith.constant 16777216 : i32'),(self.body().replace('%r = arith.subi %a, %b : i32','%r = arith.addi %limit, %a : i32'),'%limit = arith.constant 1.0 : f32'),(self.body().replace('%r = arith.subi %a, %b : i32','%r = arith.subi %x, %b : i32'),''),(self.body().replace('%r : i32 to f32','%r : i16 to f32'),''),(self.body('muli'),''),(self.body(ext='extsi',conversion='uitofp'), '')]
  for body,outer in bad:
   with self.subTest(body=body),self.assertRaises(A.AbiError):self.compile(body,outer)
if __name__=='__main__':unittest.main()
