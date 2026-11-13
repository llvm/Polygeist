from pathlib import Path
import ctypes as C,importlib.util,sys,unittest,re,struct,random,math
from unittest.mock import patch
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'scripts/correctness/tests'),str(R/'scripts/correctness'),str(R/'scripts/correctness/lib_v2')]
import abi_v2 as A
import runtime_adapter as B
from test_pointwise_scalar_closure import fixture
lib=C.CDLL('libm.so.6');ops={n:getattr(lib,n+'f')for n in ['ceil','floor','trunc']}
for f in ops.values():f.argtypes=[C.c_float];f.restype=C.c_float
modes={int(k):v for k,v in re.findall(r'case (\d+): \*mode = CUDNN_POINTWISE_(\w+); return true;',(R/'runtime/polygeist_cublas_rt_cuda.c').read_text())}
def eval_graph(graph,x):
 values={0:x,1:0.,2:0.};values.update({4+i:v for i,v in enumerate(graph['scalars'])})
 for i in range(graph['nodes']):
  word=(graph['words'][i//2]>>(32*(i%2)))&0xffffffff;mode=modes[word>>24];a=values[(word>>16)&255];b=values.get((word>>8)&255);c=values.get(word&255)
  if mode=='CMP_LT':v=a<b
  elif mode in ['FLOOR','CEIL']:v=ops[mode.lower()](a)
  elif mode=='BINARY_SELECT':v=b if a else c
  else:raise AssertionError(mode)
  values[12+i]=v
 return values[11+graph['nodes']]
class TruncGraph(unittest.TestCase):
 def compile(self,body):
  with patch.object(B,'reference_region',return_value=fixture(body)):
   return B.pointwise_graph(A,'test',[],'n')
 def test_floating_boundaries_signed_zero_and_nan(self):
  g=self.compile('%answer = math.trunc %x : f32');rng=random.Random(20261112)
  bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x3f000000,0xbf000000,0x3f7fffff,0xbf7fffff,0x4b000000,0xcb000000,0x4b800000,0xcb800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001]+[rng.getrandbits(32)for _ in range(65536)]
  for b in bits:
   x=struct.unpack('f',struct.pack('I',b))[0];want=ops['trunc'](x);got=eval_graph(g,x)
   if math.isnan(want):self.assertTrue(math.isnan(got))
   else:self.assertEqual(struct.pack('f',want),struct.pack('f',got),hex(b))
 def test_other_types_and_arity_are_rejected(self):
  for body in ['%answer = math.trunc %x : f64','%answer = math.trunc %true : f32','%answer = math.trunc %x, %y : f32']:
   with self.subTest(body=body),self.assertRaises(A.AbiError):self.compile(body)
if __name__=='__main__':unittest.main()
