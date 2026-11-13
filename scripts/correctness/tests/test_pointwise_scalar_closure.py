"""Captured pure scalar operations and boolean selection retain exact semantics."""
from pathlib import Path
import itertools
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import abi_v2 as A
import kernel_match as K
import mlir_lite as ML
import runtime_adapter as R


def fixture(body, outer='', scalar=False):
    text='''module {
  kernel.defn @test(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xf32>, %out: tensor<?xf32>SCALAR) -> tensor<?xf32> {
    %zero = arith.constant 0.0 : f32
    %yes_value = arith.constant 7.0 : f32
    %no_value = arith.constant -11.0 : f32
    %true = arith.constant true
    %false = arith.constant false
    OUTER
    %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} ins(%a, %b, %c : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32, %old: f32):
      BODY
      linalg.yield %answer : f32
    } -> tensor<?xf32>
    kernel.yield %r : tensor<?xf32>
  }
}'''.replace('SCALAR',', %limit: f32' if scalar else '').replace('OUTER',outer).replace('BODY',body)
    f=ML.Module(text).funcs[0]
    g=next(o for o in f.ops if o.name=='linalg.generic')
    gb=K.parse_generics(f.text,infer_outputs_from_yield=True)[0]
    return f,g,gb,['a','b','c'],'out'


def evaluate(graph, inputs, parameters=None):
    values={i:v for i,v in enumerate(inputs)}
    values.update({4+i:v for i,v in enumerate(graph['scalars'])})
    for key,index in graph['scalar_arguments'].items():values[4+index]=parameters[key]
    modes={int(number):name for number,name in re.findall(
        r'case (\d+): \*mode = CUDNN_POINTWISE_(\w+); return true;',
        (A.REPO/'runtime/polygeist_cublas_rt_cuda.c').read_text())}
    for i in range(graph['nodes']):
        instruction=(graph['words'][i//2]>>(32*(i%2)))&0xffffffff
        mode=modes[instruction>>24]
        a=values[(instruction>>16)&255]
        b=values.get((instruction>>8)&255)
        c=values.get(instruction&255)
        if mode=='CMP_GT':value=a>b
        elif mode=='CMP_LT':value=a<b
        elif mode=='LOGICAL_AND':value=bool(a and b)
        elif mode=='LOGICAL_OR':value=bool(a or b)
        elif mode=='LOGICAL_NOT':value=not a
        elif mode=='BINARY_SELECT':value=b if a else c
        elif mode=='NEG':value=-a
        else:raise AssertionError(mode)
        values[12+i]=value
    return values[11+graph['nodes']]


class PointwiseScalarClosure(unittest.TestCase):
    def compile(self, body, outer='', scalar=False):
        args=[SimpleNamespace(kind='scalar',ctype='float',name='limit')] if scalar else []
        with patch.object(R,'reference_region',return_value=fixture(body,outer,scalar)):
            return R.pointwise_graph(A,'test',args,'n')

    def test_boolean_select_truth_tables(self):
        for left,right in [('%q','%r0'),('%q','%false'),('%true','%r0'),
                           ('%false','%r0'),('%q','%true'),('%true','%false'),
                           ('%false','%true')]:
            body='''%p = arith.cmpf ogt, %x, %zero : f32
      %q = arith.cmpf ogt, %y, %zero : f32
      %r0 = arith.cmpf ogt, %z, %zero : f32
      %selected = arith.select %p, LEFT, RIGHT : i1
      %answer = arith.select %selected, %yes_value, %no_value : f32'''.replace('LEFT',left).replace('RIGHT',right)
            graph=self.compile(body)
            for p,q,r in itertools.product([False,True],repeat=3):
                terms={'%q':q,'%r0':r,'%true':True,'%false':False}
                expected=7. if (terms[left] if p else terms[right]) else -11.
                with self.subTest(left=left,right=right,p=p,q=q,r=r):
                    self.assertEqual(evaluate(graph,[1. if b else -1. for b in [p,q,r]]),expected)

    def test_captured_pure_expression_preserves_dynamic_parameter(self):
        graph=self.compile('''%condition = arith.cmpf olt, %x, %negative : f32
      %answer = arith.select %condition, %yes_value, %no_value : f32''',
            '%negative = arith.negf %limit : f32',scalar=True)
        for limit in [-.75,.25]:
            for x in [-1.,0.,1.,float('nan')]:
                self.assertEqual(evaluate(graph,[x,0.,0.],{'limit':limit}),7. if x < -limit else -11.)

    def test_effectful_capture_is_rejected(self):
        with self.assertRaises(A.AbiError):
            self.compile('''%condition = arith.cmpf olt, %x, %loaded : f32
      %answer = arith.select %condition, %yes_value, %no_value : f32''',
                '%loaded = memref.load %memory[%index] : memref<f32>')


if __name__=='__main__':unittest.main()
