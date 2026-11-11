"""Independent library calls must not hide provable neighboring regions."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_match.mlir_parse import parse_module,Op
from region_match.symexec import execute,EUn,ESel
from region_match.equations import Builder,FConst,Reject
from region_match import matcher as M
class RegionBoundaries(unittest.TestCase):
 def program(self,initializer='%zero',touch='%other'):
  text="""module {
 func.func @f(%value: f32, %other: memref<1xf32>, %out: memref<1xf32>) {
  %zero = arith.constant 0.0 : f32
  %i = arith.constant 0 : index
  %scratch = memref.alloca() : memref<f32>
  memref.store INITIAL, %scratch[] : memref<f32>
  kernel.launch @opaque(TOUCH) : (TYPE) -> ()
  %v = memref.load %scratch[] : memref<f32>
  memref.store %v, %out[%i] : memref<1xf32>
  return
 }
}
""".replace('INITIAL',initializer).replace('TOUCH',touch).replace('TYPE','memref<f32>' if touch=='%scratch' else 'memref<1xf32>')
  fs,a=parse_module(text);return execute(fs[0],a,None,lambda _: [0])
 def test_private_constant_survives_unrelated_opaque_call(self):
  p=self.program();b=Builder(p,{p.stmts[-1].top});self.assertEqual(b.stats['incoming_scalar_constants'],1)
  eqs=b.element_equations(p.stmts[-1].buf);self.assertEqual(eqs[0][1],FConst(0))
 def test_opaque_write_invalidates_incoming_constant(self):
  p=self.program(touch='%scratch');b=Builder(p,{p.stmts[-1].top});self.assertEqual(b.stats['incoming_scalar_constants'],0)
  with self.assertRaises(Reject):b.element_equations(p.stmts[-1].buf)
 def test_computed_value_is_not_an_incoming_constant(self):
  p=self.program(initializer='%value');b=Builder(p,{p.stmts[-1].top});self.assertEqual(b.stats['incoming_scalar_constants'],0)
  with self.assertRaises(Reject):b.element_equations(p.stmts[-1].buf)
 def test_region_local_definitions_are_not_live_outs(self):
  inner=Op('arith.addf',['%local'],'',0,0)
  outer=Op('linalg.generic',['%result'],'',0,1,[[inner]])
  self.assertEqual(M._defined_names(outer),{'%result'})
 def test_boolean_extensions_retain_signedness_and_condition(self):
  for cast,conv in [('extui','uitofp'),('extsi','sitofp')]:
   text="""module {
 func.func @f(%n: i32, %out: memref<1xf32>) {
  %zero = arith.constant 0 : i32
  %i = arith.constant 0 : index
  %cond = arith.cmpi sgt, %n, %zero : i32
  %v = arith.CAST %cond : i1 to i32
  %f = arith.CONV %v : i32 to f32
  memref.store %f, %out[%i] : memref<1xf32>
  return
 }
}
""".replace('CAST',cast).replace('CONV',conv)
   fs,a=parse_module(text);p=execute(fs[0],a);e=p.stmts[-1].value
   self.assertIsInstance(e,EUn);self.assertEqual(e.a.op,'arith.'+cast+':i1 to i32');self.assertIsInstance(e.a.a,ESel)
 def test_launch_effects_come_from_selected_suite(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);(base/'lib_v3').mkdir();(base/'kernel_library_v3.mlir').write_text('// lib_v2-meta: {"spec":"only_v3","params":["in","out"],"outputs":["out"]}\n')
   resolve=M._launch_output_resolver([],str(base/'lib_v3'));self.assertEqual(resolve('only_v3'),[1]);self.assertIsNone(resolve('missing'))
if __name__=='__main__':unittest.main()
