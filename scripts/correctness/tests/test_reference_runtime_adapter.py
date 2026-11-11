"""Imported reference bindings preserve types, layout, backend, and provenance."""
import copy,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE/'lib_v2')]
import abi_v2 as A,gen_v2_runtime as G,runtime_adapter as R,build_lib_v2 as B,spec_match as S
class ReferenceRuntimeAdapter(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.calls,cls.specs,errors=A.build_table(['aten_add_f32','ggml_swiglu_f32','cutlass_rmsnorm_f32','openblas_somatcopy_f32'])
  if errors:raise AssertionError(errors)
 def contract(self,name='cutlass_rmsnorm_f32',target=None,args=None):
  sa=self.specs[name];return R.contract(A,name,target or A.manifest_entry(name)[1]['target'],sa.ret,args or sa.args)
 def test_gpu_only_bindings_have_no_cpu_wrapper(self):
  self.assertTrue(all(c.backends==['cuda'] for c in self.calls.values()))
  self.assertEqual(G.gen_cpu(self.calls),'')
  rows=S.registry();cpu,_=S.capabilities(rows,'cpu',only=list(self.specs));self.assertFalse(cpu)
 def test_unbound_or_mistyped_runtime_parameters_rejected(self):
  for edit in ['missing','wrong_type','wrong_rank','wrong_binding']:
   sa=self.specs['cutlass_rmsnorm_f32'];args=copy.deepcopy(sa.args);target=copy.deepcopy(A.manifest_entry(sa.spec)[1]['target'])
   if edit=='missing':del target['runtime_adapter']['arguments']['Weight']
   elif edit=='wrong_type':args[1].ctype='double'
   elif edit=='wrong_rank':args[1].rank=2
   else:target['runtime_adapter']['arguments']['Weight']='output'
   with self.subTest(edit=edit),self.assertRaises(A.AbiError):self.contract(target=target,args=args)
 def test_gpu_backend_cannot_be_advertised_as_cpu(self):
  target=copy.deepcopy(A.manifest_entry('cutlass_rmsnorm_f32')[1]['target']);target['backends']=['cpu','cuda']
  with self.assertRaises(A.AbiError):self.contract(target=target)
 def test_generated_adapters_require_gpu_and_disjoint_rows(self):
  source=G.gen_cuda(self.calls)
  self.assertIn('polygeist_require_gpu_libraries_begin()',source)
  self.assertIn('polygeist_v2_overlaps',source)
  self.assertIn('input_stride != 1',source)
  self.assertIn('polygeist_rmsnorm_f32(n,',source)
 def test_blas_cuda_only_has_no_fictitious_fortran(self):
  ca=self.calls['cublasSgeam'];self.assertIsNone(ca.fortran)
  self.assertIn('cublasSgeam(',G.gen_cuda({'cublasSgeam':ca}))
 def test_vla_layout_is_independent_of_element_type(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'ref.c';p.write_text('/* double stale[][wrong] */\nvoid f(int a,int b,float X[][a],SCALAR Y[][b]);')
   self.assertEqual(B.vla_leading_dims(p),{'a':'X','b':'Y'})
if __name__=='__main__':unittest.main()
