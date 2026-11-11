"""Exact FP scalar-return binding and emitted cleanup order, without a GPU."""
from pathlib import Path
import os,sys,tempfile,unittest,subprocess
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import typed_wrapper as T
import runtime_adapter as R
import gen_v2_runtime as G
class ScalarReturns(unittest.TestCase):
 def setup_contract(self,ret,declared=None):
  cfg=dict(schema=1,kind='typed_c_wrapper',entry='wrapper',arguments=['n','input'],arrays={'input':['n']},bounds={'n':{'min':1,'max':64}},alias_policy='disjoint')
  target=dict(library='Vendor',call='Reduce',backends=['cuda'],runtime_adapter=cfg)
  args=[A.SpecArg('n','int',ctype='int'),A.SpecArg('input','array',rank=1,intent='in',ctype=ret)]
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'runtime').mkdir();(root/'runtime/polygeist_cublas_rt.h').write_text('/* POLYGEIST_TYPED_VENDOR_ENTRY Vendor Reduce wrapper */\n'+(declared or ret)+' wrapper(int,const '+ret+' *);\n')
   with patch.object(A,'REPO',root),patch.object(A,'CLANG','cc'):
    return T.contract(A,'reference',target,ret,args)
 def test_return_types_are_exact(self):
  for ret in ['float','double']:
   with self.subTest(ret=ret):self.setup_contract(ret)
   for wrong in ['void','double' if ret=='float' else 'float']:
    with self.subTest(ret=ret,wrong=wrong),self.assertRaises(A.AbiError):self.setup_contract(ret,wrong)
  with self.assertRaises(A.AbiError):self.setup_contract('int')
 def test_call_abi_carries_return_type(self):
  for ret in ['float','double','void']:
   sa=A.SpecAbi('reference','Reduce_reference',ret,[],{},{});target=dict(library='Vendor',call='Reduce',backends=['cuda'])
   with patch.object(A,'manifest_entry',return_value=(None,{'target':target})),patch.object(R,'contract',return_value=([],{},{})):
    self.assertEqual(R.build_call(A,'Reduce_reference',[sa]).returns,ret if ret!='void' else None)
 def test_compiled_scalar_result_and_cleanup(self):
  for ret in ['float','double']:
   params,_,cfg=self.setup_contract(ret);ca=A.CallAbi('scalar_'+ret,'Vendor','Reduce',params,[],returns=ret,backends=['cuda'],runtime_adapter=cfg)
   code='''#include <stdint.h>
#include <stdlib.h>
#include <math.h>
static int done;
static void polygeist_v2_fatal(const char*a,const char*b){(void)a;(void)b;exit(73);}
typedef struct{const void*p;} polygeist_v2_map_t;
static void polygeist_v2_map_init(polygeist_v2_map_t*m,const char*a,const char*b){m->p=0;(void)a;(void)b;}
static int polygeist_v2_map_add(polygeist_v2_map_t*m,const void*p,int64_t n,int w){m->p=p;(void)n;(void)w;return 0;}
static void polygeist_v2_map_to_device(polygeist_v2_map_t*m){(void)m;}
static void *polygeist_v2_map_dev(polygeist_v2_map_t*m,int k){(void)k;return(void*)m->p;}
static void polygeist_require_gpu_libraries_begin(void){done=0;}
static void polygeist_cublas_pipeline_begin(void){}
static void polygeist_cublas_pipeline_end(void){done|=1;}
static void polygeist_require_gpu_libraries_end(void){done|=2;}
static void polygeist_v2_map_to_host(polygeist_v2_map_t*m){(void)m;done|=4;}
'''+ret+' wrapper(int n,const '+ret+' *x){(void)n;return x[0];}\n'+T.generate(ca,G.proto)+'''
int main(int argc,char**argv){(void)argv;SCALAR input[]={-0.0,INFINITY,-INFINITY,NAN,1.23456789012345};
 if(argc>1){polygeist_v2_scalar_TYPE(1,input,1,1);return 99;}
 for(int i=0;i<5;++i){SCALAR result=polygeist_v2_scalar_TYPE(1,input+i,sizeof(SCALAR),1);
 if(done!=7)return 2;if(isnan(input[i])){if(!isnan(result))return 3;}
 else if(result!=input[i] || !!signbit(result)!=!!signbit(input[i]))return 4;}return 0;}
'''.replace('SCALAR',ret).replace('TYPE',ret)
   with tempfile.TemporaryDirectory() as temp:
    p=Path(temp);(p/'test.c').write_text(code);subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',str(p/'test.c'),'-o',str(p/'test')],check=True,capture_output=True)
    self.assertEqual(subprocess.run([str(p/'test')]).returncode,0)
    self.assertEqual(subprocess.run([str(p/'test'),'bad-capacity']).returncode,73)
if __name__=='__main__':unittest.main()
