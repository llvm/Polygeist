"""Optional external wrappers link without their SDK and fail closed if called."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib_v2'))
import abi_v2 as A
import gen_v2_runtime as G
import typed_wrapper as T

class OptionalTypedLinkage(unittest.TestCase):
    def test_linked_missing_and_unused_backend(self):
        params = [A.AbiParam('n', 'int')]
        for name in ['input', 'output']:
            params += [A.AbiParam(name, 'array', elem='float', rank=1,
                                 layout=name+'_stride', written=name=='output'),
                       A.AbiParam(name+'_stride', 'int', layout_of=name)]
        cfg = dict(kind='typed_c_wrapper', memory_domain='host', entry='optional_vendor',
                   linkage='optional', bounds={'n': {'min':1, 'max':16}},
                   arrays={'input':['n'], 'output':['n']},
                   ordered_arguments=['n','input','output'])
        ca = A.CallAbi('optional_reference','Vendor','submit', params, [],
                       backends=['pva'], runtime_adapter=cfg)
        code = '''#include <stdint.h>
#include <stdlib.h>
void optional_vendor(int32_t n,const float*x,float*y);
static void polygeist_v2_fatal(const char*a,const char*b){(void)a;(void)b;exit(73);}
static double polygeist_v2_now_ms(void){return 0;}
static void polygeist_v2_done(const char*a,const char*b,double t){(void)a;(void)b;(void)t;}
static int polygeist_v2_overlaps(const void*a,int64_t an,const void*b,int64_t bn){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<(uint64_t)an:x-y<(uint64_t)bn;}
'''+T.generate(ca,G.proto)+'''
#ifdef LINKED_VENDOR
void optional_vendor(int32_t n,const float*x,float*y){for(int i=0;i<n;i++)y[i]=x[i];}
#endif
int main(int argc,char**argv){(void)argv;if(argc==1)return 0;float x[3]={1,2,3},y[3]={0};polygeist_v2_optional_reference(3,x,12,1,y,12,1);return y[2]==3?0:99;}
'''
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);src=p/'test.c';src.write_text(code)
            for linked in [False,True]:
                binary=p/str(linked)
                cmd=['cc','-std=c11','-Wall','-Wextra','-Werror',str(src),'-o',str(binary)]
                if linked:cmd.append('-DLINKED_VENDOR')
                subprocess.run(cmd,check=True,capture_output=True)
                self.assertEqual(subprocess.run([str(binary)]).returncode,0)
                self.assertEqual(subprocess.run([str(binary),'invoke']).returncode,0 if linked else 73)

if __name__=='__main__':unittest.main()
