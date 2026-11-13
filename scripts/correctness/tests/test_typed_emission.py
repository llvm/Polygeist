from pathlib import Path
import copy
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import gen_v2_runtime as G
import typed_wrapper as T

class TypedEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        cfg=dict(kind='typed_c_wrapper',memory_domain='host',entry='vendor',
                 bounds={'h':{'min':3,'max':2147483647},'w':{'min':3,'max':2147483647}},
                 arrays={'input':['h','w'],'output':[{'dimension':'h','offset':-1},'w']},
                 ordered_arguments=['h','w','input','output'])
        params=[A.AbiParam('h','int'),A.AbiParam('w','int')]
        for name in ['input','output']:
            params += [A.AbiParam(name,'array',elem='unsigned short',rank=2,layout=name+'_stride',written=name=='output'),A.AbiParam(name+'_stride','int',layout_of=name)]
        cls.ca=A.CallAbi('submit_reference','Vendor','submit',params,[],backends=['pva'],runtime_adapter=cfg)
        overflow=copy.deepcopy(cls.ca);overflow.call='overflow';overflow.runtime_adapter['entry']='vendor64'
        for p in overflow.params:
            if p.tkind=='array':p.elem='unsigned long'
        code='''#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
static int calls;
static void polygeist_v2_fatal(const char*a,const char*b){(void)a;(void)b;exit(73);}
static double polygeist_v2_now_ms(void){return 0;}
static void polygeist_v2_done(const char*a,const char*b,double t){(void)a;(void)b;(void)t;}
static int polygeist_v2_overlaps(const void*a,int64_t an,const void*b,int64_t bn){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<(uint64_t)an:x-y<(uint64_t)bn;}
static void vendor(int32_t h,int32_t w,const uint16_t*input,uint16_t*output){++calls;for(int i=1;i<h-1;i++)for(int j=1;j<w-1;j++)output[i*w+j]=input[i*w+j];}
static void vendor64(int32_t h,int32_t w,const unsigned long*x,unsigned long*y){(void)h;(void)w;(void)x;(void)y;exit(99);}
'''+T.generate(cls.ca,G.proto)+T.generate(overflow,G.proto)+'''
int main(int argc,char**argv){
 int mode=argc>1?atoi(argv[1]):0;uint16_t x[32],y[32];for(int i=0;i<32;i++){x[i]=(uint16_t)(32768+i);y[i]=65535;}
 int h=4,w=5,si=5,so=5;int64_t xb=40,yb=30;uint16_t*out=y;
 if(mode==1)yb=29;if(mode==2)so=4;if(mode==3)h=2;if(mode==4)out=x;if(mode==5){polygeist_v2_overflow(2147483647,2147483647,(unsigned long*)x,INT64_MAX,2147483647,(unsigned long*)y,INT64_MAX,2147483647);return 99;}
 polygeist_v2_submit_reference(h,w,x,xb,si,out,yb,so);
 if(calls!=1)return 2;for(int i=0;i<32;i++){if(x[i]!=(uint16_t)(32768+i))return 3;uint16_t want=(i/5>=1&&i/5<3&&i%5>=1&&i%5<4)?x[i]:65535;if(y[i]!=want)return 4;}return 0;}
'''
        src=cls.path/'test.c';src.write_text(code)
        subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',str(src),'-o',str(cls.path/'test')],check=True,capture_output=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_partial_output_preserves_border_and_guards(self):
        self.assertEqual(subprocess.run([str(self.path/'test'),'0']).returncode,0)
    def test_capacity_layout_bounds_overlap_and_overflow_fail_closed(self):
        for mode in range(1,6):
            with self.subTest(mode=mode):self.assertEqual(subprocess.run([str(self.path/'test'),str(mode)]).returncode,73)
    def test_cuda_uses_exact_pointer_types_and_device_mapping(self):
        ca=copy.deepcopy(self.ca);ca.backends=['cuda'];ca.runtime_adapter['memory_domain']='cuda_device'
        code=T.generate(ca,G.proto)
        self.assertIn('(const unsigned short *)polygeist_v2_map_dev',code)
        self.assertIn('(unsigned short *)polygeist_v2_map_dev',code)
        self.assertIn('polygeist_require_gpu_libraries_begin',code)
        self.assertNotIn('memcpy',code)

if __name__=='__main__':unittest.main()
