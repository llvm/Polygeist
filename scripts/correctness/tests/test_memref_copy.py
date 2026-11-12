import importlib.util
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get('POLYGEIST_ROOT', '/home/arjaiswal/Polygeist'))
STAGED = ROOT / 'issues/aten_remaining_v3_20261111/staged-resident-copy'
SPEC = importlib.util.spec_from_file_location('copy_lower', Path(os.environ.get(
    'TEST_MEMREF_LOWER', ROOT / 'scripts/correctness/lib_v2/lower_v2.py')))
LOWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOWER)
TEXT = '''module {
 func.func @copy2(%a: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                  %b: memref<?x?xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface} {
  memref.copy %a, %b : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
 }
}'''

def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, **kwargs)

class MemrefCopy(unittest.TestCase):
    def test_cpu_preserved_and_cuda_ir_valid(self):
        self.assertEqual(LOWER.rewrite_memory_copies(TEXT, 'cpu'), TEXT)
        self.assertEqual(LOWER.rewrite_memory_copies(TEXT, 'pva'), TEXT)
        output = LOWER.rewrite_memory_copies(TEXT, 'cuda')
        self.assertNotIn('memref.copy ', output)
        self.assertIn('memory copy: shape mismatch', output)
        p = run([ROOT/'build/bin/polygeist-opt'], input=output)
        self.assertEqual(p.returncode, 0, p.stderr)
        identity = TEXT.replace('memref.copy %a, %b', 'memref.copy %a, %a')
        self.assertNotIn('polygeist_v2_memref_copy', LOWER.rewrite_memory_copies(identity, 'cuda'))

    def test_lowered_descriptor_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            output=LOWER.rewrite_memory_copies(TEXT, 'cuda')
            p=run([ROOT/'llvm-project/build/bin/mlir-opt','--expand-strided-metadata',
                   '--lower-affine','--convert-arith-to-llvm','--convert-index-to-llvm',
                   '--convert-cf-to-llvm','--finalize-memref-to-llvm','--convert-func-to-llvm',
                   '--reconcile-unrealized-casts'],input=output)
            self.assertEqual(p.returncode,0,p.stderr)
            q=run([ROOT/'llvm-project/build/bin/mlir-translate','--mlir-to-llvmir'],input=p.stdout)
            self.assertEqual(q.returncode,0,q.stderr);(d/'copy.ll').write_text(q.stdout)
            (d/'main.c').write_text('''#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {float *allocated,*aligned; int64_t offset,sizes[2],strides[2];} Desc;
void _mlir_ciface_copy2(Desc*,Desc*);
int main(int argc,char**argv){float a[64],b[64],want[64];
for(int i=0;i<64;i++){a[i]=i+0.25f;b[i]=want[i]=-7.0f;}
Desc x={a,a,2,{3,4},{9,2}},y={b,b,3,{3,4},{8,1}};
if(argc>1)y.sizes[0]=2;
for(int i=0;i<3;i++)for(int j=0;j<4;j++)want[3+i*8+j]=a[2+i*9+j*2];
_mlir_ciface_copy2(&x,&y);
return memcmp(b,want,sizeof(b))!=0;}
''')
            helper=Path(os.environ.get('TEST_MEMREF_RUNTIME', ROOT/'runtime'))
            (d/'helper.c').write_text('#include "polygeist_v2_matrix_layout.inc"\n#include "polygeist_v2_memref_copy.inc"\n')
            p=run([ROOT/'llvm-project/build/bin/clang','-O2','-I'+str(helper),
                   '-I'+str(ROOT/'runtime'),d/'main.c',d/'helper.c',d/'copy.ll','-o',d/'program'])
            self.assertEqual(p.returncode,0,p.stderr)
            for args,ok in [([],True),(['shape'],False)]:
                p=run([d/'program',*args],preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
                self.assertEqual(p.returncode==0,ok,p.stderr)

    def test_runtime_layout_and_overflow_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);here=Path(__file__).resolve().parent
            helper=Path(os.environ.get('TEST_MEMREF_RUNTIME', ROOT/'runtime'))
            p=run(['gcc','-O2','-I'+str(helper),'-I'+str(ROOT/'runtime'),
                   here/'test_memref_copy.c','-o',d/'program'])
            self.assertEqual(p.returncode,0,p.stderr)
            for args,ok in [([],True),(['negative'],False),(['overflow'],False),(['null'],False)]:
                p=run([d/'program',*args],preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
                self.assertEqual(p.returncode==0,ok,p.stderr)

if __name__ == '__main__': unittest.main()
