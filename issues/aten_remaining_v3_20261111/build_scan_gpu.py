from pathlib import Path
import os,subprocess,json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'scan/cumsum';src=R/'issues/aten_c_kernels/aten_cumsum.c'
p=json.loads((D/'match.json').read_text());assert p['emitted']==['cub_inclusive_scan_f32']
header='''#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "polygeist_cublas_rt.h"
#define NA 2
#define NS 0
static float parameters[1];
static const size_t sizes[]={256,256};
static const int outputs[]={0,1};
void aten_cumsum(float*,float*);void native_reference(float*,float*);
static void invoke(float **p){aten_cumsum(p[0],p[1]);}
static void reference(float **p){native_reference(p[0],p[1]);}
static void direct(float **p){polygeist_require_gpu_libraries_begin();polygeist_cublas_pipeline_begin();polygeist_cub_scan_inclusive_seed_f32(256,0.0f,p[0],p[1]);polygeist_cublas_pipeline_end();polygeist_require_gpu_libraries_end();}
'''
tail=(O/'ternary_harness.c').read_text().replace('direct_cudnn','direct_cub').replace('float edges[]={','int zeros=argc>1&&!strcmp(argv[1],"zeros");\n float edges[]={').replace('h[a][i+4]=outputs[a]?127.25f:x;','if(zeros)x=-0.0f;\n   h[a][i+4]=outputs[a]?127.25f:x;')
(D/'harness.c').write_text(header+tail)
env=dict(os.environ,PYTHON='/usr/bin/python3',PYTHONPATH=str(R/'scripts/correctness')+':'+str(R/'scripts/correctness/lib_v2'),POLYGEIST_NUMERIC_MODEL='strict',POLYGEIST_BUILD_WORK_DIR=str(D/'work'),POLYGEIST_MATCH_PROOF=str(D/'match.json'),POLYGEIST_CUSTOM_CUDA_OBJ=str(D/'reference.o'),POLYGEIST_MINIMAL_CUDNN_RUNTIME='1',POLYGEIST_ENABLE_CUB='1')
commands=[['aarch64-linux-gnu-gcc','-O0','-ffp-contract=off','-Daten_cumsum=native_reference','-c',str(src),'-o',str(D/'reference.o')],[str(R/'scripts/correctness/polygeist_build.sh'),'--target=jetson','--function=aten_cumsum','--semantic-mlir='+str(D/'matched.mlir'),'--harness='+str(D/'harness.c'),'-o',str(D/'program'),str(src),'-I'+str(R/'runtime'),'-I/usr/local/cuda-12.6/targets/sbsa-linux/include','-I/usr/include/aarch64-linux-gnu']]
with (D/'build.log').open('w') as log:
 for cmd in commands:
  rc=subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=600).returncode
  if rc:break
row=dict(rc=rc,commands=commands,proof_sha256=hashlib.sha256((D/'match.json').read_bytes()).hexdigest())
if not rc:row['binary_sha256']=hashlib.sha256((D/'program').read_bytes()).hexdigest()
(D/'build.json').write_text(json.dumps(row,indent=2)+'\n');print('cumsum build',rc,flush=True)
raise SystemExit(rc)
