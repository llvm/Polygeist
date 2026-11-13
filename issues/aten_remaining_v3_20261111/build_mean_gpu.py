from pathlib import Path
import os,subprocess,json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'reductions/mean';src=R/'issues/aten_c_kernels/aten_mean.c'
p=json.loads((D/'match.json').read_text());assert p['emitted']==['cub_reduce_value_sum_f64']
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
static double parameters[1];
static const size_t sizes[]={256,1};
static const int outputs[]={0,1};
void aten_mean(double*,double*);void native_reference(double*,double*);
static void invoke(double **p){aten_mean(p[0],p[1]);}
static void reference(double **p){native_reference(p[0],p[1]);}
static void direct(double **p){polygeist_require_gpu_libraries_begin();polygeist_cublas_pipeline_begin();double result=polygeist_cub_reduce_value_sum_f64(256,0.0,p[0]);p[1][0]=result/256.0;polygeist_cublas_pipeline_end();polygeist_require_gpu_libraries_end();}
'''
tail=(O/'mean_harness_tail.c').read_text().replace('direct_cudnn','direct_cub').replace('double edges[]={','int zeros=argc>1&&!strcmp(argv[1],"zeros");\n double edges[]={').replace('h[a][i+4]=outputs[a]?127.25f:x;','if(zeros)x=-0.0f;\n   h[a][i+4]=outputs[a]?127.25f:x;')
(D/'harness.c').write_text(header+tail)
env=dict(os.environ,PYTHON='/usr/bin/python3',PYTHONPATH=str(R/'scripts/correctness')+':'+str(R/'scripts/correctness/lib_v2'),POLYGEIST_NUMERIC_MODEL='strict',POLYGEIST_BUILD_WORK_DIR=str(D/'work'),POLYGEIST_MATCH_PROOF=str(D/'match.json'),POLYGEIST_CUSTOM_CUDA_OBJ=str(D/'reference.o'),POLYGEIST_MINIMAL_CUDNN_RUNTIME='1',POLYGEIST_ENABLE_CUB='1')
commands=[['aarch64-linux-gnu-gcc','-O0','-ffp-contract=off','-Daten_mean=native_reference','-c',str(src),'-o',str(D/'reference.o')],[str(R/'scripts/correctness/polygeist_build.sh'),'--target=jetson','--function=aten_mean','--semantic-mlir='+str(D/'matched.mlir'),'--harness='+str(D/'harness.c'),'-o',str(D/'program'),str(src),'-I'+str(R/'runtime'),'-I/usr/local/cuda-12.6/targets/sbsa-linux/include','-I/usr/include/aarch64-linux-gnu']]
with (D/'build.log').open('w') as log:
 for cmd in commands:
  rc=subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=600).returncode
  if rc:break
row=dict(rc=rc,commands=commands,proof_sha256=hashlib.sha256((D/'match.json').read_bytes()).hexdigest())
if not rc:row['binary_sha256']=hashlib.sha256((D/'program').read_bytes()).hexdigest()
(D/'build.json').write_text(json.dumps(row,indent=2)+'\n');print('mean build',rc,flush=True)
raise SystemExit(rc)
