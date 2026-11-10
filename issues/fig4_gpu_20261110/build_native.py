#!/usr/bin/env python3
"""Import verbatim upstream CUDA kernel bodies into explicit ABI adapters."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
from build import ROOT, OUT, PB, CASES, command, sha

UP = ROOT/'third_party/polybenchGpu'
OLD = ROOT/'issues/polybench_section42/polybenchgpu_adapters'
SOURCES = dict(zip(CASES, ['2MM/2mm','3MM/3mm','ATAX/atax','BICG/bicg',
                         'CORR/correlation','COVAR/covariance','GEMM/gemm',
                         'GEMVER/gemver','GESUMMV/gesummv','MVT/mvt',
                         'SYR2K/syr2k','SYRK/syrk']))

def extent(text,start):
    brace=text.index('{',start)
    depth=1; end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return end

COMMON='''#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
static double start_wall;
static double wall_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static cudaError_t event_start(cudaEvent_t e) {
  start_wall=wall_ms(); return cudaEventRecord(e); }
static cudaError_t event_stop(cudaEvent_t e) {
  cudaError_t rc=cudaEventSynchronize(e);
  fprintf(stderr,"POLYBENCH_NATIVE_GPU_WALL_MS %.9f\\n",wall_ms()-start_wall);
  return rc;
}
static void pbgpu_checked(cudaError_t rc,const char *msg) {
  if(rc!=cudaSuccess) { fprintf(stderr,"CUDA %s: %s\\n",msg,cudaGetErrorString(rc));abort(); }
}
static void checked(cudaError_t rc,const char *msg) { pbgpu_checked(rc,msg); }
static void pbgpu_report(cudaEvent_t start,cudaEvent_t stop) {
  pbgpu_checked(cudaEventRecord(stop),"record stop");
  pbgpu_checked(event_stop(stop),"synchronize stop");
  float ms; pbgpu_checked(cudaEventElapsedTime(&ms,start,stop),"elapsed");
  fprintf(stderr,"POLYBENCH_NATIVE_GPU_TIMING device_ms=%.9f\\n",(double)ms);
}
'''

def build(name):
    d=OUT/name
    old=(OLD/(name+'.cu')).read_text()
    original=UP/'CUDA'/(SOURCES[name]+'.cu')
    text=original.read_text()
    blocks=[text[m.start():extent(text,m.start())] for m in
            re.finditer(r'__global__\s+void\s+\w+\s*\(',text)]
    start=old.index('extern "C" void kernel_'+name)
    host=old[start:extent(old,start)]
    defines='\n'.join(re.findall(r'^#define (?:N|M|NI|NJ|NK|NL|NM|NX|NY) \d+.*$',old,re.M))
    definitions=['#define DATA_TYPE double','#define FLOAT_N ((double)N)','#define EPS 0.1']
    for x in ['M','N','NI','NJ','NK','NL','NM','NX','NY']:
        definitions.append('#define _PB_'+x+' '+x.lower())
    if name in ('syrk','syr2k'):
        definitions+=['#define NI N','#define NJ M']
        host=host.replace('pbgpu_'+name+'_kernel',name+'_kernel')
    header=original.with_suffix('.cuh').read_text()
    definitions+=re.findall(r'^#define DIM_THREAD[^\n]+',header,re.M)
    qualifier='upstream CUDA kernels, canonical FP64 shapes and initialization'
    if name=='correlation':
        for oldname,newname in [('pbgpu_corr_mean','mean_kernel'),('pbgpu_corr_std','std_kernel'),
                                ('pbgpu_corr_reduce','reduce_kernel'),('pbgpu_corr_matrix','corr_kernel')]:
            host=host.replace(oldname,newname)
        host=host.replace('m, n, float_n,','m, n,')
        host=host.replace('  pbgpu_report(start, stop);','  pbgpu_report(start, stop);\n'
            '  double last_diagonal=1.0;\n'
            '  pbgpu_checked(cudaMemcpy(dcorr+M*M-1,&last_diagonal,sizeof(double),cudaMemcpyHostToDevice),"last diagonal");')
        qualifier+='; FLOAT_N and EPS configured to canonical C; upstream final diagonal copy retained'
    if name=='covariance':
        host=host.replace('pbgpu_covar_mean','mean_kernel').replace('pbgpu_covar_reduce','reduce_kernel').replace('pbgpu_covar_matrix','covar_kernel')
        host=host.replace('m, n, float_n,','m, n,')
        # Upstream computes an unnormalized covariance. Include a real vendor
        # scaling call in the interval to implement the canonical C contract.
        host=host.replace('  cudaEventRecord(start);',
            '  cublasHandle_t scale_handle;\n'
            '  if(cublasCreate(&scale_handle)!=CUBLAS_STATUS_SUCCESS) abort();\n'
            '  double scale=1.0/(float_n-1.0);\n'
            '  cudaEventRecord(start);')
        host=host.replace('  pbgpu_report(start, stop);',
            '  if(cublasDscal(scale_handle,M*M,&scale,dcov,1)!=CUBLAS_STATUS_SUCCESS) abort();\n'
            '  pbgpu_report(start, stop);\n  cublasDestroy(scale_handle);')
        qualifier+='; upstream CUDA plus timed cuBLAS normalization (not pure PolyBenchGPU)'
    host=host.replace('cudaEventRecord(start)','event_start(start)')
    host=host.replace('cudaEventSynchronize(stop)','event_stop(stop)')
    code='// Computational kernel bodies below are verbatim from the pinned upstream source.\n'
    code+='#include "native_common.cuh"\n#include <cublas_v2.h>\n'+defines+'\n'+'\n'.join(definitions)+'\n'
    code+='\n\n'.join(blocks)+'\n'+host+'\n'
    (d/'native_common.cuh').write_text(COMMON)
    (d/'native_adapter.cu').write_text(code)
    record={'upstream_source':str(original.relative_to(ROOT)), 'upstream_sha256':sha(original),
            'kernel_sha256':[hashlib.sha256(x.encode()).hexdigest() for x in blocks],
            'computational_bodies_verbatim':True,'qualifier':qualifier,
            'generated_source_sha256':sha(d/'native_adapter.cu')}
    log=d/'native-build.log';log.write_text('')
    env=dict(os.environ)
    rc=command(['/usr/local/cuda-12.6/bin/nvcc','-O3','-std=c++17','-arch=sm_87',
                '-ccbin','aarch64-linux-gnu-g++','-c',d/'native_adapter.cu','-o',d/'native_adapter.o'],log,env)
    if not rc:
        rc=command(['aarch64-linux-gnu-g++',d/'native_adapter.o',d/'export/harness.o',
                    d/'export/polybench.o',OUT/'iteration_driver.o',
                    '-L/usr/local/cuda-12.6/targets/sbsa-linux/lib',
                    '-L/usr/local/cuda-12.6/targets/sbsa-linux/lib/stubs',
                    '-lcudart','-lcublas','-lm','-o',d/'native'],log,env)
    record['build_rc']=rc
    if not rc:record['executable_sha256']=sha(d/'native')
    (d/'native-build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(name,'native build',rc,flush=True)

if __name__=='__main__':
    actual=subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()
    assert actual=='5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c',actual
    for name in sys.argv[1:] or CASES:build(name)
