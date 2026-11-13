#!/usr/bin/env python3
"""Reproduce the paper's locally authored CUDA adapters, labelled explicitly."""
import json
import os
from build import OUT, ROOT, command, sha
from build_native import OLD, UP

for name in ('correlation', 'covariance'):
    d = OUT/name
    original = OLD/(name+'.cu')
    text = original.read_text().replace('#include "common.cuh"', '#include "native_common.cuh"')
    text = text.replace('cudaEventRecord(start)', 'event_start(start)')
    source = d/'paper_baseline.cu'
    source.write_text(text)
    log = d/'paper-baseline-build.log'
    log.write_text('Only timing instrumentation is changed; computational bodies are the paper adapters.\n')
    rc = command(['/usr/local/cuda-12.6/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_87',
                  '-ccbin', 'aarch64-linux-gnu-g++', '-I'+str(UP), '-I'+str(OLD),
                  '-c', source, '-o', d/'paper_baseline.o'], log, os.environ)
    if not rc:
        rc = command(['aarch64-linux-gnu-g++', d/'paper_baseline.o', d/'export/harness.o',
                      d/'export/polybench.o', OUT/'iteration_driver.o',
                      '-L/usr/local/cuda-12.6/targets/sbsa-linux/lib', '-lcudart', '-lm',
                      '-o', d/'paper-baseline'], log, os.environ)
    record = dict(original_source=str(original.relative_to(ROOT)), original_sha256=sha(original),
                  generated_source_sha256=sha(source), build_rc=rc,
                  qualifier='paper local CUDA adaptation; not verbatim upstream PolyBenchGPU')
    if not rc:
        record['executable_sha256'] = sha(d/'paper-baseline')
    (d/'paper-baseline-build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(name, 'paper baseline build', rc, flush=True)
