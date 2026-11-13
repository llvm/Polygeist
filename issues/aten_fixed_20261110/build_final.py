#!/usr/bin/env python3
"""Build precisely the pinned paper's 115 ATen cases from source."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
names = [r['kernel'] for r in csv.DictReader((HERE/'paper_targets.csv').open())]
names.sort(key=lambda name: name not in ['aten_sparse_coo_to_csr_cpu','aten_convert_coo_to_csr_cpu'])
env = os.environ.copy()
env['POLYGEIST_FORCE_RESIDENT'] = '1'
env['POLYGEIST_KEEP_WORK'] = '1'
for key in ('POLYGEIST_CUTENSORNET_ROOT', 'POLYGEIST_CUTENSORNET_INCLUDE',
            'POLYGEIST_CUTENSORNET_LIBDIR'):
    env.pop(key, None)
paths = ['build/bin/cgeist', 'build/bin/polygeist-opt',
         'scripts/correctness/kernel_match_rewrite.py',
         'scripts/correctness/aten_pointwise_graph_silicon.py',
         'scripts/correctness/polygeist_build.sh',
         'runtime/polygeist_cublas_rt_cuda.c',
         'generic_solver/kernel_library_phase2.mlir', 'runtime/polygeist_cub_rt.cu', 'lib/polygeist/Passes/FoldSCFIf.cpp', 'lib/polygeist/Passes/LowerKernelLaunchToCuBLAS.cpp']
def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
before = hashes()
command = ['/usr/bin/python3', str(ROOT/'scripts/correctness/aten_pointwise_graph_silicon.py'),
           '--output', str(HERE/'raised_final'), '--jobs', '4']
for name in names:
    command += ['--kernel', name]
rc = subprocess.run(command, cwd=ROOT, env=env).returncode
after = hashes()
(HERE/'final-build-provenance.json').write_text(json.dumps(
    dict(command=command, before=before, after=after, stable=before==after,
         exit_code=rc, kernels=names), indent=2)+'\n')
raise SystemExit(rc if before == after else 2)
