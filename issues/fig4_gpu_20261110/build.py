#!/usr/bin/env python3
"""Fresh C kernel builds with separately declared, algorithm-neutral harnesses."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PB = ROOT / 'tools/cgeist/Test/polybench'
SCRIPTS = ROOT / 'scripts/correctness'
CASES = '2mm 3mm atax bicg correlation covariance gemm gemver gesummv mvt syr2k syrk'.split()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def function(text, name):
    match = re.search(r'^(?:static\s+)?(?:void|int)\s+' + re.escape(name) + r'\s*\(', text, re.M)
    if not match:
        raise ValueError(name)
    start = text.index('{', match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[match.start():start], text[match.start():end]

def command(argv, log, env):
    with log.open('a') as f:
        f.write('\nCOMMAND ' + shlex.join(map(str, argv)) + '\n')
        f.flush()
        try:
            return subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                                  stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
        except subprocess.TimeoutExpired:
            return 124

def build(name):
    d = OUT / name
    d.mkdir(exist_ok=True)
    original, = PB.rglob(name + '.c')
    source = d / original.name
    shutil.copyfile(original, source)
    header = original.with_suffix('.h')
    (d / header.name).write_text(re.sub(r'(#\s*define DATA_PRINTF_MODIFIER )"%0\.2l?f "',
                                      r'\1"%.17g "', header.read_text()))
    text = original.read_text()
    fn = 'kernel_' + name
    prototype, _ = function(text, fn)
    prototype = re.sub(r'^static\s+', '', prototype)
    harness = ('/* Test fixture: original initialization and output routines,\n'
               ' * invoking a separately compiled kernel ABI. This is not a\n'
               ' * whole-program transformation or symbol replacement. */\n'
               '#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n'
               '#include <string.h>\n#include <polybench.h>\n'
               '#include "' + header.name + '"\nextern ' + prototype.strip() + ';\n')
    for helper in ('init_array', 'print_array', 'main'):
        harness += function(text, helper)[1] + '\n'
    (d / 'harness.c').write_text(harness)
    flags = ['-O3', '-I' + str(PB / 'utilities'), '-I' + str(d),
             '-DLARGE_DATASET', '-DDATA_TYPE_IS_DOUBLE', '-DPOLYBENCH_TIME']
    env = dict(os.environ, PYTHON='/usr/bin/python3', POLYGEIST_KEEP_WORK='1',
               POLYGEIST_CGEIST_FLAGS='--no-inline', POLYGEIST_WRAP_KERNEL_PIPELINE='0',
               POLYGEIST_MINIMAL_CUDA_RUNTIME='1', POLYGEIST_SKIP_LINK='1',
               POLYGEIST_POLYBENCH_REPETITIONS='0',
               POLYGEIST_HARNESS_CFLAGS='-Dmain=polygeist_polybench_iteration',
               POLYGEIST_BUILD_WORK_DIR=str(d / 'export'),
               POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=os.environ.get('POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE', 'auto'))
    log = d / 'fresh-build.log'
    log.write_text('')
    row = dict(kernel=name, source=str(original.relative_to(ROOT)),
               source_sha256=sha(original), harness_sha256=sha(d / 'harness.c'),
               header_sha256=sha(d / header.name), flags=flags,
               scope='isolated compiler-generated kernel, explicit ABI test harness')
    row['environment'] = {k: v for k, v in env.items() if k.startswith('POLYGEIST_')}
    rc = command([SCRIPTS / 'polygeist_build.sh', '--target=jetson',
                  '--function=' + fn, '--harness=' + str(d / 'harness.c'),
                  '-o', d / 'unused', source, *flags], log, env)
    row['source_build_rc'] = rc
    if not rc:
        row['matched_calls'] = re.findall(r'kernel.launch\s+@([^ (]+)', (d/'export/matched.mlir').read_text())
        env.update(POLYGEIST_BUILD_WORK_DIR=str(d / 'gpu'),
                   POLYGEIST_GPU_RESIDUAL_FUNCTION=fn,
                   POLYGEIST_GPU_DATA_RESIDENCY_FUNCTION=fn,
                   POLYGEIST_GPU_DATA_RESIDENCY_ASSUME_NO_ALIAS='1',
                   POLYGEIST_GPU_REGION_TIMING='1',
                   POLYGEIST_GPU_REGION_TIMING_FUNCTION=fn)
        rc = command([SCRIPTS / 'build_jetson.sh', d / 'export/abi.mlir',
                      d / 'raised', d / 'export/wrapper.o', d / 'export/harness.o',
                      d / 'export/polybench.o', OUT / 'iteration_driver.o'], log, env)
        row['gpu_build_rc'] = rc
    if not rc:
        row['raised_sha256'] = sha(d / 'raised')
    native = [['aarch64-linux-gnu-gcc', *flags, '-Dmain=polygeist_polybench_iteration',
               '-include', ROOT / 'runtime/polybench_repeat_entry.h',
               '-c', source, '-o', d / 'reference.o'],
              ['aarch64-linux-gnu-gcc', '-O3', *flags, '-c', PB/'utilities/polybench.c',
               '-o', d / 'polybench.o'],
              ['aarch64-linux-gnu-gcc', d/'reference.o', d/'polybench.o',
               OUT/'iteration_driver.o', '-lm', '-o', d/'reference']]
    for cmd in native:
        nr = command(cmd, log, env)
        if nr:
            break
    row['reference_build_rc'] = nr
    row['environment'] = {k: v for k, v in env.items() if k.startswith('POLYGEIST_')}
    row['compiler_sha256'] = sha(ROOT/'build/bin/polygeist-opt')
    row['runtime_source_sha256'] = sha(ROOT/'runtime/polygeist_cublas_rt_cuda.c')
    (d / 'build.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, 'raised', rc, 'reference', nr, 'matches', row.get('matched_calls', []), flush=True)
    return row

if __name__ == '__main__':
    subprocess.run(['aarch64-linux-gnu-gcc', '-O2', '-c', str(OUT/'iteration_driver.c'),
                    '-o', str(OUT/'iteration_driver.o')], check=True)
    for name in sys.argv[1:] or CASES:
        build(name)
