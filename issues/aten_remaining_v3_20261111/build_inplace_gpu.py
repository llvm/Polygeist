"""Build the existing in-place fixture; obtain vendor arguments from its proved launch."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
D = OUT / 'inplace_add'
sys.path[:0] = [str(ROOT / 'scripts/correctness'), str(ROOT / 'scripts/correctness/lib_v2')]
import abi_v2 as A
import mlir_lite as ML
import spec_match as S

os.environ['POLYGEIST_NUMERIC_MODEL'] = 'strict'
text = (D / 'matched.mlir').read_text()
proof = json.loads((D / 'match.json').read_text())
S.verify(text, proof, backend='cuda')
assert proof['emitted'] == ['aten_add_f32']
module = ML.Module(text)
func = next(f for f in module.funcs if f.name == 'aten_add')
launch = next(op for op in func.ops if op.name == 'kernel.launch')
calls, specs, errors = A.build_table(proof['emitted'])
assert not errors, errors
cfg = next(iter(calls.values())).runtime_adapter
graph = cfg['graph']
bindings = dict(zip([a.name for a in specs['aten_add_f32'].args], launch.operands))
defs = {r: op for op in func.ops for r in op.results}
formals = {arg: i for i, (arg, _) in enumerate(func.args)}

def expression(value):
    if value in formals:
        return 'p[%d]' % formals[value]
    op = defs[value]
    if op.name == 'arith.constant':
        return re.search(r'constant\s+(-?\d+)\s*:', op.text)[1]
    if op.name in ('arith.index_cast', 'bufferization.to_tensor', 'memref.collapse_shape'):
        return expression(op.operands[0])
    raise ValueError('Unsupported proved argument: ' + op.name)

source = ROOT / 'issues/aten_c_kernels/aten_add.c'
preprocessed = subprocess.check_output(['gcc', '-E', '-P', str(source)], text=True)
signature = re.search(r'void aten_add\(([^)]*)\)', preprocessed)[1]
sizes = [math.prod(map(int, re.findall(r'\[(\d+)\]', p))) for p in signature.split(',')]
assert len(sizes) == 2 and sizes[0] == sizes[1]
args = [expression(bindings[cfg['extent_arg']])] + [str(v) + 'LL' for v in graph['words']]
args += [str(graph['nodes'])] + [repr(float(v)) + 'f' for v in graph['scalars']]
assert not graph.get('scalar_arguments')
args += ['0.0f'] * (8 - len(graph['scalars'])) + ['1'] * 5
args += [expression(bindings[p]) for p in graph['inputs']] + ['NULL'] * (4 - len(graph['inputs']))
args += [expression(bindings[graph['output']])]
header = ('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n'
          '#include <string.h>\n#include <time.h>\n#include <cuda_runtime_api.h>\n'
          '#include "' + str(ROOT / 'runtime/polygeist_cublas_rt.h') + '"\n')
header += '#define N %d\n' % sizes[0]
header += 'void aten_add(float*,float*);void native_reference(float*,float*);\n'
header += 'static void invoke(float **p){aten_add(p[0],p[1]);}\n'
header += 'static void reference(float **p){native_reference(p[0],p[1]);}\n'
header += ('static void direct(float **p){polygeist_require_gpu_libraries_begin();'
           'polygeist_cublas_pipeline_begin();' + cfg['entry'] + '(' + ','.join(args) + ');'
           'polygeist_cublas_pipeline_end();polygeist_require_gpu_libraries_end();}\n')
(D / 'harness.c').write_text(header + (OUT / 'inplace_harness.c').read_text())
env = dict(os.environ, PYTHON='/usr/bin/python3',
           PYTHONPATH=str(ROOT / 'scripts/correctness') + ':' + str(ROOT / 'scripts/correctness/lib_v2'),
           POLYGEIST_BUILD_WORK_DIR=str(D / 'work'), POLYGEIST_MATCH_PROOF=str(D / 'match.json'),
           POLYGEIST_CUSTOM_CUDA_OBJ=str(D / 'reference.o'), POLYGEIST_MINIMAL_CUDNN_RUNTIME='1')
commands = [['aarch64-linux-gnu-gcc', '-O0', '-ffp-contract=off', '-Daten_add=native_reference',
             '-c', str(source), '-o', str(D / 'reference.o')],
            [str(ROOT / 'scripts/correctness/polygeist_build.sh'), '--target=jetson',
             '--function=aten_add', '--semantic-mlir=' + str(D / 'matched.mlir'),
             '--harness=' + str(D / 'harness.c'), '-o', str(D / 'program'), str(source),
             '-I/usr/local/cuda-12.6/targets/sbsa-linux/include', '-I/usr/include/aarch64-linux-gnu']]
with (D / 'build.log').open('w') as log:
    for command in commands:
        result = subprocess.run(command, env=env, stdout=log, stderr=log)
        if result.returncode:
            break
row = dict(name='add', sizes=sizes, rc=result.returncode,
           proof_sha256=hashlib.sha256((D / 'match.json').read_bytes()).hexdigest())
(OUT / 'inplace-build-results.json').write_text(json.dumps([row], indent=2) + '\n')
print('In-place add build:', result.returncode, flush=True)
raise SystemExit(result.returncode)
