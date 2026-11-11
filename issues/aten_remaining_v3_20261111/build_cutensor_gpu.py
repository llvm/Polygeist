"""Build proven pointwise replacements; direct vendor graphs use proof bindings."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT / 'scripts/correctness'), str(ROOT / 'scripts/correctness/lib_v2')]
import abi_v2 as A
import mlir_lite as ML

results_path = OUT / 'cutensor-build-results.json'
rows = json.loads(results_path.read_text()) if results_path.exists() else []
for match in json.loads((OUT / 'cutensor-matches.json').read_text()):
    name = match['name']
    if len(match.get('emitted', [])) != 1:
        continue
    directory = OUT / 'cutensor' / name
    proof_path = directory / 'match.json'
    proof_hash = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    if any(r['name'] == name and r['rc'] == 0 and r['match_proof_sha256'] == proof_hash
           for r in rows) and (directory / 'program').exists():
        continue
    rows = [r for r in rows if r['name'] != name]
    proof = json.loads(proof_path.read_text())
    spec = match['emitted'][0]
    evidence = proof['matches'][0]['details']
    bindings = {key.lstrip('%'): value for key, value in evidence['binding'].items()}
    source = ROOT / match['source']
    function = 'aten_' + name
    text = subprocess.check_output(['gcc', '-E', '-P', str(source)], text=True)
    prototype = re.search(r'void\s+' + function + r'\s*\(([^)]*)\)', text).group(1)
    parameters = []
    for parameter in prototype.split(','):
        found = re.fullmatch(r'\s*float\s+(\w+)\s*(?:\[\s*(\d+)\s*\])?\s*', parameter)
        if not found:
            raise ValueError('Unsupported harness parameter: ' + parameter)
        parameters.append(dict(name=found[1], extent=int(found[2]) if found[2] else None))
    arrays = [i for i, p in enumerate(parameters) if p['extent'] is not None]
    scalars = [i for i, p in enumerate(parameters) if p['extent'] is None]
    arguments = ['p[%d]' % arrays.index(i) if i in arrays else 'parameters[%d]' % scalars.index(i)
                 for i in range(len(parameters))]
    sizes = [parameters[i]['extent'] for i in arrays]
    module = ML.Module((directory / 'source_debuf.mlir').read_text())
    func = next(f for f in module.funcs if f.name == function)
    definitions = {r: op for op in func.ops for r in op.results}
    formals = {arg: i for i, (arg, _) in enumerate(func.args)}

    def expression(value):
        value = value.removesuffix(' (in-place)')
        if value.startswith('literal '):
            return repr(float(value.split()[1])) + 'f'
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', value):
            return value
        if value.startswith('index_cast('):
            return expression(value[len('index_cast('):-1])
        if value in formals:
            return arguments[formals[value]]
        op = definitions.get(value)
        if op and op.name == 'arith.constant':
            number = re.search(r'arith.constant\s+([^ :]+)', op.text)[1]
            return repr(float(number)) + 'f'
        if op and op.name in ('bufferization.to_tensor', 'bufferization.to_memref',
                               'arith.index_cast', 'polygeist.submap'):
            return expression(op.operands[0])
        raise ValueError('Unsupported proof binding: ' + value)

    calls, _, errors = A.build_table([spec])
    assert not errors, errors
    graph = next(iter(calls.values())).runtime_adapter['primitive']
    output = expression(bindings[graph['output']])
    output_index = arguments.index(output)
    output_array = arrays.index(output_index)
    header = ('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n'
              '#include <stdint.h>\n#include <string.h>\n#include <time.h>\n'
              '#include <cuda_runtime_api.h>\n#include "' + str(ROOT / 'runtime/polygeist_cublas_rt.h') + '"\n')
    header += f'#define NA {len(arrays)}\n#define NS {len(scalars)}\n'
    header += 'static float parameters[NS ? NS : 1];\n'
    header += 'static const size_t sizes[]={' + ','.join(map(str, sizes)) + '};\n'
    header += 'static const int outputs[]={' + ','.join(str(int(i == output_array)) for i in range(len(arrays))) + '};\n'
    cparams = ','.join('float*' if i in arrays else 'float' for i in range(len(parameters)))
    callargs = ','.join(arguments)
    header += f'void {function}({cparams}); void native_reference({cparams});\n'
    header += f'static void invoke(float **p){{{function}({callargs});}}\n'
    header += f'static void reference(float **p){{native_reference({callargs});}}\n'
    graph_args = [graph['opcode'], expression(bindings['n']),
                  expression(bindings[graph['input']]), expression(bindings[graph['output']])]
    header += ('static void direct(float **p){polygeist_require_gpu_libraries_begin();'
               'polygeist_cublas_pipeline_begin();polygeist_cutensor_unary_f32('
               + ','.join(graph_args) + ');polygeist_cublas_pipeline_end();'
               'polygeist_require_gpu_libraries_end();}\n')
    (directory / 'harness.c').write_text(header + (OUT / 'cutensor_harness.c').read_text())
    env = dict(os.environ, PYTHON='/usr/bin/python3', PYTHONPATH=str(ROOT/'scripts/correctness')+':'+str(ROOT/'scripts/correctness/lib_v2'), POLYGEIST_NUMERIC_MODEL='strict',
               POLYGEIST_BUILD_WORK_DIR=str(directory / 'work'),
               POLYGEIST_MATCH_PROOF=str(proof_path), POLYGEIST_CUSTOM_CUDA_OBJ=str(directory / 'reference.o'),
               POLYGEIST_MINIMAL_CUDNN_RUNTIME='1', POLYGEIST_CUTENSOR_ROOT='/home/arjaiswal/cutensor_sbsa')
    commands = [['aarch64-linux-gnu-gcc', '-O0', '-ffp-contract=off', '-D' + function + '=native_reference',
                 '-c', str(source), '-o', str(directory / 'reference.o')],
                [str(ROOT / 'scripts/correctness/polygeist_build.sh'), '--target=jetson', '--function=' + function,
                 '--semantic-mlir=' + str(directory / 'matched.mlir'), '--harness=' + str(directory / 'harness.c'),
                 '-o', str(directory / 'program'), str(source),
                 '-I/usr/local/cuda-12.6/targets/sbsa-linux/include', '-I/usr/include/aarch64-linux-gnu']]
    with (directory / 'build.log').open('w') as log:
        for command in commands:
            rc = subprocess.run(command, env=env, stdout=log, stderr=log, timeout=300).returncode
            if rc:
                break
    row = dict(name=name, sizes=sizes, scalars=len(scalars), spec=spec, rc=rc, graph=graph,
               commands=commands, match_proof_sha256=proof_hash)
    rows.append(row)
    (OUT / 'cutensor-build-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(name, rc, flush=True)
