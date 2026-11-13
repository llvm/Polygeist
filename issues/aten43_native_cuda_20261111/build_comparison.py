"""Relink frozen, validated raised objects with genuine preallocated ATen APIs."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import re
import subprocess

OUT = Path(__file__).resolve().parent
TORCH = Path('/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch')
CUDA = Path('/usr/local/cuda-12.6/targets/sbsa-linux')
CASES = json.loads((OUT / 'frozen-43-results.json').read_text())


def replace_block(text, start, replacement):
    pos = text.index(start)
    begin = text.index('{', pos)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[:pos] + replacement + text[end:]


def build(row):
    name = row['name']
    source = OUT / 'frozen' / name
    directory = OUT / 'comparison' / name
    directory.mkdir(parents=True, exist_ok=True)
    text = (source / 'harness.c').read_text()
    if '#define NA ' not in text:
        extra = '#define NA 2\n#define NS 0\nstatic float parameters[1];\nstatic const size_t sizes[]={N,N};\n'
        text = text.replace('static void direct(', extra + 'static void direct(', 1)
    bridge = '''void *native_prepare(const char*,float**,const size_t*,int);
void native_run(void*,const float*);void native_export(void*);
static void *native_context; static float *native_pointers[NA];
static void direct(float **p){
 int change=!native_context;for(int a=0;a<NA;a++)change|=(p[a]!=native_pointers[a]);
 if(change){native_context=native_prepare("NAME",p,sizes,NA);for(int a=0;a<NA;a++)native_pointers[a]=p[a];}
 native_run(native_context,parameters);
}
'''.replace('NAME', name)
    text = replace_block(text, 'static void direct(', bridge)
    timing_start = 'if(timing)' if name == 'add' else 'if(argc>1&&!strcmp(argv[1],"timing"))'
    reset = ''
    if name == 'add':
        reset = 'for(int a=0;a<NA;a++)CU(cudaMemcpy(d[a],initial[a],sizes[a]*sizeof(float),cudaMemcpyHostToDevice));'
    timing = '''if(argc>1&&!strcmp(argv[1],"timing")){
 for(int k=0;k<2;k++)for(int w=0;w<5;w++){RESET if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());}
 for(int i=0;i<5;i++)for(int z=0;z<2;z++){
  int k=z^(i&1);RESET CU(cudaDeviceSynchronize());double t=now_ms();
  if(k)direct(d);else invoke(d);CU(cudaDeviceSynchronize());
  printf("SAMPLE path=%s sample=%d ms=%.9f\\n",k?"native_aten_cuda":"raised",i,now_ms()-t);
 }
 puts("PASS timing");return 0;
}'''.replace('RESET', reset)
    text = replace_block(text, timing_start, timing)
    # Export only allocating API outputs, after correctness dispatch and outside timing.
    text = text.replace('reference(r);direct(d);', 'reference(r);direct(d);native_export(native_context);')
    text = text.replace('reference(rp);direct(dp);', 'reference(rp);direct(dp);native_export(native_context);')
    # Preserve the fixture's existing CPU tolerance and NaN/Inf/signed-zero policy.
    # Native ATen is independent: compare with that same tolerance, not vendor bit equality.
    text = text.replace('if(!equal(got[i+4],v[a][i],1))vendorbad++;',
        'if(!equal(got[i+4],v[a][i],0)){if(vendorbad<6)printf("NATIVE_MISMATCH i=%zu raised=%a native=%a\\n",(size_t)i,got[i+4],v[a][i]);vendorbad++;}')
    text = text.replace('if(!equal(got,vendor,1))vendorbad++;',
        'if(!equal(got,vendor,!output)){if(vendorbad<6)printf("NATIVE_MISMATCH i=%zu raised=%a native=%a\\n",i,got,vendor);vendorbad++;}')
    text = text.replace('vendorbad', 'nativebad')
    (directory / 'harness.c').write_text(text)
    compile_cmd = ['aarch64-linux-gnu-gcc', '-O2', '-I' + str(CUDA / 'include'),
                   '-c', str(directory / 'harness.c'), '-o', str(directory / 'harness.o')]
    objects = [source / 'work' / (n + '.o') for n in
               ['kernel', 'wrapper', 'rt', 'spec_runtime', 'mlir_runner_utils']]
    objects += [source / 'reference.o', directory / 'harness.o', OUT / 'native_bridge.o']
    link_cmd = ['aarch64-linux-gnu-g++', '-O2', *map(str, objects),
        '-L' + str(TORCH / 'lib'), '-L' + str(CUDA / 'lib'), '-L' + str(CUDA / 'lib/stubs'),
        '-L/usr/lib/aarch64-linux-gnu', '-Wl,--allow-shlib-undefined',
        '-Wl,--no-as-needed', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10_cuda', '-lc10',
        '-lcudnn', '-lcublasLt', '-lcublas', '-lcusolver', '-lcudart',
        '-lm', '-lpthread', '-ldl', '-Wl,--gc-sections',
        '-Wl,-rpath,/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/jetson-cuda-libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu',
        '-o', str(directory / 'program')]
    if name in ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cosh', 'sinh', 'tan']:
        link_cmd[3:3] = ['-L/home/arjaiswal/cutensor_sbsa/lib']
        link_cmd += ['-l:libcutensor.so.2']
    with (directory / 'build.log').open('w') as log:
        for cmd in [compile_cmd, link_cmd]:
            rc = subprocess.run(cmd, stdout=log, stderr=log).returncode
            if rc:
                break
    result = dict(name=name, rc=rc, commands=[compile_cmd, link_cmd],
                  frozen_program_sha256=row['program_sha256'])
    if rc == 0:
        result['comparison_program_sha256'] = hashlib.sha256((directory / 'program').read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(build, CASES))
    (OUT / 'build-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('Built', sum(r['rc'] == 0 for r in rows), '/', len(rows))
    for row in rows:
        if row['rc']:
            print(row['name'], 'failed; see comparison/' + row['name'] + '/build.log')
