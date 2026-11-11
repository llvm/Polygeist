"""Relink unchanged proved compiler objects after a harness-only correction."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

D = Path(__file__).resolve().parent / 'pcg-application'
R = D.parents[2]
W = D / 'work'
archive = D / 'before-finite-check'
archive.mkdir(exist_ok=True)
for name in ('program', 'build.json', 'result.json', 'run.log'):
    if (D / name).exists():
        shutil.copy2(D / name, archive / name)
objects = [W / n for n in ('kernel.o', 'wrapper.o', 'rt.o', 'mlir_runner_utils.o', 'spec_runtime.o')]
original = {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}
old = json.loads((D / 'build.json').read_text())
flags = [x for x in old['command'] if x.startswith('-D')]
harness = R / 'issues/mfem_c_kernels/application_extractions/mfem_application_reference_harness.c'
cmd = ['aarch64-linux-gnu-gcc', *flags, '-O3', '-fno-inline', '-fno-inline-functions',
       '-fsemantic-interposition', '-fno-ipa-cp', '-fno-ipa-cp-clone', '-fno-ipa-sra',
       '-c', str(harness), '-o', str(W / 'harness.o')]
link = ['aarch64-linux-gnu-gcc', '-O2', *map(str, objects), str(W / 'harness.o'),
        '-L/usr/local/cuda-12.6/targets/sbsa-linux/lib',
        '-L/usr/local/cuda-12.6/targets/sbsa-linux/lib/stubs', '-L/usr/lib/aarch64-linux-gnu',
        '-lcudnn', '-lcublasLt', '-lcublas', '-lcufft', '-lcusparse', '-lcusolver',
        '-lcudart', '-lm', '-lpthread', '-ldl', '-Wl,--gc-sections', '-o', str(D / 'program')]
with (D / 'harness-relink.log').open('w') as log:
    for command in (cmd, link):
        subprocess.run(command, stdout=log, stderr=log, check=True)
assert all(hashlib.sha256((R / p).read_bytes()).hexdigest() == h for p, h in original.items())
old.update(program_sha256=hashlib.sha256((D / 'program').read_bytes()).hexdigest(),
           harness_relink=dict(commands=[cmd, link], unchanged_objects=original,
                               harness_dependencies={str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in (harness, harness.with_name('mfem_application_jetson_harness.c'))}))
(D / 'build.json').write_text(json.dumps(old, indent=2) + '\n')
print('Relinked corrected harness; five compiler/runtime objects unchanged')
