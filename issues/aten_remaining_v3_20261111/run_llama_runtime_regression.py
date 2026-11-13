"""Compare frozen Llama kernels with old/current runtime in one GPU window."""
from pathlib import Path
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'issues/llama_rank_collapse_20261111'
D = OUT / 'llama-runtime-regression'
sys.path.insert(0, str(ROOT / 'scripts/correctness'))
from compare_llama_outputs import read_logits, compare
loader = importlib.util.spec_from_file_location(
    'board', Path.home() / '.config/polygeist-silicon/reproduction_20261109/remote.py')
B = importlib.util.module_from_spec(loader)
loader.loader.exec_module(B)
ENV = ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLYGEIST_CUDA_GRAPH=1 '
       'LD_LIBRARY_PATH=/tmp/llama-current-v3-20261111:'
       '/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:'
       '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu ')

def preflight(label):
    rc, log = B.run("python3 - <<'PY'\n" +
                    (ROOT / 'scripts/correctness/fig3_board_preflight.py').read_text() +
                    '\nPY', timeout=35)
    (D / ('board-' + label + '.json')).write_text(log)
    assert rc == 0

(D / 'host-uptime.txt').write_text(subprocess.check_output(['uptime'], text=True))
preflight('before')
rows = []
for name, binary in [('before', OLD / 'candidate/llama_resident'),
                     ('after', D / 'llama_resident')]:
    remote = '/tmp/aten-runtime-llama-' + name
    assert B.upload(binary, remote) == 0
    rc, log = B.run('chmod +x ' + remote, timeout=15)
    assert rc == 0
    rc, log = B.run(ENV + 'timeout 90 taskset -c 0 ' + remote, timeout=100)
    (D / (name + '.log')).write_text(log)
    samples = [float(line.split(',')[-1]) for line in log.splitlines()
               if line.startswith('RESIDENT_SAMPLE,')]
    row = dict(name=name, binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
               rc=rc, samples_ms=samples)
    if samples:
        row.update(minimum_ms=min(samples), median_ms=statistics.median(samples))
    rows.append(row)
    (D / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(name, 'rc', rc, 'samples', len(samples), 'minimum_ms', row.get('minimum_ms'), flush=True)
    assert rc == 0 and len(samples) == 5
preflight('after')
outputs = {name: read_logits(D / (name + '.log'), 32000) for name in ['before', 'after']}
expert = read_logits(OLD / 'expert.log', 32000)
comparisons = {'after_vs_before': compare(outputs['after'], outputs['before']),
               'after_vs_expert': compare(outputs['after'], expert)}
(D / 'correctness.json').write_text(json.dumps(comparisons, indent=2) + '\n')
print(json.dumps(comparisons), flush=True)
assert all(row['strict_failures'] == 0 for row in comparisons.values())
