"""Check full in-place outputs, exact aliases and rejected partial overlap on Orin."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
D = OUT / 'inplace_add'
RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
loader = importlib.util.spec_from_file_location(
    'board', Path.home() / '.config/polygeist-silicon/reproduction_20261109/remote.py')
B = importlib.util.module_from_spec(loader)
loader.loader.exec_module(B)
ENV = ('LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:'
       '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu ')

def preflight(label):
    rc, log = B.run("python3 - <<'PY'\n" +
                    (ROOT / 'scripts/correctness/fig3_board_preflight.py').read_text() + '\nPY', timeout=35)
    (D / ('board-' + RUN_ID + '-' + label + '.json')).write_text(log)
    assert rc == 0

def save():
    (OUT / 'inplace-gpu-results.json').write_text(json.dumps([row], indent=2) + '\n')

(D / ('host-' + RUN_ID + '-uptime.txt')).write_text(subprocess.check_output(['uptime'], text=True))
build = json.loads((OUT / 'inplace-build-results.json').read_text())[0]
assert build['rc'] == 0
row = dict(name='add', run_id=RUN_ID, sizes=build['sizes'], build_rc=0,
           program_sha256=hashlib.sha256((D / 'program').read_bytes()).hexdigest(), correctness={})
preflight('before')
remote = '/tmp/aten-inplace-add-20261111'
assert B.upload(D / 'program', remote) == 0
rc, log = B.run('chmod +x ' + remote, timeout=15)
assert rc == 0
for kind in ['normal', 'edge']:
    rc, log = B.run(ENV + 'POLYGEIST_V2_TRACE=1 POLYGEIST_RT_TIMING=1 timeout 60 ' + remote +
                    (' edge' if kind == 'edge' else ''), timeout=65)
    (D / (kind + '-correctness.log')).write_text(log)
    checks = [dict(edge=int(e), alias=int(a), resident=int(r), count=int(n), bad=int(b),
                   vendorbad=int(v), maxabs=float(x), maxrel=float(y))
              for e,a,r,n,b,v,x,y in re.findall(
                  r'CHECK edge=(\d+) alias=(\d+) resident=(\d+) count=(\d+) bad=(\d+) '
                  r'vendorbad=(\d+) maxabs=([\deE.+-]+) maxrel=([\deE.+-]+)', log)]
    calls = len(re.findall(r'POLYGEIST_V2_CALL.*entry=cudnnBackendExecute', log))
    row['correctness'][kind] = dict(rc=rc, passed=rc == 0 and 'PASS correctness' in log
        and len(checks) == 4 and calls >= 4 and all(c['bad'] == c['vendorbad'] == 0 for c in checks),
        checks=checks, vendor_calls=calls)
    print(kind, row['correctness'][kind]['passed'], flush=True)
    save()
rc, log = B.run(ENV + 'POLYGEIST_V2_TRACE=1 timeout 60 ' + remote + ' partial', timeout=65)
(D / 'partial-overlap.log').write_text(log)
row['partial_overlap'] = dict(rc=rc, passed=rc != 0 and 'unsupported output overlap' in log
                              and 'POLYGEIST_V2_CALL' not in log)
print('partial overlap rejected', row['partial_overlap']['passed'], flush=True)
save()
if all(c['passed'] for c in row['correctness'].values()) and row['partial_overlap']['passed']:
    rc, log = B.run(ENV + 'timeout 60 ' + remote + ' timing', timeout=65)
    (D / 'timing.log').write_text(log)
    samples = [dict(path=p, sample=int(i), ms=float(t)) for p,i,t in re.findall(
        r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)', log)]
    row['timing'] = dict(rc=rc, passed=rc == 0 and len(samples) == 10 and 'PASS timing' in log,
                         samples=samples,
                         statistic='minimum of five synchronized resident samples after five warmups',
                         baseline='direct cuDNN graph from imported reference; not native ATen')
    if row['timing']['passed']:
        row['timing']['minimum_ms'] = {p:min(s['ms'] for s in samples if s['path']==p)
                                       for p in ['raised','direct_cudnn']}
    save()
preflight('after')
print('Timing', row.get('timing', {}).get('passed', False), flush=True)
