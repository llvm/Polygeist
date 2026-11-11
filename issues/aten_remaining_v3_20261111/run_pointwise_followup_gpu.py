"""Run the additional pointwise-followup pointwise references on Orin; no native ATen timing claim."""
from pathlib import Path
import hashlib
import datetime
import importlib.util
import json
import re
import subprocess

RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
loader = importlib.util.spec_from_file_location(
    'board', Path.home() / '.config/polygeist-silicon/reproduction_20261109/remote.py')
B = importlib.util.module_from_spec(loader)
loader.loader.exec_module(B)
ENV = ('LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:'
       '/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:'
       '/usr/lib/aarch64-linux-gnu ')


def preflight(label):
    command = "python3 - <<'PY'\n" + (
        ROOT / 'scripts/correctness/fig3_board_preflight.py').read_text() + '\nPY'
    rc, log = B.run(command, timeout=40)
    (OUT / ('pointwise-followup-' + RUN_ID + '-board-' + label + '.json')).write_text(log)
    if rc:
        raise RuntimeError('Board preflight failed: ' + label)


def write_results():
    (OUT / 'pointwise-followup-gpu-results.json').write_text(json.dumps(rows, indent=2) + '\n')


(OUT / ('pointwise-followup-' + RUN_ID + '-host-uptime.txt')).write_text(subprocess.check_output(['uptime'], text=True))
preflight('before')
result_path = OUT / 'pointwise-followup-gpu-results.json'
rows = json.loads(result_path.read_text()) if result_path.exists() else []
for build in json.loads((OUT / 'pointwise-followup-build-results.json').read_text()):
    name = build['name']
    row = dict(run_id=RUN_ID, name=name, sizes=build['sizes'], scalars=build['scalars'], build_rc=build['rc'])
    if build['rc']:
        continue
    directory = OUT / 'pointwise-followup' / name
    if build['match_proof_sha256'] != hashlib.sha256((directory / 'match.json').read_bytes()).hexdigest():
        continue  # A refreshed proof is still waiting for its corresponding build.
    program = directory / 'program'
    row['program_sha256'] = hashlib.sha256(program.read_bytes()).hexdigest()
    if any(r['name'] == name and r.get('program_sha256') == row['program_sha256'] for r in rows):
        continue
    rows = [r for r in rows if r['name'] != name]
    rows.append(row)
    remote = '/tmp/aten-pointwise-followup-20261111-' + name
    assert B.upload(program, remote) == 0
    rc, log = B.run('chmod +x ' + remote, timeout=15)
    assert rc == 0, log
    row['correctness'] = {}
    for kind in ['normal', 'edge']:
        rc, log = B.run(ENV + 'POLYGEIST_V2_TRACE=1 POLYGEIST_RT_TIMING=1 '
                       'timeout 60 ' + remote + (' edge' if kind == 'edge' else ''), timeout=65)
        (directory / (kind + '-correctness.log')).write_text(log)
        checks = [dict(edge=int(a), parameters=int(p), resident=int(b), count=int(c), bad=int(d),
                       vendorbad=int(e), maxabs=float(f), maxrel=float(g))
                  for a, p, b, c, d, e, f, g in re.findall(
                      r'CHECK edge=(\d+) parameters=(\d+) resident=(\d+) count=(\d+) bad=(\d+) '
                      r'vendorbad=(\d+) maxabs=([\deE.+-]+) maxrel=([\deE.+-]+)', log)]
        calls = len(re.findall(r'POLYGEIST_V2_CALL.*entry=cudnnBackendExecute', log))
        row['correctness'][kind] = dict(
            rc=rc, passed=rc == 0 and 'PASS correctness' in log and len(checks) == (4 if build['scalars'] else 2)
            and calls >= len(checks) and all(c['bad'] == c['vendorbad'] == 0 for c in checks),
            checks=checks, vendor_calls=calls,
            mismatches=[line for line in log.splitlines() if 'MISMATCH' in line])
        print(name, kind, row['correctness'][kind]['passed'], flush=True)
        write_results()
    if all(c['passed'] for c in row['correctness'].values()):
        rc, log = B.run(ENV + 'timeout 60 ' + remote + ' timing', timeout=65)
        (directory / 'timing.log').write_text(log)
        samples = [dict(path=a, sample=int(b), ms=float(c)) for a, b, c in re.findall(
            r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)', log)]
        row['timing'] = dict(rc=rc, samples=samples,
                             passed=rc == 0 and len(samples) == 10 and 'PASS timing' in log,
                             statistic='minimum of five synchronized resident samples after five warmups',
                             baseline='direct cuDNN graph from imported reference; not native ATen')
        if row['timing']['passed']:
            row['timing']['minimum_ms'] = {
                path: min(s['ms'] for s in samples if s['path'] == path)
                for path in ['raised', 'direct_cudnn']}
        write_results()
preflight('after')
print('Completed', len(rows), 'cases;',
      sum(r.get('correctness', {}).get('normal', {}).get('passed', False) for r in rows),
      'normal checks;',
      sum(r.get('correctness', {}).get('edge', {}).get('passed', False) for r in rows),
      'edge checks;', sum(r.get('timing', {}).get('passed', False) for r in rows),
      'timings', flush=True)
