"""Run the frozen 43-case ATen comparison serially under an exclusive Orin reservation."""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN_ID = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
loader = importlib.util.spec_from_file_location('board', Path.home() / '.config/polygeist-silicon/reproduction_20261109/remote.py')
B = importlib.util.module_from_spec(loader)
loader.loader.exec_module(B)
ENV = ('LD_LIBRARY_PATH=/home/nvidia/polygeist_aten_native_24_12:'
       '/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:'
       '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu ')


def preflight(label):
    rc, log = B.run("python3 - <<'PY'\n" + (ROOT / 'scripts/correctness/fig3_board_preflight.py').read_text() + '\nPY', timeout=40)
    (OUT / (RUN_ID + '-board-' + label + '.json')).write_text(log)
    if rc:
        raise RuntimeError('Board preflight failed')


def write_results(rows):
    (OUT / 'comparison-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    keys = ['name', 'normal_pass', 'edge_pass', 'timing_pass', 'raised_min_ms', 'native_min_ms',
            'raised_median_ms', 'native_median_ms', 'native_over_raised_min', 'qualification']
    with (OUT / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


if __name__ == '__main__':
    (OUT / (RUN_ID + '-host-uptime.txt')).write_text(subprocess.check_output(['uptime'], text=True))
    preflight('before')
    rc, log = B.run(ENV + 'ldd /tmp/aten43-native-abs; uname -a; cat /etc/nv_tegra_release 2>/dev/null', timeout=30)
    (OUT / (RUN_ID + '-native-libraries.log')).write_text(log)
    rows = []
    for build in json.loads((OUT / 'build-results.json').read_text()):
        name = build['name']
        directory = OUT / 'comparison' / name
        row = dict(name=name, run_id=RUN_ID, build_rc=build['rc'], qualification='')
        rows.append(row)
        if build['rc']:
            row['qualification'] = 'build failed'
            write_results(rows)
            continue
        program = directory / 'program'
        row['program_sha256'] = hashlib.sha256(program.read_bytes()).hexdigest()
        assert row['program_sha256'] == build['comparison_program_sha256']
        remote = '/tmp/aten43-native-' + name
        assert B.upload(program, remote) == 0
        rc, log = B.run('chmod +x ' + remote + '; sha256sum ' + remote, timeout=15)
        (directory / 'remote-sha256.log').write_text(log)
        assert rc == 0 and row['program_sha256'] in log
        row['correctness'] = {}
        for kind in ['normal', 'edge']:
            rc, log = B.run(ENV + 'POLYGEIST_V2_TRACE=1 timeout 75 ' + remote + (' edge' if kind == 'edge' else ''), timeout=80)
            (directory / (kind + '-native.log')).write_text(log)
            checks = []
            for line in log.splitlines():
                if line.startswith('CHECK '):
                    checks.append({k: float(v) if any(c in v for c in '.eE') else int(v)
                                   for k, v in re.findall(r'(\w+)=([+\-\d.eE]+)', line)})
            passed = rc == 0 and 'PASS correctness' in log and bool(checks) and all(c['bad'] == c['nativebad'] == 0 for c in checks)
            row[kind + '_pass'] = passed
            row['correctness'][kind] = dict(rc=rc, passed=passed, checks=checks,
                mismatches=[line for line in log.splitlines() if 'MISMATCH' in line],
                log=str(directory / (kind + '-native.log')))
            write_results(rows)
        rc, log = B.run(ENV + 'timeout 75 ' + remote + ' timing', timeout=80)
        (directory / 'timing-native.log').write_text(log)
        samples = [dict(path=a, sample=int(b), ms=float(c)) for a, b, c in
                   re.findall(r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)', log)]
        row['timing_pass'] = rc == 0 and len(samples) == 10 and 'PASS timing' in log
        row['timing'] = dict(rc=rc, samples=samples,
            interval='synchronized host latency in same compiled C harness; CUDA-resident tensors; five warmups/five samples per path; alternating order',
            native_api_output='allocated' if name == 'rms_norm' else 'preallocated',
            raised_api_output='preallocated', inplace_reset='outside each timed interval' if name == 'add' else None)
        if row['timing_pass']:
            for path, prefix in [('raised', 'raised'), ('native_aten_cuda', 'native')]:
                values = [s['ms'] for s in samples if s['path'] == path]
                row[prefix + '_min_ms'] = min(values)
                row[prefix + '_median_ms'] = statistics.median(values)
            if name == 'rms_norm':
                row['qualification'] = 'Native rms_norm allocates its output; raised output is preallocated. Do not treat latency ratio as allocation-matched speedup.'
            elif row['normal_pass']:
                row['native_over_raised_min'] = row['native_min_ms'] / row['raised_min_ms']
            else:
                row['qualification'] = 'Normal correctness/parameter suite incomplete or mismatching; timing ratio not accepted.'
            if not row['edge_pass']:
                row['qualification'] += ' Existing edge-suite disagreement requires investigation.'
        print(name, 'normal', row['normal_pass'], 'edge', row['edge_pass'], 'timing', row['timing_pass'], flush=True)
        (OUT / 'STATUS.txt').write_text(f'Exclusive Orin comparison running: {len(rows)}/43 cases measured. '
            f'{sum(r.get("normal_pass", False) for r in rows)} normal, {sum(r.get("edge_pass", False) for r in rows)} edge, '
            f'{sum(r.get("timing_pass", False) for r in rows)} timing suites passed. Unresolved comparisons retained, not hidden.\n')
        write_results(rows)
    preflight('after')
    print('FINISHED', len(rows), 'cases', flush=True)
