#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def stats(xs):
    q = statistics.quantiles(xs, n=4, method='inclusive')
    return dict(samples_ms=xs, min_ms=min(xs), median_ms=statistics.median(xs),
                max_ms=max(xs), iqr_ms=q[2]-q[0])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote-helper', required=True)
    p.add_argument('kernel')
    p.add_argument('variant')
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('board', args.remote_helper)
    board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(board)
    d = HERE / args.kernel
    exe = d / args.variant
    if args.variant == 'raised':
        build = json.loads((d/'build.json').read_text())
        if (build.get('source_build_rc') != 0 or build.get('gpu_build_rc') != 0 or
                build.get('raised_sha256') != hashlib.sha256(exe.read_bytes()).hexdigest()):
            raise SystemExit('Refusing stale or failed raised build')
    remote = '/tmp/fig4-gpu-20261110'
    target = remote + '/' + args.kernel + '-' + args.variant
    def clean(text):
        for key, value in getattr(board, 'c', {}).items():
            if isinstance(value, str) and len(value)>2 and ('HOST' in key or 'PASS' in key):
                text = text.replace(value, '[private]')
        return text
    probe = (ROOT/'scripts/correctness/fig3_board_preflight.py').read_text()
    rc, out = board.run("python3 - <<'PY'\n"+probe+"\nPY",timeout=40)
    (d/(args.variant+'-preflight.json')).write_text(clean(out))
    if rc:
        raise SystemExit('Board preflight rejected; no executable started')
    rc, out = board.run('mkdir -p '+remote,timeout=30)
    if rc or board.upload(exe,target):
        raise SystemExit('Deployment failed')
    monitor = target+'.thermal'
    command = ('ulimit -c 0; chmod +x '+target+'; '
               'export LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu; '
               'tegrastats --interval 100 > '+monitor+' 2>&1 & monitor=$!; '
               'taskset -c 0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
               'POLYGEIST_CUDA_GRAPH=0 timeout 900 '+target+'; '
               'rc=$?; kill "$monitor" 2>/dev/null; wait "$monitor" 2>/dev/null; exit "$rc"')
    start=time.time()
    executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()
    host_uptime=subprocess.check_output(['uptime'],text=True).strip()
    rc,out=board.run(command,timeout=940)
    (d/(args.variant+'.log')).write_text(clean(out))
    tr,thermal=board.run('cat '+monitor,timeout=30)
    (d/(args.variant+'-thermal.log')).write_text(clean(thermal))
    row=dict(kernel=args.kernel,variant=args.variant,rc=rc,elapsed_s=time.time()-start,
             sha256=executable_sha256,cuda_graph=False,started_at_unix=start,
             host_uptime=host_uptime)
    records=re.findall(r'POLYGEIST_GPU_REGION_TIMING[^\n]+',out)
    if records:
        for key in ('compute_device_ms','compute_wall_ms','memory_device_ms','e2e_host_ms'):
            xs=[float(re.search(r'\b'+key+r'=([0-9.]+)',line)[1]) for line in records]
            row[key]={'all_iterations':xs}
            if len(xs)==10:row[key].update(stats(xs[5:]))
    xs=[float(x) for x in re.findall(r'POLYBENCH_NATIVE_GPU_TIMING device_ms=([0-9.]+)',out)]
    if xs:
        row['compute_device_ms']={'all_iterations':xs}
        if len(xs)==10:row['compute_device_ms'].update(stats(xs[5:]))
    xs=[float(x) for x in re.findall(r'POLYBENCH_NATIVE_GPU_WALL_MS ([0-9.]+)',out)]
    if xs:
        row['compute_wall_ms']={'all_iterations':xs}
        if len(xs)==10:row['compute_wall_ms'].update(stats(xs[5:]))
    row['warmups']=out.count('phase=warmup')
    row['measured']=out.count('phase=sample')
    row['dump_count']=out.count('==BEGIN DUMP_ARRAYS==')
    temperatures=[float(x) for x in re.findall(r'@([0-9.]+)C',thermal)]
    row['max_temperature_c']=max(temperatures or [0])
    row['thermal_ok']=tr==0 and bool(temperatures) and row['max_temperature_c']<85
    (d/(args.variant+'-run.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(args.kernel,args.variant,'rc',rc,'samples',row['measured'],'dumps',row['dump_count'],
          'device_ms',row.get('compute_device_ms',{}).get('min_ms'),flush=True)
    raise SystemExit(rc)

if __name__=='__main__':main()
