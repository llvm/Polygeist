#!/usr/bin/env python3
"""Build from C, run serially on the configured board, and audit Figure 4."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build import CASES, OUT, ROOT, sha

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote-helper', required=True,
                   help='Private adapter implementing run(command, timeout) and upload(path, destination)')
    p.add_argument('--use-built-executables', action='store_true')
    p.add_argument('--reuse-reference', action='store_true')
    p.add_argument('--reuse-upstream-audit', action='store_true')
    args = p.parse_args()
    if not args.use_built_executables:
        for script in ('build.py', 'build_native.py', 'build_paper_baselines.py'):
            subprocess.run([sys.executable, str(OUT/script)], cwd=ROOT, check=True)
    compiler = sha(ROOT/'build/bin/polygeist-opt')
    runtime = sha(ROOT/'runtime/polygeist_cublas_rt_cuda.c')
    for name in CASES:
        d = OUT/name
        meta = json.loads((d/'build.json').read_text())
        assert meta.get('source_build_rc') == meta.get('gpu_build_rc') == 0, name
        assert meta.get('reference_build_rc') == 0, 'Failed reference: '+name
        assert meta.get('compiler_sha256') == compiler, 'Stale compiler: '+name
        assert meta.get('runtime_source_sha256') == runtime, 'Stale runtime: '+name
        assert meta.get('raised_sha256') == sha(d/'raised'), 'Stale binary: '+name
        for variant in (['native', 'paper-baseline'] if name in ('correlation', 'covariance') else ['native']):
            baseline = json.loads((d/(variant+'-build.json')).read_text())
            assert baseline['build_rc'] == 0, 'Failed baseline: '+name
            assert baseline['executable_sha256'] == sha(d/variant), 'Stale baseline: '+name
        if args.reuse_reference:
            ref = json.loads((d/'reference-run.json').read_text())
            assert ref['rc'] == 0 and ref['sha256'] == sha(d/'reference'), 'Stale reference: '+name
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    order = []
    failures = []
    def run(name, variant):
        d = OUT/name
        archive = d/'attempts'/stamp
        archive.mkdir(parents=True, exist_ok=True)
        for suffix in ('.log', '-run.json', '-thermal.log', '-preflight.json'):
            path = d/(variant+suffix)
            if path.exists():
                shutil.move(path, archive/path.name)
        argv = [sys.executable, str(OUT/'run.py'), '--remote-helper', args.remote_helper, name, variant]
        rc = subprocess.run(argv, cwd=ROOT).returncode
        order.append(dict(kernel=name, variant=variant, exit_code=rc))
        if rc:
            failures.append((name, variant, rc))
        (OUT/'run-order.json').write_text(json.dumps(order,indent=2)+'\n')
    for i, name in enumerate(CASES):
        if not args.reuse_reference:
            run(name, 'reference')
        baseline = 'paper-baseline' if name in ('correlation', 'covariance') else 'native'
        variants = ['raised', baseline] if i % 2 == 0 else [baseline, 'raised']
        for variant in variants:
            run(name, variant)
        if baseline == 'paper-baseline' and not args.reuse_upstream_audit:
            run(name, 'native')
    subprocess.run([sys.executable, str(OUT/'audit.py')], check=True)
    print('Execution failures:', failures, flush=True)
    raise SystemExit(bool(failures))

if __name__ == '__main__':
    main()
