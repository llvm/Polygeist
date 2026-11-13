#!/usr/bin/env python3
"""Rebuild ATen coverage with both IR forms used by the GPU build pipeline.

Counts distinguish an emitted call, a complete ABI mapping, and a candidate
with no residual host computation. None of these counts implies runtime
correctness. This uses the older phase-2 matcher, not the lib_v3 pipeline.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from aten_pointwise_graph_silicon import _device_unsafe_residuals
from inject_kernel_library import inject

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'issues/aten_c_kernels'
OPT = ROOT / 'build/bin/polygeist-opt'
CGEIST = ROOT / 'build/bin/cgeist'
MATCHER = ROOT / 'scripts/correctness/kernel_match_rewrite.py'
LIBRARY = ROOT / 'generic_solver/kernel_library_phase2.mlir'
LOOPS = r'\blinalg\.|\b(?:affine|scf)\.(?:for|parallel|while)\b|\bkernel\.launch\b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log, timeout=120):
    with log.open('w') as stream:
        try:
            return subprocess.run(list(map(str, command)), stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--kernel', action='append')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ([CORPUS / (name + '.c') for name in args.kernel] if args.kernel
               else sorted(CORPUS.glob('aten_*.c')))
    tracked = [OPT, CGEIST, MATCHER, LIBRARY,
               ROOT / 'scripts/correctness/kernel_match.py', Path(__file__)]
    before = {str(p.relative_to(ROOT)): digest(p) for p in tracked}
    library = LIBRARY.read_text()
    resource = subprocess.check_output(
        [ROOT / 'llvm-project/build/bin/clang', '-print-resource-dir'], text=True).strip()
    census = {r['kernel']: r for r in csv.DictReader(
        (CORPUS / 'native_cuda_external_library_calls.csv').open())}

    def build(source):
        name = source.stem
        out = args.output / name
        out.mkdir(exist_ok=True)
        row = dict(kernel=name, source_sha256=digest(source),
                   native_external=census[name]['audit_status'] == 'EXTERNAL_LIBRARY',
                   candidates=[])
        row['frontend_rc'] = run([CGEIST, source, '--function=' + name,
            '--resource-dir=' + resource, '--raise-scf-to-affine', '-S', '-o', out / 'source.mlir'],
            out / 'frontend.log')
        if row['frontend_rc']:
            return row
        for mode in ['preserved', 'normalized', 'fallback_auto']:
            if mode == 'fallback_auto' and any(
                    c.get('abi_rc') == 0 for c in row['candidates']):
                continue
            folder = out / mode
            folder.mkdir(exist_ok=True)
            candidate = dict(mode=mode, launches=0, complete_abi_candidate=False,
                             gpu_safe_candidate=False)
            row['candidates'].append(candidate)
            command = [OPT, '--select-func=func-name=' + name, '--remove-iter-args',
                       '--affine-parallelize', '--raise-affine-to-linalg-pipeline']
            if mode != 'preserved':
                command.append('--lower-polygeist-submap')
            debufferize = ('--linalg-debufferize' if mode == 'fallback_auto'
                           else '--linalg-debufferize=use-multi-root=true')
            command += [debufferize, out / 'source.mlir',
                        '-o', folder / 'raised.mlir']
            candidate['raise_rc'] = run(command, folder / 'raise.log')
            if candidate['raise_rc']:
                continue
            with (folder / 'matched.mlir').open('w') as stream, (folder / 'match.log').open('w') as log:
                try:
                    candidate['match_rc'] = subprocess.run(
                        [sys.executable, str(MATCHER), str(folder / 'raised.mlir'),
                         '--enable-structured-rewrite'], stdout=stream, stderr=log,
                        timeout=60).returncode
                except subprocess.TimeoutExpired:
                    candidate['match_rc'] = 124
            if candidate['match_rc']:
                continue
            text = (folder / 'matched.mlir').read_text()
            symbols = sorted(set(re.findall(r'kernel\.launch\s+@([\w.$]+)', text)))
            candidate['symbols'] = symbols
            candidate['launches'] = len(re.findall(r'\bkernel\.launch\s+@', text))
            candidate['text_residuals'] = len(re.findall(LOOPS, text)) - candidate['launches']
            (folder / 'with-definitions.mlir').write_text(inject(text, library, symbols) if symbols else text)
            candidate['abi_rc'] = run([OPT, folder / 'with-definitions.mlir',
                '--lower-kernel-launch-to-cublas', '--canonicalize', '-o', folder / 'abi.mlir'],
                folder / 'abi.log')
            if candidate['abi_rc']:
                continue
            lowered = (folder / 'abi.mlir').read_text()
            candidate['calls'] = sorted(set(re.findall(r'(?:func\.)?call\s+@(polygeist_[\w.$]+)', lowered)))
            candidate['residuals'] = _device_unsafe_residuals(lowered)
            candidate['complete_abi_candidate'] = bool(candidate['calls']) and not re.search(LOOPS, lowered)
            candidate['gpu_safe_candidate'] = candidate['complete_abi_candidate'] and not candidate['residuals']
        valid = [c for c in row['candidates'] if c.get('abi_rc') == 0]
        if valid:
            # Prefer verified whole mappings, then the same structural ranking
            # as polygeist_build.sh. Failed ABIs never count as matches.
            best = min(valid, key=lambda c: (not c['gpu_safe_candidate'],
                not c['complete_abi_candidate'], c['text_residuals'], -c['launches']))
            row['selected'] = best
            shutil.copyfile(out / best['mode'] / 'matched.mlir', out / 'matched.mlir')
        print(name, row.get('selected', {}).get('mode', 'failed'), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(build, sources))
    summary = dict(cases=len(rows), native_external_cases=sum(r['native_external'] for r in rows))
    for label, predicate in [
            ('any_call', lambda c: c.get('launches', 0) > 0),
            ('complete_abi_candidates', lambda c: c.get('complete_abi_candidate', False)),
            ('gpu_safe_candidates', lambda c: c.get('gpu_safe_candidate', False))]:
        summary[label] = sum(predicate(r.get('selected', {})) for r in rows)
        summary['native_' + label] = sum(r['native_external'] and predicate(r.get('selected', {})) for r in rows)
    summary['stable'] = before == {str(p.relative_to(ROOT)): digest(p) for p in tracked}
    report = dict(scope=__doc__, summary=summary, compiler_inputs=before, cases=rows)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['stable']:
        raise SystemExit('Compiler inputs changed during the sweep')


if __name__ == '__main__':
    main()
