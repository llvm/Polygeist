#!/usr/bin/env python3
"""Render the retained audit as a compact, explicitly qualified result report."""
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE/'audit.json').read_text())
summary = []
lines = [
    'LoopLibrarian Figure 4 GPU reproduction — 2026-11-10',
    '',
    'Scope: the 12 FP64 LARGE PolyBench GPU cases in the pinned final paper.',
    'Fresh C kernels compiled by the current compiler; explicit ABI harnesses',
    'reuse the original initialization and output routines. This is isolated',
    'kernel reproduction, not an untouched whole-program compilation claim.',
    '',
    'Timings below are CUDA-event compute device milliseconds, matching the',
    'published CSV. Synchronized compute host wall times are recorded separately.',
    'Every reported minimum is from the final five measured iterations in one',
    'process after five warmups. Earlier attempts are retained and not pooled.',
    '',
    'kernel       paper raised   current raised   paper speedup   current speedup   outputs',
]
for row in rows:
    name = row['kernel']
    base = row['paper_comparison_baseline']
    r = row['variants']['raised']
    b = row['variants'][base]
    ok = r.get('accepted', False) and b.get('accepted', False)
    current = r.get('run', {}).get('compute_device_ms', {}).get('min_ms')
    target = r.get('paper_ms')
    speedup = row.get('device_speedup')
    old_speedup = row.get('paper_speedup')
    if None in (current, target, speedup, old_speedup):
        lines.append(f'{name:12} INCOMPLETE — see audit.json')
        continue
    lines.append(f'{name:12} {target:12.6f}   {current:14.6f}   {old_speedup:12.6f}x  {speedup:14.6f}x   {"PASS" if ok else "FAIL"}')
    summary.append(dict(kernel=name, correctness_passed=ok, baseline=base,
        paper_raised_ms=target, raised_device_ms=current,
        raised_change_percent=r['time_change_percent'],
        baseline_device_ms=b['run']['compute_device_ms']['min_ms'],
        paper_device_speedup=old_speedup, device_speedup=speedup,
        raised_compute_wall_ms=r['run'].get('compute_wall_ms',{}).get('min_ms'),
        baseline_compute_wall_ms=b['run'].get('compute_wall_ms',{}).get('min_ms'),
        raised_end_to_end_ms=r['run'].get('e2e_host_ms',{}).get('min_ms')))
lines += ['', 'Numerical checks:',
    'Complete canonical output dumps, printed with 17 significant digits.',
    'For every element: abs(actual-reference) <= 1e-8 + 1e-8*abs(reference).',
    'Missing arrays, wrong counts, NaNs, infinities, failed runs, missing samples,',
    'and stale executable hashes are rejected. The reference is original C',
    'cross-compiled for the same Orin. Array counts and errors are in audit.json.',
    '', 'Interpretation:']
if len(summary) == 12 and all(r['correctness_passed'] for r in summary):
    close = [r['kernel'] for r in summary if abs(r['raised_change_percent']) <= 5]
    faster = [r['kernel'] for r in summary if r['raised_change_percent'] < -5]
    slower = [r['kernel'] for r in summary if r['raised_change_percent'] > 5]
    lines += [f'All 12 raised implementations and their paper comparison baselines pass.',
        f'Within 5% of the paper raised time: {len(close)} ({", ".join(close)}).',
        f'More than 5% faster: {len(faster)} ({", ".join(faster) or "none"}).',
        f'More than 5% slower: {len(slower)} ({", ".join(slower) or "none"}).',
        'The 5% bands are descriptive reporting bands, not a paper-defined acceptance rule.',
        f'Median device speedup: {statistics.median(r["device_speedup"] for r in summary):.6f}x;',
        f'published CSV median: {statistics.median(r["paper_device_speedup"] for r in summary):.6f}x.',
        '3mm is essentially tied with its baseline; do not call it a speedup.',
        'GEMVER now uses transposed GEMV directly, eliminating the explicit matrix',
        'transpose present in the historical GPU IR. GESUMMV avoids redundant',
        'temporary copies through buffer-aware vector BLAS lowering. Faster current',
        'results are reported as improvements, not exact reproductions of old times.']
lines += ['', 'Baseline provenance:',
    'For correlation and covariance, the paper used locally authored CUDA',
    'adaptations. Those exact computational bodies are the paper comparison',
    'baselines above; only timing instrumentation was added. They must not be',
    'described as verbatim upstream PolyBenchGPU kernels.',
    'The separate upstream audit imports computational CUDA bodies verbatim from',
    'PolyBenchGPU commit 5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c.',
    'All cases use canonical FP64 data, dimensions, and initialization.',
    'Correlation configures FLOAT_N and EPS to match canonical C and retains',
    'the upstream final-diagonal host copy. Upstream covariance lacks the',
    'canonical division by (N-1), so its audit includes a timed cuBLAS Dscal.',
    'That covariance audit is upstream CUDA plus vendor normalization, not pure',
    'upstream PolyBenchGPU. It is not used to reproduce the published speedup.']
for row in rows:
    if row['kernel'] in ('correlation', 'covariance'):
        native = row['variants']['native']
        lines.append(f'  {row["kernel"]}: upstream audit {native["run"]["compute_device_ms"]["min_ms"]:.6f} ms; correctness {native.get("accepted")}.')
lines += ['', 'Changes in this task:',
    '- Retry valid library candidates after rejecting an unconnected composition.',
    '- Preserve destination metadata and avoid duplicate attributes in bufferization.',
    '- Support bufferized GEMV, zero-fill, and AXPBY; preserve offsets and strides.',
    '- Enable this path for supported vector BLAS chains containing AXPBY.',
    '- Remove only adjacent copy-to-temporary/copy-back pairs with no other uses.',
    '- Stage rejected host-to-device copies through pinned storage, preserving',
    '  stream ordering and lifetime; this fixes repeated BiCG/GEMVER failures.',
    '- Add source-based builds, full-output audits, provenance, and rerun scripts.',
    '', 'Validation and remaining artifact qualifications:',
    '19 distinct compiler/matcher checks passed; see passed-tests.txt. Two CUDA',
    'smoke programs also passed on Orin. The pre-fix matcher fails the new retry',
    'regression, and the initial BiCG/GEMVER runtime failures are retained.',
    'An existing cublas-daxpby-smoke.c has no lit RUN line; it was compiled and',
    'run directly on Orin successfully instead.',
    'CPU clocks are fixed at 2.2016 GHz, GPU at 1.3005 GHz, power mode MAXN,',
    'CPU affinity 0, one CPU/BLAS/OpenMP thread, serialized board measurements.',
    'The fan remains automatic: the configured account cannot set it. Therefore',
    'these rows do not fully satisfy the repository fixed-fan publication protocol.',
    'Preflight, temperatures, raw samples, statistics, binaries, hashes, source',
    'revisions, and library hashes are retained. No commits or pushes were made.',
    'Pre-existing unrelated runtime changes were left byte-for-byte unchanged.',
    '', 'Reproduction:',
    'From the repository root, export POLYGEIST_ROOT and source envsetup.sh.',
    'Run: python3 issues/fig4_gpu_20261110/reproduce.py --remote-helper PRIVATE_ADAPTER',
    'The adapter contains private access configuration and is not part of the artifact.',
    'It must provide run(command, timeout) and upload(local, destination).',
    'Then run: python3 issues/fig4_gpu_20261110/report.py',
    'Never run board measurements concurrently. Existing runs are archived first.',
    'Data: audit.json, results.csv, run-order.json, provenance.json, board-provenance.json.',
]
(HERE/'report.txt').write_text('\n'.join(lines)+'\n')
if summary:
    with (HERE/'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
print('Wrote report.txt and results.csv')
