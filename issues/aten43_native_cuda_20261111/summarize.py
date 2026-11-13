"""Summarize the frozen 43-case native run and retain every qualification."""
from pathlib import Path
import csv
import json
import math
import re

OUT = Path(__file__).resolve().parent
rows = json.loads((OUT / 'comparison-results.json').read_text())
assert len(rows) == 43 and len({r['name'] for r in rows}) == 43
for row in rows:
    name = row['name']
    if name == 'softshrink':
        for kind in ['normal', 'edge']:
            path = OUT / 'comparison' / name / ('linebuffered-' + kind + '.log')
            log = path.read_text()
            checks = [{k: float(v) if any(c in v for c in '.eE') else int(v)
                       for k, v in re.findall(r'(\w+)=([+\-\d.eE]+)', line)}
                      for line in log.splitlines() if line.startswith('CHECK ')]
            row['correctness'][kind]['checks'] = checks
            row['correctness'][kind]['mismatches'] = [line for line in log.splitlines() if 'MISMATCH' in line]
            row['correctness'][kind]['supplemental_linebuffered_log'] = str(path)
        row['qualification'] = ('Positive lambda: signed-zero differences (122 normal outputs) and signed-zero/NaN differences '
                                '(862 edge outputs) per residency mode. Existing negative lambda -0.75 is rejected by native API. '
                                'Raw timing only; no semantically validated speedup ratio.')
    elif name == 'hardswish_backward':
        row['qualification'] = ('Native PyTorch 24.12 uses older CUDA endpoint derivatives: mismatches at self=-3/+3 '
                                '(1 normal, 317 edge outputs per residency mode). Current reference uses upstream fix 97272e4b49. '
                                'Raw timing only; no semantically validated speedup ratio.')
    row['native_equivalence'] = 'pass' if row['normal_pass'] and row['edge_pass'] else 'mismatch'
    row['timing_comparison'] = ('raw_allocation_difference' if name == 'rms_norm' else
                                'comparable' if row['native_equivalence'] == 'pass' else 'raw_semantic_difference')

(OUT / 'comparison-results.json').write_text(json.dumps(rows, indent=2) + '\n')
fields = ['name', 'native_equivalence', 'normal_pass', 'edge_pass', 'timing_pass', 'timing_comparison',
          'raised_min_ms', 'native_min_ms', 'raised_median_ms', 'native_median_ms',
          'native_over_raised_min', 'qualification']
with (OUT / 'summary.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})

ratios = [r['native_over_raised_min'] for r in rows if r['timing_comparison'] == 'comparable']
checks = [c for r in rows for kind in r['correctness'].values() for c in kind['checks']]
local_libraries = json.loads((OUT / 'local-library-fingerprints.json').read_text())
remote_libraries = json.loads((OUT / 'native-library-fingerprints.json').read_text())
library_hashes_match = all(remote_libraries[name]['sha256'] == digest for name, digest in local_libraries.items())
assert library_hashes_match
summary = dict(cases=43, full_native_correctness_pass=sum(r['native_equivalence'] == 'pass' for r in rows),
               complete_raw_timing_cases=sum(r['timing_pass'] for r in rows), comparable_timing_cases=len(ratios),
               raised_faster_cases=sum(x > 1 for x in ratios),
               native_over_raised_min_range=[min(ratios), max(ratios)],
               native_over_raised_min_geomean=math.exp(sum(math.log(v) for v in ratios) / len(ratios)),
               full_output_checks=len(checks), compared_output_elements=sum(c['count'] for c in checks),
               mismatching_native_output_elements=sum(c['nativebad'] for c in checks),
               raised_vs_extracted_cpu_reference_failures=sum(c['bad'] for c in checks),
               same_library_fingerprints_verified=library_hashes_match,
               scope='43 original targeted fixtures; not paper Figure 5 sizes or complete 115-case comparison')
(OUT / 'aggregate-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
report = '''43-case genuine ATen CUDA comparison — 2026-11-11

41/43 cases pass every existing normal and edge output comparison with real
ATen CUDA. All 43 produced fresh same-process timing samples. The raised paths
continue to agree with their extracted CPU references. Two native/reference
semantic differences remain; they are detailed below and not counted as passes.

Performance at these existing tiny fixture sizes
-----------------------------------------------
40 cases have both full native correctness and matching preallocated-output
timing boundaries. All 40 are slower than native ATen CUDA. Native/raised minimum
latency ratios span RANGE (geometric mean GEOMEAN). Above one would mean raised
is faster. Minimum and median, along with all five raw samples, are retained.

RMSNorm passes correctness: raised RMS_RAISED us, native ATen RMS_NATIVE us. Native
rms_norm allocates its output while raised output is preallocated; that raw
latency difference is not claimed as an allocation-matched speedup. The two
semantically disagreeing cases have raw timings but no accepted speedup ratio.

These are the frozen targeted fixtures: mostly 4096 elements, tanh/sigmoid/silu
256 elements, and RMSNorm 128 elements. They are not the paper Figure 5 benchmark
shapes and do not close its 115-case comparison. No new shapes or optimization
variants were introduced. No shared compiler/matcher/runtime source was changed.

Unresolved semantic differences
------------------------------
1. hardswish_backward: one normal output and 317 edge outputs per residency mode
   differ at self=-3 or +3. Installed native PyTorch 24.12 uses self<-3 and self<=3;
   the pinned current reference uses self<=-3 and self<3. This is an upstream
   CUDA endpoint-gradient fix (97272e4b49, 2025-03-14), not a new raising failure.
   Source-semantics analysis predicts exactly the observed 1 and 317 mismatches.
   A native build aligned with the pinned corpus revision is needed for an exact
   same-semantics comparison; changing the correct reference to the older bug
   is not an appropriate fix.

2. softshrink: with valid lambda 0.25, 122 normal outputs have raised -0 versus
   native +0. The edge suite has 862 differences per mode: negative-zero and
   NaN propagation. The extracted CPU scalar body multiplies the central result
   by zero; current CUDA uses positive zero, so signed-zero behavior differs
   even in current upstream source. The installed older CUDA additionally maps
   NaN to zero; later upstream source preserves NaN. The existing lambda=-0.75
   test is rejected by the native public API. That scalar-body extension cannot
   be reported as public-API equivalence. No tolerance/NaN/zero policy was relaxed.

Environment and method
----------------------
Real NVIDIA libtorch 2.6.0a0+df5bbc09d1.nv24.12 executes on Orin. Local cross-link
and remote loaded libtorch/libc10 file hashes match. CUDA runtime 12.6, driver 13.3,
cuDNN 92501 and cuTENSOR 20700 were recorded. CPU clocks were fixed at 2.2016 GHz;
the post-run GPU frequency settings were fixed at 1.3005 GHz. Board/load/thermal/
fan/power snapshots bracket the exclusive run. Fan settings were left unchanged.

Both paths use the same C harness, resident inputs, five warmups and five
alternating synchronized CLOCK_MONOTONIC samples. These are host latencies,
not CUDA-event intervals and not Python-dispatch timings. ATen tensor wrappers
and all correctness copies are outside timing. In-place inputs are reset outside
every measurement. See METHODOLOGY.txt for API/allocation/source-version limits.

Files
-----
summary.csv: all 43 cases with correctness status, minimum/median latencies,
ratios where valid, and individual qualifications.
comparison-results.json: every raw sample and complete-output check count.
aggregate-summary.json: aggregate counts without unsupported speedup claims.
native-api-map.json: exact dispatched ATen API and header/provenance mapping.
frozen-43-results.json and frozen-artifact-index.json: original scope and hashes.
source-audit/ and disagreement-audit.json: source versions and mismatch causes.
comparison/<case>/*native.log: raw output, mismatches and timing samples.
build_native.sh, build_comparison.py, run_comparison.py: reproduction commands.

These artifacts remain local/uncommitted. The dependency sysroot and frozen
objects are local prerequisites; this directory is not a standalone submission.

Per-case results (latencies in microseconds; ratio=native_min/raised_min)
---------------------------------------------------------------------
'''
report = report.replace('RANGE', f'{min(ratios):.4f}–{max(ratios):.4f}x').replace('GEOMEAN', f'{summary["native_over_raised_min_geomean"]:.4f}x')
rms = next(r for r in rows if r['name'] == 'rms_norm')
report = report.replace('RMS_RAISED', f'{rms["raised_min_ms"]*1000:.3f}').replace('RMS_NATIVE', f'{rms["native_min_ms"]*1000:.3f}')
for r in rows:
    ratio = f'{r["native_over_raised_min"]:.4f}' if r.get('native_over_raised_min') is not None else 'qualified'
    report += f'{r["name"]:24} {r["native_equivalence"]:8} raised={r["raised_min_ms"]*1000:9.3f} native={r["native_min_ms"]*1000:8.3f} ratio={ratio}\n'
(OUT / 'REPORT.txt').write_text(report)
(OUT / 'STATUS.txt').write_text('Completed all 43 frozen cases against genuine native ATen CUDA. '
    '41 full normal/edge correctness passes; 2 explicitly diagnosed native/reference semantic differences. '
    '43 fresh raw timing sets; 40 allocation-matched, fully correct comparisons; all 40 slower at existing tiny sizes. '
    'RMSNorm raw timings allocation-qualified. GPU reservation released. No shared sources changed, commits or pushes. '
    'See REPORT.txt and summary.csv.\n')
print(json.dumps(summary, indent=2))
