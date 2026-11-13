"""Exercise real validated imports without changing the production catalog."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/correctness/lib_v2'))
import build_lib_v2 as B
import spec_provenance as P
import validate_specs as V

names = ['aten_abs_f32', 'aten_addcmul_f32', 'openblas_dgemm_nn',
         'opencv_morphology_dilate_u8']
original = {s.name: s for s in B.load_specs()}
work = OUT / 'isolated'
work.mkdir(exist_ok=True)
specs = []
for name in names:
    spec = original[name]
    fragment = work / (name + '.json')
    reference = work / (name + '.c')
    shutil.copyfile(spec.fragment, fragment)
    shutil.copyfile(spec.ref_c, reference)
    specs.append(replace(spec, fragment=fragment, ref_c=reference))
B.LIB, B.RAISED = work, work / 'raised'
B.LIBRARY_OUT, B.STATUS_MD = work / 'library.mlir', work / 'status.md'
B.load_specs = lambda: specs
V.RAISED, V.WORK = B.RAISED, work / 'validation'

def run(label, incremental=True):
    sys.argv = ['build_lib_v2.py', '-j', '4'] + (['--incremental'] if incremental else [])
    output = io.StringIO()
    start = time.monotonic()
    with redirect_stdout(output):
        rc = B.main()
    elapsed = time.monotonic() - start
    (OUT / (label + '.log')).write_text(output.getvalue())
    assert rc == 0
    manifest = json.loads((work / 'import_manifest.json').read_text())
    errors, common = P.validate(work / 'import_manifest.json')
    assert not errors and not common, (errors, common)
    rows = manifest['specifications']
    assert all(r['status'] == 'RAISED' and all(
        r['numerical_validation'][stage] in ('EXACT', 'WITHIN_1e-12')
        for stage in ('affine', 'linalg', 'defn')) for r in rows.values())
    return dict(label=label, seconds=elapsed,
                reused=[n for n, r in rows.items() if r.get('validation_reuse')],
                library_sha256=hashlib.sha256(B.LIBRARY_OUT.read_bytes()).hexdigest())

(OUT / 'host-load.txt').write_text(subprocess.check_output(['uptime'], text=True))
results = [run('cold'), run('warm')]
assert results[0]['reused'] == []
assert set(results[1]['reused']) == set(names)
assert results[0]['library_sha256'] == results[1]['library_sha256']
reference = specs[1].ref_c
reference.write_text(reference.read_text() + '\n/* Input-change invalidation probe. */\n')
results.append(run('one-source-changed'))
assert set(results[-1]['reused']) == set(names) - {names[1]}
artifact = B.RAISED / (names[0] + '_debuf.mlir')
artifact.write_text(artifact.read_text() + '\n// Artifact-change invalidation probe.\n')
results.append(run('one-artifact-changed'))
assert set(results[-1]['reused']) == set(names) - {names[0]}
results.append(run('forced-full', incremental=False))
assert results[-1]['reused'] == []
assert results[-1]['library_sha256'] == results[-2]['library_sha256']
(OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
for row in results:
    print(row['label'], f"{row['seconds']:.3f}s", f"reused={len(row['reused'])}/4")
