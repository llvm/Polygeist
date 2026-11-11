"""Compare fresh-process incremental and full imports at one frozen revision."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIB = ROOT / 'generic_solver/lib_v3'
rows = []
for label, flags in [('catalog-cold', ['--incremental']),
                     ('catalog-warm', ['--incremental']), ('catalog-full', [])]:
    command = [sys.executable, str(ROOT / 'scripts/correctness/lib_v2/build_lib_v2.py'),
               '-j', '8', *flags]
    load = subprocess.check_output(['uptime'], text=True).strip()
    start = time.monotonic()
    with (OUT / (label + '.log')).open('w') as log:
        run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    assert run.returncode == 0, (label, run.returncode)
    manifest = json.loads((LIB / 'import_manifest.json').read_text())
    statuses = {name: (row['status'], row['numerical_validation'])
                for name, row in manifest['specifications'].items()}
    library_hash = hashlib.sha256((ROOT / 'generic_solver/kernel_library_v3.mlir').read_bytes()).hexdigest()
    row = dict(label=label, command=command, seconds=elapsed, host_load=load,
               inputs=len(statuses), reused=sum(bool(row.get('validation_reuse'))
                   for row in manifest['specifications'].values()), library_sha256=library_hash)
    if rows:
        assert library_hash == rows[0]['library_sha256'], 'catalog body changed'
        assert statuses == prior_statuses, 'status or numerical validation changed'
    prior_statuses = statuses
    rows.append(row)
    (OUT / 'catalog-results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(label, f'{elapsed:.3f}s', f"reuse={row['reused']}/{row['inputs']}", flush=True)
