"""Compare retained production evidence with the historical fixture census."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

D = Path(__file__).resolve().parent
R = D.parents[1]
old = (R / 'issues/mfem_c_kernels/FULL_OUTPUT_VALIDATION_2026-09-08.md').read_text()
historical = {name: int(count) for name, count in re.findall(r'^(\w+)\s+sites=(\d+)', old, re.M)}
inputs = json.loads((R / 'issues/reproduction_20261109/mfem/fresh_validation/summary.json').read_text())
rows = []
for item in inputs:
    name = item['kernel']
    folder = D / 'mfem-production' / name
    row = dict(name=name, historical_sites=historical.get(name.replace('_apply_', '_')))
    for key, filename in [('build', 'build.json'), ('execution', 'result.json')]:
        path = folder / filename
        if path.exists():
            row[key] = json.loads(path.read_text())
    if (folder / 'work/match.json').exists():
        proof = json.loads((folder / 'work/match.json').read_text())
        row['recognized_sites'] = len(proof['emitted'])
        row['specifications'] = proof['emitted']
        row['proof_sha256'] = hashlib.sha256((folder / 'work/match.json').read_bytes()).hexdigest()
    if (folder / 'work/abi.json').exists():
        abi = json.loads((folder / 'work/abi.json').read_text())
        row['lowered_sites'] = len(abi.get('launches', []))
    if (folder / 'run.log').exists():
        log = (folder / 'run.log').read_text()
        row['correctness_lines'] = [line for line in log.splitlines() if 'correctness=' in line]
    executable = folder / 'program'
    current_hash = hashlib.sha256(executable.read_bytes()).hexdigest() if executable.exists() else None
    row['binary_hash_agrees'] = bool(current_hash and
        current_hash == row.get('build', {}).get('program_sha256') ==
        row.get('execution', {}).get('program_sha256'))
    row['fully_validated'] = (row['binary_hash_agrees'] and row.get('build', {}).get('rc') == 0 and
                              row.get('execution', {}).get('passed') is True and
                              row.get('execution', {}).get('actual_vendor_invocations', 0) > 0)
    rows.append(row)
summary = dict(fixtures=len(rows), recognized=sum(bool(x.get('recognized_sites')) for x in rows),
               built=sum(x.get('build', {}).get('rc') == 0 for x in rows),
               full_output_validated=sum(x['fully_validated'] for x in rows),
               historical_sites=sum(x['historical_sites'] or 0 for x in rows),
               recognized_sites=sum(x.get('recognized_sites', 0) for x in rows),
               lowered_sites=sum(x.get('lowered_sites', 0) for x in rows),
               validated_sites=sum(x.get('lowered_sites', 0) for x in rows if x['fully_validated']),
               note='Static call counts may change with the library and generic rank slicing; '
                    'compare full fixture coverage and outputs before equating site counts.', rows=rows)
counts = Counter(spec for row in rows for spec in row.get('specifications', []))
correctness = [line for row in rows if row['fully_validated'] for line in row.get('correctness_lines', [])]
summary.update(
    full_output_values=sum(int(re.search(r'elements=(\d+)', line)[1]) for line in correctness),
    maximum_absolute_error=max((float(re.search(r'max_abs=(\S+)', line)[1]) for line in correctness), default=0.),
    specifications=dict(counts),
    actual_vendor_invocations=sum(row.get('execution', {}).get('actual_vendor_invocations', 0) for row in rows if row['fully_validated']))
(D / 'mfem-production-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print({k: v for k, v in summary.items() if k != 'rows'})
