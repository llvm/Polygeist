"""Targeted validation of new reusable imports against existing fixtures."""
from pathlib import Path
import dataclasses
import json
import os
import sys
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [str(ROOT / 'scripts/correctness'), str(ROOT / 'scripts/correctness/lib_v2')]
import recipe as R
import spec_match as S

os.environ['POLYGEIST_NUMERIC_MODEL'] = 'strict'
names = ['acos','acosh','asin','asinh','atan','atanh','cosh','sinh']
if os.environ.get('ATEN_UNARY_CASES'):
    names = os.environ['ATEN_UNARY_CASES'].split(',')
result_path = OUT / 'cutensor-matches.json'
rows = json.loads(result_path.read_text()) if result_path.exists() else []
for name in names:
    directory = OUT / 'cutensor' / name
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'issues/aten_c_kernels' / ('aten_' + name + '.c')
    row = dict(name=name, source=str(source.relative_to(ROOT)))
    try:
        steps = R.raise_c(R.PRODUCTION, str(source), 'aten_' + name, str(directory / 'source'), [])
        (directory / 'raise.json').write_text(json.dumps(
            steps, default=lambda o: dataclasses.asdict(o), indent=2) + '\n')
        assert steps.get('debuf') and steps['debuf'].ok, 'production raising failed'
        text = Path(steps['paths']['debuf']).read_text()
        result, proof = S.match(text, backend='cuda')
        S.verify(result, proof, backend='cuda')
        (directory / 'matched.mlir').write_text(result)
        (directory / 'match.json').write_text(json.dumps(proof, indent=2) + '\n')
        row.update(emitted=proof['emitted'], matches=proof['matches'], elapsed_ms=proof['elapsed_ms'])
    except Exception as error:
        row['error'] = str(error)
        (directory / 'error.log').write_text(traceback.format_exc())
    rows = [r for r in rows if r["name"] != name]
    rows.append(row)
    (OUT / 'cutensor-matches.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(name, row.get('emitted', []), row.get('error', ''), flush=True)
