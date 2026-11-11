"""Parallel local builds from one immutable proof/runtime snapshot."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys

OUT = Path(__file__).resolve().parent
matches = json.loads((OUT / 'pointwise-followup-matches.json').read_text())
names = [x['name'] for x in matches if len(x.get('emitted', [])) == 1]


def one(name):
    env = dict(os.environ, ATEN_FOLLOWUP_CASE=name)
    with (OUT / ('pointwise-followup-build-' + name + '.log')).open('w') as log:
        rc = subprocess.run([sys.executable, str(OUT / 'build_pointwise_followup_gpu.py')],
                            env=env, stdout=log, stderr=log).returncode
    return name, rc


rows = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(one, name) for name in names]):
        name, rc = future.result()
        path = OUT / ('pointwise-followup-build-' + name + '.json')
        if path.exists():
            rows.extend(json.loads(path.read_text()))
        else:
            rows.append(dict(name=name, rc=rc or 1))
        (OUT / 'pointwise-followup-build-results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(name, rc, flush=True)
if any(row['rc'] for row in rows):
    raise SystemExit(1)
