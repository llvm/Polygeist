from pathlib import Path
import hashlib
import importlib.util
import json
import re

D = Path(__file__).resolve().parent / 'pcg-application'
s = importlib.util.spec_from_file_location('remote', Path.home() / '.config/polygeist-silicon/reproduction_20261109/remote.py')
B = importlib.util.module_from_spec(s)
s.loader.exec_module(B)
p = D / 'program'
remote = '/tmp/mfem-v3-pcg-application-20261111'
assert B.upload(p, remote) == 0
assert B.run('chmod +x ' + remote, timeout=15)[0] == 0
cmd = ('LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:'
       '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu POLYGEIST_V2_TRACE=1 timeout 300 ' + remote)
rc, log = B.run(cmd, timeout=310)
(D / 'run.log').write_text(log)
calls = re.findall(r'POLYGEIST_V2_CALL backend=cuda call=(\S+)', log)
result = dict(rc=rc, program_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
              command=cmd, correctness_lines=[x for x in log.splitlines() if 'correctness=' in x],
              vendor_calls=sorted(set(calls)), actual_vendor_invocations=len(calls),
              all_outputs_pass=rc == 0 and 'correctness=PASS' in log,
              daxpy_observed=any('cublasDaxpy' in x for x in calls),
              daxpby_observed=any('cublasDgeam' in x for x in calls))
(D / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(result)
