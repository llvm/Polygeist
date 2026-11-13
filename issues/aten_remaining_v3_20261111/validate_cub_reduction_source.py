from pathlib import Path
import hashlib
import json
import subprocess

D=Path(__file__).resolve().parent/'staged-cub-reductions'
specs=json.loads((D/'manifest/cub_reduce.json').read_text())['specs']
commands=[]
objects=[]
with (D/'source-validation.log').open('w') as log:
    for spec in specs:
        obj=D/(spec['name']+'.o');objects.append(str(obj))
        cmd=['gcc','-std=c11','-O0','-ffp-contract=off','-DNETLIB_SPEC='+spec['name']]
        cmd+=['-D'+k+'='+str(v) for k,v in spec['defines'].items()]
        cmd+=['-c',str(D/'ref_c/cub_reduce.c'),'-o',str(obj)]
        commands.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=log)
    cmd=['g++','-std=c++17','-O0','-ffp-contract=off',str(D/'source_validation.cpp')]+objects+['-o',str(D/'source_validation')]
    commands.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=log)
    result=subprocess.run([str(D/'source_validation')],text=True,capture_output=True)
    log.write(result.stdout+result.stderr)
result.check_returncode()
(D/'source-validation.json').write_text(json.dumps(dict(passed=True,commands=commands,
    output=result.stdout,hashes={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [D/'ref_c/cub_reduce.c',D/'source_validation.cpp',D/'upstream-specialization.hpp']}),indent=2)+'\n')
print(result.stdout,end='')
