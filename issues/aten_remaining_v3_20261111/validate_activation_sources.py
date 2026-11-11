"""Compare retained upstream C++ lambdas with C reference specializations."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import math
import re
import struct
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
D=OUT/'activation-source-validation'
D.mkdir(exist_ok=True)
source=ROOT/'third_party/pytorch/aten/src/ATen/native/cpu/Activation.cpp'
text=source.read_text()
rows=[]
for saved in json.loads((OUT/'activation-reference-extraction.json').read_text()):
 fn=saved['upstream_function'];start=text.index('void '+fn+'(');end=text.find('\nvoid ',start+1)
 block=text[start:end if end>=0 else len(text)]
 if fn!='shrink_backward_kernel':block=block[block.rindex('AT_DISPATCH_FLOATING_TYPES('):]
 cpu=block.index('cpu_kernel_vec(')
 found=re.search(r'\[[^]]*\]\s*\(([^)]*)\)\s*(?:->\s*scalar_t\s*)?\{',block[cpu:])
 begin=cpu+found.end();depth=1;k=begin
 while depth:depth+=(block[k]=='{')-(block[k]=='}');k+=1
 body=block[begin:k-1]
 assert hashlib.sha256(body.encode()).hexdigest()==saved['body_source_sha256']
 ref=ROOT/saved['reference'];name=ref.stem.removeprefix('aten_')
 man=json.loads((ROOT/'generic_solver/lib_v3/manifest'/(ref.stem+'.json')).read_text())['specs'][0]
 arraynames=[p.strip().split()[-1] for p in found[1].split(',')]
 scalars=man['args'][len(arraynames)+2:]
 args=arraynames+scalars
 cpp='using scalar_t=float;\nextern "C" float upstream('+','.join('float '+a for a in args)+'){\n'
 cpp+=''.join('const scalar_t '+key+'('+value+');\n' for key,value in saved['constants'])+body+'\n}\n'
 path=D/(name+'.cpp');path.write_text(cpp)
 subprocess.run(['g++','-O0','-shared','-fPIC','-ffp-contract=off',str(path),'-o',str(D/(name+'-upstream.so'))],check=True)
 subprocess.run(['gcc','-O0','-shared','-fPIC','-ffp-contract=off','-DNETLIB_SPEC=reference',str(ref),'-o',str(D/(name+'-reference.so'))],check=True)
 u=C.CDLL(str(D/(name+'-upstream.so'))).upstream;u.argtypes=[C.c_float]*len(args);u.restype=C.c_float
 r=C.CDLL(str(D/(name+'-reference.so'))).reference;r.argtypes=[C.c_int]+[C.POINTER(C.c_float)]*(len(arraynames)+1)+[C.c_float]*len(scalars)
 edge=[0.,-0.,1.,-1.,3.,-3.,.25,-.25,.75,-.75,float('inf'),-float('inf'),float('nan'),1e-40,-1e-40]
 n=len(edge)**len(arraynames)
 arrays=[(C.c_float*n)(*[edge[(i//len(edge)**j)%len(edge)] for i in range(n)]) for j in range(len(arraynames))]
 parameter_sets=[[]] if not scalars else [[.25]*len(scalars),[-.75 if j==0 else .75 for j in range(len(scalars))],[float('nan')]*len(scalars)]
 bad=0;checks=0
 for params in parameter_sets:
  output=(C.c_float*n)();r(n,*arrays,output,*params)
  for i in range(n):
   want=u(*[a[i] for a in arrays],*params);got=output[i];checks+=1
   if not((math.isnan(got) and math.isnan(want)) or struct.pack('f',got)==struct.pack('f',want)):bad+=1
 rows.append(dict(name=name,values_checked=checks,failures=bad,upstream_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
 print(name,checks,'PASS' if not bad else 'FAIL',flush=True)
(OUT/'activation-source-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
assert all(r['failures']==0 for r in rows)
