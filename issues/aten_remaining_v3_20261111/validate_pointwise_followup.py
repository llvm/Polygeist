from pathlib import Path
import ctypes as C,json,subprocess,math,struct,itertools,hashlib
O=Path(__file__).resolve().parent;D=O/'staged-pointwise-followup';rows=[]
edge=[0.,-0.,1.,-1.,3.,-3.,.25,-.25,.75,-.75,float('inf'),-float('inf'),float('nan'),1e-40,-1e-40,80.,-80.]
for row in json.loads((D/'extraction.json').read_text()):
 name=row['name'];ref=D/'ref_c'/('aten_'+name+'.c');cpp=D/'cpp'/(name+'.cpp')
 subprocess.run(['g++','-std=c++17','-O0','-shared','-fPIC','-ffp-contract=off',str(cpp),'-o',str(D/(name+'-upstream.so'))],check=True,capture_output=True)
 subprocess.run(['gcc','-O0','-shared','-fPIC','-ffp-contract=off','-DNETLIB_SPEC=reference',str(ref),'-lm','-o',str(D/(name+'-reference.so'))],check=True,capture_output=True)
 u=C.CDLL(str(D/(name+'-upstream.so'))).upstream;u.argtypes=[C.c_float]*(len(row['arrays'])+len(row['scalars']));u.restype=C.c_float
 r=C.CDLL(str(D/(name+'-reference.so'))).reference;r.argtypes=[C.c_int]+[C.POINTER(C.c_float)]*(len(row['arrays'])+1)+[C.c_float]*len(row['scalars'])
 n=len(edge);a=(C.c_float*n)(*edge);bad=checks=0
 for params in itertools.product(edge,repeat=len(row['scalars'])):
  out=(C.c_float*n)();r(n,a,out,*params)
  for i in range(n):
   want=u(a[i],*params);got=out[i];checks+=1
   if not((math.isnan(want) and math.isnan(got)) or struct.pack('f',want)==struct.pack('f',got)):bad+=1
 rows.append(dict(name=name,checks=checks,failures=bad,reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()));print(name,checks,'PASS' if not bad else 'FAIL',flush=True)
(D/'source-validation.json').write_text(json.dumps(rows,indent=2)+'\n');assert all(not r['failures'] for r in rows)
