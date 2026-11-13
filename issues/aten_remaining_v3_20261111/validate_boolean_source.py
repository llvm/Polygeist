from pathlib import Path
import ctypes as C,json,subprocess,struct,itertools,hashlib
O=Path(__file__).resolve().parent;D=O/'staged-boolean-family';rows=[]
edge=[0.,-0.,1.,-1.,3.,-3.,.25,-.25,.75,-.75,float('inf'),-float('inf'),float('nan'),1e-40,-1e-40,80.,-80.]
for row in json.loads((D/'extraction.json').read_text()):
 name=row['name'];ref=D/'ref_c'/('aten_'+name+'.c');cpp=D/'cpp'/(name+'.cpp');arity=len(row['arrays'])
 for command in [['g++','-std=c++17','-O0','-shared','-fPIC','-ffp-contract=off',str(cpp),'-o',str(D/(name+'-upstream.so'))],['gcc','-O0','-shared','-fPIC','-ffp-contract=off','-DNETLIB_SPEC=reference',str(ref),'-o',str(D/(name+'-reference.so'))]]:subprocess.run(command,check=True,capture_output=True)
 u=C.CDLL(str(D/(name+'-upstream.so'))).upstream;u.argtypes=[C.c_float]*arity;u.restype=C.c_float
 r=C.CDLL(str(D/(name+'-reference.so'))).reference;r.argtypes=[C.c_int]+[C.POINTER(C.c_float)]*(arity+1)
 cases=list(itertools.product(edge,repeat=arity));n=len(cases);arrays=[(C.c_float*n)(*[v[i] for v in cases]) for i in range(arity)];out=(C.c_float*n)();r(n,*arrays,out)
 bad=sum(struct.pack('f',u(*v))!=struct.pack('f',out[i]) for i,v in enumerate(cases))
 rows.append(dict(name=name,checks=n,failures=bad,reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()));print(name,n,'PASS' if not bad else 'FAIL',flush=True)
(D/'source-validation.json').write_text(json.dumps(rows,indent=2)+'\n');assert all(not r['failures'] for r in rows)
