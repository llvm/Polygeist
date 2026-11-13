from pathlib import Path
import hashlib,importlib.util,json,re,shlex,sys
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('remote',Path.home()/'.config/polygeist-silicon/reproduction_20261109/remote.py');B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
folders=[D/'mfem-production'/name for name in sys.argv[1:]] if len(sys.argv)>1 else sorted((D/'mfem-production').glob('*'))
results=[]
for f in folders:
 if not (f/'program').exists():continue
 p=f/'program';name=f.name;remote='/tmp/mfem-v3-'+name+'-20261111';assert B.upload(p,remote)==0;assert B.run('chmod +x '+shlex.quote(remote),timeout=15)[0]==0
 cmd='ulimit -s unlimited; LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu POLYGEIST_V2_TRACE=1 timeout 300 '+shlex.quote(remote)
 rc,log=B.run(cmd,timeout=310);(f/'run.log').write_text(log)
 calls=re.findall(r'POLYGEIST_V2_CALL backend=cuda call=(\S+)',log);r=dict(name=name,rc=rc,passed=rc==0 and 'correctness=PASS' in log,actual_vendor_invocations=len(calls),calls=sorted(set(calls)),program_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),command=cmd)
 (f/'result.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);(D/'mfem-production/runs.json').write_text(json.dumps(results,indent=2)+'\n');print(name,r['passed'],r['actual_vendor_invocations'],rc,flush=True)
