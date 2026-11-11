from pathlib import Path
import importlib.util,hashlib,json,re,subprocess,datetime
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'scan/cumsum'
s=importlib.util.spec_from_file_location('board',Path.home()/'.config/polygeist-silicon/reproduction_20261109/remote.py');B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
ENV='LD_LIBRARY_PATH=/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu '
def pre(label):
 rc,log=B.run("python3 - <<'PY'\n"+(R/'scripts/correctness/fig3_board_preflight.py').read_text()+'\nPY',timeout=40);(D/('board-'+label+'.json')).write_text(log);assert rc==0
row=dict(program_sha256=hashlib.sha256((D/'program').read_bytes()).hexdigest(),host_uptime=subprocess.check_output(['uptime'],text=True),run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),correctness={})
def save():(D/'gpu.json').write_text(json.dumps(row,indent=2)+'\n')
pre('before');remote='/tmp/aten-cub-scan-production-20261111';assert B.upload(D/'program',remote)==0
rc,log=B.run('chmod +x '+remote,timeout=15);assert rc==0
for mode in ['normal','edge','zeros']:
 rc,log=B.run(ENV+'POLYGEIST_V2_TRACE=1 POLYGEIST_CUB_TRACE=1 timeout 60 '+remote+('' if mode=='normal' else ' '+mode),timeout=65);(D/(mode+'-correctness.log')).write_text(log)
 checks=[dict(edge=int(a),parameters=int(p),resident=int(b),count=int(c),bad=int(d),vendorbad=int(e),maxabs=float(f),maxrel=float(g)) for a,p,b,c,d,e,f,g in re.findall(r'CHECK edge=(\d+) parameters=(\d+) resident=(\d+) count=(\d+) bad=(\d+) vendorbad=(\d+) maxabs=([\deE.+-]+) maxrel=([\deE.+-]+)',log)]
 calls=log.count('POLYGEIST_CUB_CALL');typed=sum('POLYGEIST_V2_CALL' in s and 'DeviceScanInclusiveScanInit' in s for s in log.splitlines())
 row['correctness'][mode]=dict(rc=rc,checks=checks,vendor_calls=calls,typed_calls=typed,passed=rc==0 and len(checks)==2 and calls>=3 and typed>=2 and all(c['bad']==c['vendorbad']==0 for c in checks))
 print(mode,row['correctness'][mode]['passed'],flush=True);save()
if all(c['passed'] for c in row['correctness'].values()):
 rc,log=B.run(ENV+'timeout 60 '+remote+' timing',timeout=65);(D/'timing.log').write_text(log)
 samples=[dict(path=a,sample=int(b),ms=float(c)) for a,b,c in re.findall(r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)',log)]
 row['timing']=dict(rc=rc,samples=samples,passed=rc==0 and len(samples)==10 and 'PASS timing' in log,statistic='minimum of five synchronized resident samples after five warmups',baseline='direct CUB API; not native ATen')
 if row['timing']['passed']:row['timing']['minimum_ms']={p:min(s['ms'] for s in samples if s['path']==p) for p in ['raised','direct_cub']}
 print('timing',row['timing'].get('minimum_ms'),flush=True);save()
pre('after');assert all(c['passed'] for c in row['correctness'].values()) and row.get('timing',{}).get('passed')
