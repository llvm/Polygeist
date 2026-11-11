from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent/'mfem-production';D.mkdir(exist_ok=True)
rows=json.loads((R/'issues/reproduction_20261109/mfem/fresh_validation/summary.json').read_text())
if len(sys.argv)>1:rows=[r for r in rows if r['kernel'] in sys.argv[1:]]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def build(row):
 f=D/row['kernel'];f.mkdir(exist_ok=True)
 env=dict(os.environ,POLYGEIST_BUILD_WORK_DIR=str(f/'work'),POLYGEIST_SPEC_SUITE='lib_v3',POLYGEIST_ABI_BACKEND='cuda',POLYGEIST_NUMERIC_MODEL='relaxed',POLYGEIST_MATCH_MAX_PARALLEL_DIMS='2',POLYGEIST_MATCH_MAX_OPERAND_RANK='2')
 cmd=[str(R/'scripts/correctness/polygeist_build.sh'),'--target=jetson','--function='+row['function'],'--harness='+str(R/'issues/mfem_c_kernels/benchmarks/mfem_full_output_validation.c'),row['source'],'-D'+row['macro'],'-DMFEM_BENCH_NE=1024','-o',str(f/'program')]
 started=time.monotonic()
 with (f/'build.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=log)
 result=dict(kernel=row['kernel'],rc=p.returncode,command=cmd,seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest())
 if (f/'program').exists():result['program_sha256']=hashlib.sha256((f/'program').read_bytes()).hexdigest()
 if (f/'work/match.json').exists():
  proof=json.loads((f/'work/match.json').read_text());result['specifications']=proof['emitted'];result['calls']=len(proof['emitted'])
 save(f/'build.json',result);return result
results=[]
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for r in pool.map(build,rows):
  results.append(r);save(D/'builds.json',results);print(r['kernel'],r['rc'],r.get('calls'),flush=True)
