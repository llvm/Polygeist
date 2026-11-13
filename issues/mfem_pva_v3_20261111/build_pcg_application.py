from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent/'pcg-application';D.mkdir(exist_ok=True)
source=R/'issues/mfem_c_kernels/application_extractions/ex9p_mass_convection_2d.c';harness=source.parent/'mfem_application_reference_harness.c'
env=dict(os.environ,POLYGEIST_BUILD_WORK_DIR=str(D/'work'),POLYGEIST_SPEC_SUITE='lib_v3',POLYGEIST_ABI_BACKEND='cuda',POLYGEIST_NUMERIC_MODEL='relaxed',POLYGEIST_KEEP_DEPENDENCIES='1',POLYGEIST_MATCH_MAX_PARALLEL_DIMS='2',POLYGEIST_MATCH_MAX_OPERAND_RANK='2')
cmd=[str(R/'scripts/correctness/polygeist_build.sh'),'--target=jetson','--function=mfem_app_ex9p_mass_convection_2d','--harness='+str(harness),str(source),'-DMFEM_APP_EX9P','-DMFEM_BENCH_NE=2','-DBENCH_ITERS=1','-DMFEM_REPLAY_WARMUPS=0','-DMFEM_PUBLICATION_WARMUPS=0','-DMFEM_PUBLICATION_SAMPLES=1','-DMFEM_DEBUG_OUTPUT_SUMMARY=1','-o',str(D/'program')]
started=time.monotonic()
with (D/'build.log').open('w') as log:p=subprocess.run(cmd,env=env,stdout=log,stderr=log)
r=dict(rc=p.returncode,command=cmd,seconds=time.monotonic()-started,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),harness_sha256=hashlib.sha256(harness.read_bytes()).hexdigest())
if (D/'work/match.json').exists():r['specifications']=json.loads((D/'work/match.json').read_text())['emitted']
if (D/'program').exists():r['program_sha256']=hashlib.sha256((D/'program').read_bytes()).hexdigest()
(D/'build.json').write_text(json.dumps(r,indent=2)+'\n');print('pcg-application',r['rc'],len(r.get('specifications',[])),flush=True)
