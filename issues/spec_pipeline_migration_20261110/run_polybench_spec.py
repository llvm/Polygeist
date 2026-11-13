#!/usr/bin/env python3
"""Fresh whole-program MINI regression, using actual external BLAS calls."""
import json, math, os, re, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'polybench'
PB=ROOT/'tools/cgeist/Test/polybench'
CASES='2mm 3mm atax bicg cholesky correlation covariance doitgen gemm gemver gesummv mvt symm syr2k syrk trisolv trmm'.split()
OUT.mkdir(exist_ok=True)
CASES=sys.argv[1:] or CASES
results=json.loads((OUT/'results.json').read_text()) if sys.argv[1:] and (OUT/'results.json').exists() else []
for name in CASES:
    work=OUT/name;work.mkdir(exist_ok=True)
    source=next(PB.rglob(name+'.c'))
    text=source.read_text().replace('#include "'+name+'.h"','#include "'+name+'.h"\n#undef DATA_PRINTF_MODIFIER\n#define DATA_PRINTF_MODIFIER "%.17g "')
    cfile=work/(name+'.c');cfile.write_text(text)
    flags=['-DMINI_DATASET','-DDATA_TYPE_IS_DOUBLE','-DPOLYBENCH_DUMP_ARRAYS','-I'+str(PB/'utilities'),'-I'+str(source.parent)]
    env=dict(os.environ, POLYGEIST_NUMERIC_MODEL='relaxed',POLYGEIST_CPU_BLAS='1',POLYGEIST_BUILD_WORK_DIR=str(work/'build'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',POLYGEIST_V2_TRACE='1')
    row=dict(case=name,dataset='MINI',target='host',numeric_model='relaxed',uptime=subprocess.check_output(['uptime'],text=True).strip())
    start=time.monotonic()
    try:
        cmd=[str(ROOT/'scripts/correctness/polygeist_build.sh'),'--target=host','--whole-program','--function=kernel_'+name,'-o',str(work/'raised.exe'),str(cfile),*flags]
        (work/'command.json').write_text(json.dumps(cmd,indent=2))
        with (work/'build.log').open('w') as log:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        row['build_seconds']=round(time.monotonic()-start,3)
        if r.returncode:raise RuntimeError('build failed: '+str(r.returncode))
        p=json.loads((work/'build/match.json').read_text());row['specifications']=p['emitted']
        cmd=[str(ROOT/'llvm-project/build/bin/clang'),'-O3',*flags,str(cfile),str(PB/'utilities/polybench.c'),'-lm','-o',str(work/'reference.exe')]
        with (work/'reference-build.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        texts={}
        for mode in ['reference','raised']:
            t=time.monotonic()
            r=subprocess.run([str(work/(mode+'.exe'))],env=env,capture_output=True,text=True,timeout=60)
            row[mode+'_process_seconds']=time.monotonic()-t
            (work/(mode+'.stdout')).write_text(r.stdout);(work/(mode+'.stderr')).write_text(r.stderr)
            if r.returncode:raise RuntimeError(mode+' execution failed: '+str(r.returncode))
            texts[mode]=r.stderr
        arrays={}
        for mode,text in texts.items():
            arrays[mode]={k:[float(x) for x in v.split()] for k,v in re.findall(r'begin dump: (\w+)\s*(.*?)\s*end\s+dump: \1',text,re.S)}
        a,b=arrays['raised'],arrays['reference']
        if not a or a.keys()!=b.keys():raise RuntimeError('missing or different output arrays')
        row['elements']=sum(len(v) for v in b.values());row['max_absolute_error']=0.;row['max_normalized_error']=0.;correct=True
        for key in b:
            if len(a[key])!=len(b[key]):raise RuntimeError('array lengths differ')
            scale=max(max(abs(y) for y in b[key]),1e-30)
            for x,y in zip(a[key],b[key]):
                if math.isnan(x) or math.isnan(y):correct &= math.isnan(x) and math.isnan(y);continue
                if math.isinf(x) or math.isinf(y):correct &= x==y;continue
                error=abs(x-y);row['max_absolute_error']=max(row['max_absolute_error'],error)
                row['max_normalized_error']=max(row['max_normalized_error'],error/scale)
                correct &= error <= 1e-12+1e-12*scale
        row['vendor_calls']=re.findall(r'POLYGEIST_V2_STATS[^\n]*',texts['raised'])
        row['correctness']='PASS' if correct else 'FAIL'
        row['replacement_executed']=bool(row['vendor_calls'])
        if row['specifications'] and not row['vendor_calls']:row['correctness']='FAIL_NO_VENDOR_EXECUTION'
        row['status']='PASS' if row['correctness']=='PASS' else 'FAIL'
    except Exception as e:
        row['status']='FAIL';row['reason']=str(e)
    results=[r for r in results if r["case"] != name]
    results.append(row)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,row['status'],','.join(row.get('specifications',[])),row.get('reason',''),flush=True)
