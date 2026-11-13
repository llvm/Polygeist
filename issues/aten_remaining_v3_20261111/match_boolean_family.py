from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,os,json,dataclasses,multiprocessing
O=Path(__file__).resolve().parent;R=O.parents[1]
def one(name):
 sys.path[:0]=[str(R/'scripts/correctness'),str(R/'scripts/correctness/lib_v2')]
 import recipe as P,spec_match as S
 os.environ['POLYGEIST_NUMERIC_MODEL']='strict';D=O/'boolean-family'/name;D.mkdir(parents=True,exist_ok=True);src=R/'issues/aten_c_kernels'/('aten_'+name+'.c')
 stages=P.raise_c(P.PRODUCTION,str(src),'aten_'+name,str(D/'source'),[])
 (D/'raise.json').write_text(json.dumps(stages,default=dataclasses.asdict,indent=2)+'\n');assert stages['debuf'].ok
 y,p=S.match((D/'source_debuf.mlir').read_text(),backend='cuda');S.verify(y,p,backend='cuda')
 (D/'matched.mlir').write_text(y);(D/'match.json').write_text(json.dumps(p,indent=2)+'\n')
 return dict(name=name,source=str(src.relative_to(R)),emitted=p['emitted'],matches=p['matches'],elapsed_ms=p['elapsed_ms'])
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(one,n) for n in ['eq','ne','lt','le','gt','ge','logical_and','logical_or','logical_xor','logical_not_f32']]):
   row=f.result();rows.append(row);(O/'boolean-family-matches.json').write_text(json.dumps(rows,indent=2)+'\n');print(row['name'],row['emitted'],flush=True)
