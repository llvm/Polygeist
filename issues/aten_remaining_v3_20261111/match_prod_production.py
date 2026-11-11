from pathlib import Path
import sys,os,json
O=Path(__file__).resolve().parent; R=O.parents[1]; D=O/'reductions'/'prod'; D.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'scripts/correctness')); import spec_match as S
os.environ['POLYGEIST_NUMERIC_MODEL']='strict'
x=(O/'staged-cub-reductions/matches/prod/source_debuf.mlir').read_text(); (D/'source_debuf.mlir').write_text(x)
y,p=S.match(x,backend='cuda',only={'cub_reduce_product_f32'})
(D/'matched.mlir').write_text(y);(D/'match.json').write_text(json.dumps(p,indent=2)+'\n')
S.verify(y,p,backend='cuda');print('emitted',p['emitted'],flush=True)
