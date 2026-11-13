from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'staged-cub-reductions'
sys.path[:0]=[str(R/'scripts/correctness'),str(R/'scripts/correctness/lib_v2')]
import build_lib_v2 as B,validate_specs as V
B.LIB=D;B.REF_C=D/'ref_c';B.MANIFEST=D/'manifest';B.RAISED=D/'raised'
V.LIB=D;V.RAISED=B.RAISED;V.WORK=B.RAISED/'validate'
specs=B.load_specs()
with ThreadPoolExecutor(max_workers=3) as p:rows=list(p.map(lambda s:B.build_one(s,B.R.PRODUCTION,300,True),specs))
B.validate_results(rows,{s.name:s for s in specs},3)
(D/'import-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['spec'],row['status'],row.get('reason',''),row.get('validation'),flush=True)
