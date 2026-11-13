from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'staged-cub';D=OUT/'application-validation';D.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import validate_specs as V
V.RAISED=D;V.WORK=D/'validate';ref=D/'reference.c';ref.write_text('#define aten_cumsum NETLIB_SPEC\n#include "'+str(ROOT/'issues/aten_c_kernels/aten_cumsum.c')+'"\n')
for suffix in ['.mlir','_linalg.mlir','_debuf.mlir']:shutil.copy2(OUT/('application-fixed'+suffix),D/('aten_cumsum'+suffix))
rows=[V.validate('aten_cumsum',{'args':['input','output']},ref,set(),level) for level in ['affine','linalg','defn']]
(D/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print([r.get('verdict') for r in rows]);assert all(r['verdict']=='EXACT' for r in rows)
