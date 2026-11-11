from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'staged-cub'
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import build_lib_v2 as B
import validate_specs as V
B.LIB=OUT;B.REF_C=OUT/'ref_c';B.MANIFEST=OUT/'manifest';B.RAISED=OUT/'raised'
V.LIB=OUT;V.RAISED=B.RAISED;V.WORK=B.RAISED/'validate'
spec=B.load_specs()[0];result=B.build_one(spec,B.R.PRODUCTION,300,True)
B.validate_results([result],{spec.name:spec},1)
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(spec.name,result['status'],result.get('reason',''),result.get('validation'))
