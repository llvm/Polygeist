from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'pva-shared-recipe';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'scripts/correctness/lib_v2'),str(ROOT/'scripts/correctness')]
import recipe as R
from region_match.mlir_parse import parse_module
from region_match.symexec import execute
rows=[]
for name in ['box_filter3x3','gaussian_filter3x3','bilateral_filter3x3','morphology_dilate3x3','image_histogram_u8','histogram_equalization']:
    fn='fixture_'+name
    result=R.raise_c(R.PRODUCTION,str(ROOT/'test/Inputs/pva_matchers/image_processing.c'),fn,str(OUT/name),[],90)
    row=dict(case=name,stages={k:dict(ok=v.ok,command=v.cmd,stderr=v.stderr[-1500:]) for k,v in result.items() if k!='paths'})
    path=Path(result['paths']['debuf'])
    if path.exists():
        try:
            functions,aliases=parse_module(path.read_text());execute(next(f for f in functions if f.name==fn),aliases)
            row['symbolic_execution']='supported'
        except Exception as e:row['symbolic_execution']='rejected';row['reason']=str(e)
    else:row['symbolic_execution']='raise-failed'
    rows.append(row)
    (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,row['symbolic_execution'],row.get('reason',''),flush=True)
