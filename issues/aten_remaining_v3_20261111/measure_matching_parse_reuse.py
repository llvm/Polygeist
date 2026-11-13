"""Fresh-process matching/proof timing with identical semantic inputs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'staged-adapter-cache'
sys.path[:0]=[str(R/'scripts/correctness'),str(R/'scripts/correctness/lib_v2')]
import spec_match as S,runtime_adapter as RA,mlir_lite as ML
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['original','cached'],required=True);a=ap.parse_args()
if a.mode=='original':RA._parsed_reference_module=lambda text:ML.Module(text)
else:RA._parsed_reference_module.cache_clear()
os.environ['POLYGEIST_NUMERIC_MODEL']='strict'
text=(O/'pointwise-followup/clamp_min_scalar_cpu/source_debuf.mlir').read_text()
before=S.semantic_inputs();start=time.monotonic()
output,proof=S.match(text,backend='cuda');matched=time.monotonic()
S.verify(output,proof,backend='cuda');end=time.monotonic()
assert before==S.semantic_inputs()
row=dict(mode=a.mode,matching_seconds=matched-start,verification_seconds=end-matched,
         total_seconds=end-start,emitted=proof['emitted'],input_sha256=hashlib.sha256(text.encode()).hexdigest(),
         output_sha256=hashlib.sha256(output.encode()).hexdigest(),semantic_inputs=before,
         uptime=subprocess.check_output(['uptime'],text=True).strip())
(D/('matching-'+a.mode+'.json')).write_text(json.dumps(row,indent=2)+'\n')
print(a.mode,'match',round(row['matching_seconds'],3),'verify',round(row['verification_seconds'],3),
      'total',round(row['total_seconds'],3),row['emitted'],flush=True)
