"""Stage a parameterized specialization of the pinned CUTLASS reference."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
source=ROOT/'generic_solver/lib_v3/ref_c/cutlass_rmsnorm.c'
text=source.read_text()
text=text.replace('epsilon=1e-5f.', 'epsilon remains an FP32 parameter.')
text=text.replace('float output[]) {', 'float output[], float epsilon) {')
text=text.replace('sq_mean + 1.0e-5f', 'sq_mean + epsilon')
manifest=json.loads((ROOT/'generic_solver/lib_v3/manifest/cutlass_rmsnorm.json').read_text())
manifest['routine']='cutlass_rmsnorm_parameterized'
manifest['source']='Pinned CUTLASS rmsnorm_host with FP32 ElementType, M=1, contiguous TensorRef access; the original epsilon argument is retained.'
spec=manifest['specs'][0]
spec['name']='cutlass_rmsnorm_parameterized_f32'
spec['target']['runtime_adapter']['entry']='polygeist_rmsnorm_f32_epsilon'
spec['target']['runtime_adapter']['arguments']['Epsilon']='epsilon'
spec['args'].append('epsilon')
spec['assumptions'][-1]='epsilon > 0; finite epsilon; floating reorderings only under relaxed model'
staged=OUT/'staged_rmsnorm'
staged.mkdir(exist_ok=True)
(staged/'cutlass_rmsnorm_parameterized.c').write_text(text)
(staged/'cutlass_rmsnorm_parameterized.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged parameterized CUTLASS reference; production catalog unchanged.')
