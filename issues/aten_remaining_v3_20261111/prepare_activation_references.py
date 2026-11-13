"""Extract reusable scalar references from pinned ATen activation implementations."""
from pathlib import Path
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
source=ROOT/'third_party/pytorch/aten/src/ATen/native/cpu/Activation.cpp'
text=source.read_text()
sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha=='66cca1278103413ab9c566585004f58213a5f870a4daa07c0bc75bd2f30b19b1'
CASES={
 'shrink_backward':('shrink_backward_kernel',['grad_val','self_val'],['lambd_val'],False),
 'softshrink':('softshrink_kernel',['a'],['lambd_val'],True),
 'hardtanh_backward':('hardtanh_backward_kernel',['grad_val','self_val'],['min_val','max_val'],True),
 'hardsigmoid_backward':('hardsigmoid_backward_kernel',['grad_val','self_val'],[],True),
 'hardswish_backward':('hardswish_backward_kernel',['grad_val','self_val'],[],True),
}
rows=[]
for name,(function,arrays,scalars,lastbranch) in CASES.items():
 start=text.index('void '+function+'(')
 end=text.find('\nvoid ',start+1)
 block=text[start:end if end>=0 else len(text)]
 if lastbranch:
  marker='AT_DISPATCH_FLOATING_TYPES('
  block=block[block.rindex(marker):]
 cpu=block.index('cpu_kernel_vec(')
 found=re.search(r'\[[^]]*\]\s*\(([^)]*)\)\s*(?:->\s*scalar_t\s*)?\{',block[cpu:])
 assert found,function
 signature=found[1]
 body_start=cpu+found.end();depth=1;cursor=body_start
 while depth:
  depth+=(block[cursor]=='{')-(block[cursor]=='}');cursor+=1
 body=block[body_start:cursor-1].strip()
 actual=[p.strip().split()[-1] for p in signature.split(',')]
 assert actual==arrays,(name,actual)
 constants=[]
 for constant,value in re.findall(r'const scalar_t (\w+)\(([^;]+)\);',block[:cpu]):
  if re.search(r'\b'+constant+r'\b',body):constants.append((constant,value))
 body=re.sub(r'\bscalar_t\(0\)','0.0f',body)
 for arg in arrays:body=re.sub(r'\b'+arg+r'\b',arg+'[i]',body)
 body=re.sub(r'\breturn\b','output[i] =',body)
 params=['int n']+['float '+a+'[]' for a in arrays]+['float output[]']+['float '+a for a in scalars]
 reference=('/* Scalar FP32 contiguous specialization of pinned ATen '+function+'.\n'
            ' * TensorIterator mapping becomes a C loop; scalar operands remain parameters.\n'
            ' * Scalar body and operation order are retained. See manifest provenance. */\n'
            '#ifndef NETLIB_SPEC\n#error "NETLIB_SPEC must name the imported function"\n#endif\n'
            'void NETLIB_SPEC('+', '.join(params)+') {\n')
 reference+=''.join('  const float '+k+' = '+v+';\n' for k,v in constants)
 reference+='  for (int i=0;i<n;++i) {\n'+''.join('    '+ln.strip()+'\n' for ln in body.splitlines())+'  }\n}\n'
 stem='aten_'+name
 ref=ROOT/'generic_solver/lib_v3/ref_c'/ (stem+'.c')
 ref.write_text(reference)
 manifest=dict(routine=stem,source='Pinned ATen scalar implementation; reusable parameterized contiguous FP32 reference',
  upstream_files=[dict(path=str(source.relative_to(ROOT)),sha256=sha)],
  adaptation='ATen '+function+' float scalar lambda. TensorIterator iteration becomes a C loop; scalar_t is float; scalar_t(0) is 0.0f; scalar parameters and operation order retained. Library API implementation is proprietary: the backend graph is mechanically emitted from this reference and its opcode semantics are separately trusted and tested.',
  specs=[dict(name=stem+'_f32',target=dict(library='cuDNN',call='cudnnBackendExecute',backends=['cuda'],runtime_adapter=dict(schema=1,kind='pointwise_reference',extent_arg='n')),
    args=['n']+arrays+['output']+scalars,assumptions=['positive n <= INT_MAX','contiguous FP32 arrays; output disjoint from inputs','errno and floating exception flags are not observed'],deviations=[])])
 (ROOT/'generic_solver/lib_v3/manifest'/(stem+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 rows.append(dict(reference=str(ref.relative_to(ROOT)),upstream_function=function,scalar_lambda=signature,constants=constants,body_source_sha256=hashlib.sha256(block[body_start:cursor-1].encode()).hexdigest()))
(OUT/'activation-reference-extraction.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Prepared',len(rows),'reusable activation references from pinned upstream scalar bodies')
