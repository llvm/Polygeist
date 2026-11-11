"""Retain pinned ATen FP32-output comparison/logical lambdas as reusable refs."""
from pathlib import Path
import re,json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'staged-boolean-family'
for sub in ['ref_c','manifest','cpp']:(D/sub).mkdir(parents=True,exist_ok=True)
rows=[]
for name in ['eq','ne','lt','le','gt','ge','logical_and','logical_or','logical_xor','logical_not']:
 source=R/'third_party/pytorch/aten/src/ATen/native/cpu'/('UnaryOpsKernel.cpp' if name=='logical_not' else 'BinaryOpsKernel.cpp')
 text=source.read_text();start=text.index('void '+name+'_kernel(');end=text.find('\nvoid ',start+1);block=text[start:end if end>=0 else len(text)]
 matches=list(re.finditer(r'\[\]\s*\(([^)]*)\)\s*->\s*scalar_t\s*\{([^}]+)\}',block));assert matches,name
 m=matches[0];params=[x.strip().split()[-1] for x in m[1].split(',')];body=m[2]
 cpp='using scalar_t=float; using self_t=float;\nextern "C" float upstream('+','.join('float '+v for v in params)+'){'+body+'}\n'
 (D/'cpp'/(name+'.cpp')).write_text(cpp)
 adapted=body.replace('static_cast<scalar_t>','(float)');adapted=re.sub(r'\bbool\((\w+)\)',r'((_Bool)(\1))',adapted)
 for p in params:adapted=re.sub(r'\b'+p+r'\b',p+'[i]',adapted)
 adapted=adapted.replace('return','output[i] =')
 ref='/* FP32-output scalar specialization of pinned '+name+'_kernel.\n * Preserve comparison/cast semantics; TensorIterator becomes a contiguous loop. */\n#ifndef NETLIB_SPEC\n#error "NETLIB_SPEC required"\n#endif\nvoid NETLIB_SPEC('+','.join(['int n']+['float '+p+'[]' for p in params]+['float output[]'])+'){\nfor(int i=0;i<n;++i){'+adapted+'}\n}\n'
 stem='aten_'+name;(D/'ref_c'/(stem+'.c')).write_text(ref)
 man=dict(routine=stem,source='Pinned ATen floating-input/floating-output scalar implementation',upstream_files=[dict(path=str(source.relative_to(R)),sha256=hashlib.sha256(source.read_bytes()).hexdigest())],adaptation=name+'_kernel scalar lambda specialized to FP32 inputs and FP32 0/1 output. TensorIterator maps to a contiguous C loop. C++ explicit casts map to exact C conversions; the C++ lambda is retained for independent checks. cuDNN graph opcodes are mechanically selected from the imported reference, with vendor semantics validated separately; this is not cuDNN implementation source.',specs=[dict(name=stem+'_f32',target=dict(library='cuDNN',call='cudnnBackendExecute',backends=['cuda'],runtime_adapter=dict(schema=1,kind='pointwise_reference',extent_arg='n')),args=['n']+params+['output'],assumptions=['positive n <= INT_MAX; contiguous FP32 arrays','output disjoint from inputs; FP32 output represents boolean 0/1','floating exceptions and errno not observed'],deviations=[])])
 (D/'manifest'/(stem+'.json')).write_text(json.dumps(man,indent=2)+'\n')
 rows.append(dict(name=name,arrays=params,body_sha256=hashlib.sha256(body.encode()).hexdigest(),reference_sha256=hashlib.sha256(ref.encode()).hexdigest(),source=str(source.relative_to(R))))
(D/'extraction.json').write_text(json.dumps(rows,indent=2)+'\n');print('staged',len(rows),'references')
