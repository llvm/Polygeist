"""Stage one reusable activation/clamp family from pinned upstream scalar bodies."""
from pathlib import Path
import re,json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'staged-pointwise-followup'
for sub in ['ref_c','manifest','cpp']:(D/sub).mkdir(parents=True,exist_ok=True)
A=R/'third_party/pytorch/aten/src/ATen/native/cpu/Activation.cpp'
C=R/'third_party/pytorch/aten/src/ATen/native/cpu/TensorCompareKernel.cpp'
G=R/'third_party/pytorch/aten/src/ATen/native/cpu/Gelu.h'
rows=[]
def body_at(text,start):
 begin=text.index('{',start)+1;i=begin;depth=1
 while depth:depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[begin:i-1]
for name,source,fn,arrays,scalars in [
 ('clamp_min',C,'clamp_min_scalar_kernel_impl',['a'],['min']),
 ('clamp_max',C,'clamp_max_scalar_kernel_impl',['a'],['max']),
 ('clamp_scalar',C,'clamp_scalar_kernel_impl',['a'],['min','max']),
 ('hardsigmoid',A,'hardsigmoid_kernel',['self_val'],[]),
 ('hardswish',A,'hardswish_kernel',['x'],[]),
 ('gelu_exact',G,'scalar_gelu',['x'],[]),
 ('gelu_tanh',G,'scalar_gelu_approximated_with_tanh',['x'],[]),
]:
 text=source.read_text();constants=[]
 if source==G:
  start=text.index('T '+fn+'(T x)');body=body_at(text,start)
  cpp_prefix='using T=float; template<class V> using reduced_fp_to_float_t=V; template<class V> V reduced_fp_to_float(V x){return x;}\n'
  cpp_prefix+='\n'.join(re.findall(r'constexpr double kGelu\w+ = [^;]+;',text))+'\n'
 else:
  start=text.index('void '+fn+'(');end=text.find('\nvoid ',start+1);block=text[start:end if end>=0 else len(text)]
  if source==A:block=block[block.rindex('AT_DISPATCH_FLOATING_TYPES('):]
  begin=block.index('cpu_kernel_vec(');m=re.search(r'\[[^]]*\]\s*\(([^)]*)\)\s*(?:->\s*scalar_t\s*)?\{',block[begin:]);assert m
  assert [v.strip().split()[-1] for v in m[1].split(',')]==arrays
  body=body_at(block,begin+m.start());cpp_prefix='using scalar_t=float;\n'
  for key,value in re.findall(r'const scalar_t (\w+)\(([^;]+)\);',block[:begin]):
   if re.search(r'\b'+key+r'\b',body):constants.append((key,value))
  cpp_prefix+=''.join('const scalar_t '+key+'('+value+');\n' for key,value in constants)
 cpp='#include <algorithm>\n#include <cmath>\n'+cpp_prefix+'extern "C" float upstream('+','.join('float '+v for v in arrays+scalars)+'){\n'+body+'\n}\n'
 (D/'cpp'/ (name+'.cpp')).write_text(cpp)
 adapted=body
 adapted=re.sub(r'\s*using opmath_t = [^;]+;','',adapted)
 adapted=adapted.replace('const auto ','const float ').replace('auto ','float ')
 adapted=adapted.replace('reduced_fp_to_float(x)','(x)')
 adapted=re.sub(r'opmath_t\(([^()]*)\)',r'((float)(\1))',adapted)
 adapted=adapted.replace('std::min','PG_MIN').replace('std::max','PG_MAX').replace('std::clamp','PG_CLAMP').replace('std::tanh','tanhf').replace('std::erf','erff')
 for a in arrays:adapted=re.sub(r'\b'+a+r'\b',a+'[i]',adapted)
 adapted=re.sub(r'\breturn\b','output[i] =',adapted)
 prefix='/* FP32 contiguous specialization of pinned '+fn+'.\n * Scalar operation order and parameter roles are retained. See manifest. */\n#include <math.h>\n#define PG_MIN(A,B) ((B)<(A)?(B):(A))\n#define PG_MAX(A,B) ((A)<(B)?(B):(A))\n#define PG_CLAMP(A,L,H) PG_MIN(PG_MAX(A,L),H)\n#ifndef NETLIB_SPEC\n#error "NETLIB_SPEC required"\n#endif\n'
 if source==G:prefix+=''.join('#define '+key+' ('+value+')\n' for key,value in re.findall(r'constexpr double (kGelu\w+) = ([^;]+);',text))
 args=['n']+arrays+['output']+scalars
 c=prefix+'void NETLIB_SPEC('+','.join(['int n']+['float '+a+'[]' for a in arrays]+['float output[]']+['float '+v for v in scalars])+'){\n'
 c+=''.join('const float '+k+'='+v+';\n' for k,v in constants)+'for(int i=0;i<n;++i){\n'+adapted+'\n}\n}\n'
 stem='aten_'+name;ref=D/'ref_c'/(stem+'.c');ref.write_text(c)
 man=dict(routine=stem,source='Pinned ATen scalar implementation; reusable contiguous FP32 reference',upstream_files=[dict(path=str(source.relative_to(R)),sha256=hashlib.sha256(source.read_bytes()).hexdigest())],adaptation=fn+' specialized to float. TensorIterator maps to a C loop. Standard min/max/clamp keep comparison/select semantics; float math overloads map to equivalent C functions. C++ constexpr constants expand at their use sites. Scalars remain parameters. The proprietary cuDNN backend graph is mechanically emitted from this reference, with separately tested opcode semantics; this is not imported cuDNN source.',specs=[dict(name=stem+'_f32',target=dict(library='cuDNN',call='cudnnBackendExecute',backends=['cuda'],runtime_adapter=dict(schema=1,kind='pointwise_reference',extent_arg='n')),args=args,assumptions=['positive n <= INT_MAX','contiguous FP32 arrays; output disjoint from inputs','errno and floating exception flags not observed'],deviations=[])])
 (D/'manifest'/(stem+'.json')).write_text(json.dumps(man,indent=2)+'\n')
 rows.append(dict(name=name,source=str(source.relative_to(R)),function=fn,arrays=arrays,scalars=scalars,body_sha256=hashlib.sha256(body.encode()).hexdigest(),reference_sha256=hashlib.sha256(c.encode()).hexdigest()))
(D/'extraction.json').write_text(json.dumps(rows,indent=2)+'\n');print('Staged',len(rows),'parameterized references')
