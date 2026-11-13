"""Stage reusable typed CUB reduction references from the pinned scalar loop."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'staged-cub-reductions'
for sub in ('ref_c', 'manifest'):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
header = ROOT / 'third_party/cuda_headers/cuda_cccl/include/cccl/cub/thread/thread_reduce.cuh'
source = header.read_text()
start = source.index('ThreadReduceSequential(const Input& input, ReductionOp reduction_op)')
start = source.index('{', start)
end = source.index('\n}', start) + 2
body = source[start:end]
assert 'retval = reduction_op(retval, input[i]);' in body
# The public seeded overload constructs [prefix, input[0], ..., input[N-1]].
# Inline that temporary into the sequential implementation and shift its loop
# index by one. This preserves every scalar operation and its order.
reference = body.replace('auto retval = static_cast<AccumT>(input[0]);', 'SCALAR retval = seed;')
reference = reference.replace('  _CCCL_PRAGMA_UNROLL_FULL()\n', '')
reference = reference.replace('for (int i = 1; i < static_size_v<Input>; ++i)',
                              'for (int i = 0; i < n; ++i)')
reference = reference.replace('reduction_op(retval, input[i])', 'REDUCE(retval, input[i])')
reference = reference.replace('return retval;', 'output[0] = retval;')
prefix = '''/* Pinned CUB seeded ThreadReduce, sequential scalar implementation.
 * Inline its prefix-array construction; preserve scalar order. Static length
 * becomes a runtime extent. The parallel DeviceReduce backend is separately
 * validated within a declared tolerance; bitwise parallel equivalence is not
 * implied. SCALAR and REDUCE are template specializations from the manifest. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if REDUCE_PRODUCT
#define REDUCE(a,b) ((a) * (b))
#else
#define REDUCE(a,b) ((a) + (b))
#endif
void NETLIB_SPEC(int n, SCALAR seed, SCALAR input[], SCALAR output[1])
'''
(OUT / 'ref_c/cub_reduce.c').write_text(prefix + reference + '\n')
# Independently compile the retained upstream loop with its prefix array.
cpp = body.replace('  _CCCL_PRAGMA_UNROLL_FULL()\n', '')
cpp = cpp.replace('static_size_v<Input>', 'N')
(OUT / 'upstream-specialization.hpp').write_text(
    'template<class AccumT,int N,class ReductionOp>\n'
    'AccumT original(const AccumT (&input)[N],ReductionOp reduction_op)\n' + cpp + '\n'
    'template<class T,int N,class Op> T original_seeded(const T (&input)[N],Op op,T seed) {\n'
    ' T prefixed[N+1]; prefixed[0]=seed;\n'
    ' for(int i=0;i<N;++i) prefixed[i+1]=input[i];\n'
    ' return original<T>(prefixed,op);\n}\n')
api = ROOT / 'third_party/cuda_headers/cuda_cccl/include/cccl/cub/device/device_reduce.cuh'
specs = []
for operation, scalar, dtype in [('sum','float','f32'),('sum','double','f64'),
                                  ('product','float','f32')]:
    name = 'cub_reduce_' + operation + '_' + dtype
    specs.append(dict(name=name, defines={'SCALAR':scalar,'REDUCE_PRODUCT':int(operation=='product')},
        target=dict(library='CUB',call='DeviceReduce',backends=['cuda'],runtime_adapter=dict(
            schema=1,kind='typed_c_wrapper',entry='polygeist_'+name,
            arguments=['n','seed','input','output'],arrays={'input':['n'],'output':[1]},
            bounds={'n':{'min':1,'max':2147483647}},alias_policy='disjoint',linkage='optional')),
        args=['n','seed','input','output'],assumptions=[
            'positive n <= INT_MAX, contiguous disjoint input and one-element output',
            'parallel backend reassociation is checked within declared numerical tolerance'],deviations=[]))
manifest = dict(routine='cub_reduce', source='Pinned public CUB ThreadReduceSequential and seeded ThreadReduce contract',
    upstream_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in (header,api)], adaptation='The seeded overload prepends the seed to its input array. '
    'Inline that temporary into the retained sequential reduction loop, shift its index by one, '
    'specialize the scalar/operator types, and expose the return as one output element. '
    'Runtime n replaces static array length. DeviceReduce uses the same typed binary operator and seed '
    'with parallel association checked separately.',specs=specs)
(OUT / 'manifest/cub_reduce.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Staged three reusable seeded reduction specializations')
