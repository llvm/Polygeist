"""Typed descriptor API contracts; no recognition or computational formulas.

The initial runtime supports CUDA cuDNN tensor4d descriptors over contiguous
FP32 rows. Unsupported contract forms fail closed. Each argument is checked
against the real vendor header and the imported reference's C signature.
"""
import math
import re


def contract(A, target, args, proto):
    cfg = target['descriptor_adapter']
    if target['library'] != 'cuDNN' or cfg.get('schema') != 1:
        raise A.AbiError('unsupported descriptor adapter family/schema')
    if target.get('backends') != ['cuda']:
        raise A.AbiError('descriptor adapter requires CUDA-only capability')
    names = {a.name: i for i, a in enumerate(args)}
    params, binding, used = [], {}, set()
    for entry in cfg['parameters']:
        name, kind = entry['name'], entry['kind']
        if not re.fullmatch(r'[A-Za-z_]\w*', name) or name in binding:
            raise A.AbiError('invalid or duplicate adapter parameter')
        source = entry['source']
        if isinstance(source, str) and source in names:
            k = names[source]
            a = args[k]
            if a.kind != kind or a.ctype != ('float' if kind == 'array' else 'int'):
                raise A.AbiError(f'{name}: adapter/reference type mismatch')
            binding[name] = ('arg', k)
            used.add(k)
        elif kind == 'int' and type(source) is int and source == 1:
            binding[name] = ('const_num', source)
            a = None
        else:
            raise A.AbiError(f'{name}: unsupported adapter source')
        p = A.AbiParam(name, kind)
        if kind == 'array':
            if entry.get('rank') != 1 or entry.get('element') != 'float' or a.rank != 1:
                raise A.AbiError('descriptor runtime requires rank-one FP32 arrays')
            p.elem, p.rank, p.layout = 'float', 1, entry.get('layout')
            p.written = a.intent in ('out', 'inout')
            p.const = not p.written
        elif kind == 'int':
            p.layout_of = entry.get('layout_of')
        else:
            raise A.AbiError('unsupported descriptor wrapper parameter kind')
        params.append(p)
    if used != set(range(len(args))):
        raise A.AbiError('unbound reference argument in descriptor adapter')
    by_name = {p.name: p for p in params}
    arrays = {p.name: p for p in params if p.tkind == 'array'}
    dimensions = {p.name for p in params if p.tkind == 'int' and not p.layout_of}
    for p in arrays.values():
        layout = by_name.get(p.layout)
        if layout is None or layout.layout_of != p.name or binding[layout.name] != ('const_num', 1):
            raise A.AbiError('every descriptor array requires a checked unit stride')
    vendor = cfg['vendor_arguments']
    if set(vendor) != {p.name for p in proto.params}:
        raise A.AbiError('descriptor contract does not bind every vendor parameter')
    descriptor_arrays, pointer_arrays = set(), set()
    for p in proto.params:
        v = vendor[p.name]
        kind = v.get('kind')
        if kind == 'handle':
            ok = p.base == 'cudnnHandle_t' and not p.is_ptr
        elif kind == 'enum':
            e = A.enum_of_constant('cuDNN', v['value'])
            ok = e is not None and e[0] == p.base and not p.is_ptr
        elif kind == 'scalar_pointer':
            ok = (p.ctype == 'const void*' and v.get('type') == 'float' and
                  type(v.get('value')) in (int, float) and math.isfinite(v['value']))
        elif kind == 'array':
            ap = arrays.get(v.get('name'))
            ok = (ap is not None and p.is_ptr and p.base == 'void' and
                  (not ap.written or not p.ctype.startswith('const')))
            pointer_arrays.add(v.get('name'))
        elif kind == 'tensor4d':
            ap = arrays.get(v.get('array'))
            dims = v.get('dimensions', [])
            fmt = A.enum_of_constant('cuDNN', v.get('format', ''))
            dtype = A.enum_of_constant('cuDNN', v.get('data_type', ''))
            ok = (p.base == 'cudnnTensorDescriptor_t' and ap is not None and
                  v.get('format') == 'CUDNN_TENSOR_NCHW' and fmt is not None and
                  v.get('data_type') == 'CUDNN_DATA_FLOAT' and dtype is not None and
                  len(dims) == 4 and dims[:3] == [1, 1, 1] and dims[3] in dimensions)
            descriptor_arrays.add(v.get('array'))
        else:
            ok = False
        if not ok:
            raise A.AbiError(f'{p.name}: descriptor argument disagrees with vendor ABI')
    if descriptor_arrays != set(arrays) or pointer_arrays != set(arrays):
        raise A.AbiError('every array requires a matching descriptor and pointer')
    return params, binding


def bind_spec(A, spec, target, ret, args, proto):
    if ret != 'void':
        raise A.AbiError('descriptor adapter does not return a scalar')
    _, binding = contract(A, target, args, proto)
    return A.SpecAbi(spec, target['call'], ret, args, binding, {})


def build_call(A, call, specs):
    target = A.manifest_entry(specs[0].spec)[1]['target']
    for sa in specs:
        if A.manifest_entry(sa.spec)[1]['target'] != target:
            raise A.AbiError('inconsistent contracts for shared descriptor call')
    proto = A.resolve_proto(target['library'], call)
    params, _ = contract(A, target, specs[0].args, proto)
    return A.CallAbi(call, target['library'], proto.name, params, [],
                     specs=sorted(sa.spec for sa in specs),
                     descriptor_adapter=target['descriptor_adapter'], backends=['cuda'])
