"""Typed C-wrapper ABI binding; no computation recognition or formulas."""
import functools
import hashlib
import re
import subprocess

from c_types import canonical_type


@functools.lru_cache(maxsize=8)
def _preprocess_header(path, digest, compiler):
    result = subprocess.run([compiler, '-E', '-P', '-x', 'c', path],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError('cannot preprocess runtime declarations: ' + result.stderr[-300:])
    return result.stdout


def _parameter(parameter, text):
    pointer = parameter.count('*')
    if pointer > 1 or '[' in parameter or '(' in parameter:
        raise ValueError('unsupported wrapper parameter: ' + parameter)
    if pointer:
        base, name = parameter.split('*')
        if name.strip() and not re.fullmatch(r'\s*\w+\s*', name):
            raise ValueError('unsupported pointer qualifier: ' + parameter)
        return canonical_type(base, text), True, bool(re.search(r'\bconst\b', base))
    try:
        return canonical_type(parameter, text), False, False
    except ValueError:
        named = re.fullmatch(r'(.+?)\s+\w+', parameter.strip())
        if named:
            return canonical_type(named[1], text), False, False
        raise


def contract(A, spec, target, ret, args):
    cfg = target['runtime_adapter']
    allowed = {'schema', 'kind', 'entry', 'arguments', 'arrays', 'bounds', 'alias_policy', 'linkage'}
    if (set(cfg) - allowed or cfg.get('schema') != 1
            or cfg.get('kind') != 'typed_c_wrapper' or ret not in ('void', 'float', 'double')):
        raise A.AbiError('unsupported typed C-wrapper schema')
    if cfg.get('linkage', 'required') not in ('required', 'optional'):
        raise A.AbiError('unsupported typed C-wrapper linkage')
    backend = target.get('backends')
    if backend not in (['cuda'], ['pva']):
        raise A.AbiError('typed C-wrapper requires one supported backend')
    if cfg.get('alias_policy', 'disjoint') != 'disjoint':
        raise A.AbiError('typed C-wrapper requires disjoint outputs')
    entry = cfg.get('entry', '')
    family, call = target['library'], target['call']
    if not all(isinstance(v, str) and re.fullmatch(r'[A-Za-z_]\w*', v)
               for v in (entry, family, call)):
        raise A.AbiError('invalid typed C-wrapper identity')
    header = A.REPO / 'runtime/polygeist_cublas_rt.h'
    raw = header.read_text()
    marker = 'POLYGEIST_TYPED_VENDOR_ENTRY ' + family + ' ' + call + ' ' + entry
    if not re.search(r'/\*\s*' + re.escape(marker) + r'\s*\*/', raw):
        raise A.AbiError('runtime entry lacks the selected vendor binding')
    try:
        text = _preprocess_header(str(header), hashlib.sha256(raw.encode()).hexdigest(), A.CLANG)
        declaration = re.search(r'\b(void|float|double)\s+' + re.escape(entry) + r'\s*\(([^;{}]*)\)\s*;', text)
        if not declaration or declaration[1] != ret:
            raise ValueError('runtime/reference return types disagree')
        prototype = [_parameter(p.strip(), text) for p in A._split_top(declaration[2])]
    except ValueError as error:
        raise A.AbiError(str(error)) from error
    ordered = cfg.get('arguments')
    references = {arg.name: arg for arg in args}
    if (not isinstance(ordered, list) or len(ordered) != len(prototype)
            or any(not isinstance(v, str) or v not in references for v in ordered)
            or set(ordered) != set(references)):
        raise A.AbiError('runtime contract must bind every reference and wrapper parameter')
    for name, (element, pointer, const) in zip(ordered, prototype):
        arg = references[name]
        if (canonical_type(arg.ctype) != element or (arg.kind == 'array') != pointer
                or (pointer and const and arg.intent in ('out', 'inout'))):
            raise A.AbiError('runtime/reference parameter types or intents disagree: ' + name)
        if not pointer and not ((arg.kind == 'int' and element == 'int') or
                                (arg.kind == 'scalar' and element in ('float', 'double'))):
            raise A.AbiError('wrapper scalar requires exact int32, FP32, or FP64 storage')
    arrays = {a.name: a for a in args if a.kind == 'array'}
    dimensions = {a.name for a in args if a.kind == 'int'}
    shapes = cfg.get('arrays', {})
    bounds = cfg.get('bounds', {})
    if not isinstance(shapes, dict) or set(shapes) != set(arrays):
        raise A.AbiError('every array needs an explicit extent contract')
    if not isinstance(bounds, dict) or set(bounds) != dimensions:
        raise A.AbiError('every integer argument needs explicit bounds')
    for name, bound in bounds.items():
        if (not isinstance(bound, dict) or set(bound) != {'min', 'max'}
                or any(type(v) is not int for v in bound.values())
                or not 0 <= bound['min'] <= bound['max'] <= 2147483647):
            raise A.AbiError('invalid integer bounds: ' + name)
    for name, shape in shapes.items():
        arg = arrays[name]
        def valid_dimension(d):
            if type(d) is int:
                return 0 < d <= 2147483647
            if isinstance(d, str):
                return d in dimensions and bounds[d]['min'] > 0
            return (isinstance(d, dict) and set(d) == {'dimension', 'offset'}
                    and isinstance(d['dimension'], str) and d['dimension'] in dimensions
                    and type(d['offset']) is int and -2147483647 <= d['offset'] <= 2147483647
                    and 0 < bounds[d['dimension']]['min'] + d['offset']
                    and bounds[d['dimension']]['max'] + d['offset'] <= 2147483647)
        if (not isinstance(shape, list) or arg.rank not in (1, 2)
                or len(shape) != arg.rank or any(not valid_dimension(d) for d in shape)):
            raise A.AbiError('array extents require positive fixed sizes or bounded dimensions')
        if arg.vla_ld is not None and shape[-1] != arg.vla_ld:
            raise A.AbiError('array extent disagrees with the reference leading dimension')
    params, binding = [], {}
    for i, arg in enumerate(args):
        if arg.kind == 'array':
            param = A.AbiParam(arg.name, 'array', elem=canonical_type(arg.ctype), rank=arg.rank,
                               layout=arg.name + '_stride', written=arg.intent in ('out', 'inout'),
                               const=arg.intent == 'in')
        elif arg.kind == 'int':
            param = A.AbiParam(arg.name, 'int')
        else:
            param = A.AbiParam(arg.name, 'scalar', elem=arg.ctype)
        params.append(param)
        binding[arg.name] = ('arg', i)
        if arg.kind == 'array':
            params.append(A.AbiParam(param.layout, 'int', layout_of=arg.name))
            binding[param.layout] = ('const_num', 1)
    return params, binding, dict(cfg, ordered_arguments=ordered,
                                 memory_domain='cuda_device' if backend == ['cuda'] else 'host')


def extent_expression(dimension):
    if isinstance(dimension, dict):
        return f'((int64_t){dimension["dimension"]} + ({dimension["offset"]}))'
    return f'(int64_t){dimension}'


def generate(ca, prototype):
    """Mechanically marshal an already proved reference to its typed C entry."""
    cfg = ca.runtime_adapter
    arrays = [p for p in ca.params if p.tkind == 'array']
    cuda = cfg['memory_domain'] == 'cuda_device'
    declaration = ''
    if cfg.get('linkage', 'required') == 'optional':
        declaration = f'extern __typeof__({cfg["entry"]}) {cfg["entry"]} __attribute__((weak));\n'
    body = [declaration + prototype(ca) + ' {\n']
    if declaration:
        body.append(f'  if (!{cfg["entry"]})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "optional vendor entry is not linked");\n')
    for name, bound in cfg['bounds'].items():
        body.append(f'  if ({name} < {bound["min"]} || {name} > {bound["max"]})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "integer outside wrapper bounds");\n')
    for p in arrays:
        dims = cfg['arrays'][p.name]
        body.append(f'  int64_t pv2_bytes_{p.name} = sizeof({p.elem});\n')
        for dim in dims:
            expr = extent_expression(dim)
            body.append(f'  if ({expr} <= 0 || pv2_bytes_{p.name} > INT64_MAX / {expr})\n'
                        f'    polygeist_v2_fatal("{ca.call}", "array extent overflow");\n'
                        f'  pv2_bytes_{p.name} *= {expr};\n')
        stride = '1' if p.rank == 1 else extent_expression(dims[-1])
        body.append(f'  if ({p.layout} != {stride} || !{p.name} || {p.name}_bytes < pv2_bytes_{p.name})\n'
                    f'    polygeist_v2_fatal("{ca.call}", "array layout or capacity mismatch");\n')
    for i, p in enumerate(arrays):
        for q in arrays[i + 1:]:
            if p.written or q.written:
                body.append(f'  if (polygeist_v2_overlaps({p.name}, {p.name}_bytes, {q.name}, {q.name}_bytes))\n'
                            f'    polygeist_v2_fatal("{ca.call}", "unsupported output overlap");\n')
    if cuda:
        body += ['  polygeist_v2_map_t pv2_map;\n',
                 f'  polygeist_v2_map_init(&pv2_map, "{ca.call}", "{cfg["entry"]}");\n']
        for p in arrays:
            body.append(f'  int pv2_k_{p.name} = polygeist_v2_map_add(&pv2_map, {p.name}, {p.name}_bytes, {int(p.written)});\n')
        body += ['  polygeist_v2_map_to_device(&pv2_map);\n',
                 '  polygeist_require_gpu_libraries_begin();\n',
                 '  polygeist_cublas_pipeline_begin();\n']
    else:
        body.append('  double pv2_t0 = polygeist_v2_now_ms();\n')
    by_name = {p.name: p for p in arrays}
    arguments = []
    for name in cfg['ordered_arguments']:
        p = by_name.get(name)
        if cuda and p:
            element = ('const ' if not p.written else '') + p.elem
            arguments.append(f'({element} *)polygeist_v2_map_dev(&pv2_map, pv2_k_{name})')
        else:
            arguments.append(name)
    result = f'{ca.returns} pv2_result = ' if ca.returns else ''
    body.append(f'  {result}{cfg["entry"]}({", ".join(arguments)});\n')
    if cuda:
        body += ['  polygeist_cublas_pipeline_end();\n',
                 '  polygeist_require_gpu_libraries_end();\n',
                 '  polygeist_v2_map_to_host(&pv2_map);\n']
    else:
        body.append(f'  polygeist_v2_done("{ca.call}", "{cfg["entry"]}", pv2_t0);\n')
    if ca.returns:
        body.append('  return pv2_result;\n')
    return ''.join(body) + '}\n\n'
