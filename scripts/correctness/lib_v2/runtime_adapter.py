"""Typed backend adapters for already-proven imported references.

Pointwise code generation reads the validated raised library body. It does not
recognize applications or contain formulas for named operations.
"""
import functools
import math
import re
import sys
from pathlib import Path

# Ordinary instruction lowering to the vendor API's operation enum.
POINTWISE = {'arith.addf':'ADD','arith.subf':'SUB','arith.mulf':'MUL',
             'arith.divf':'DIV','arith.negf':'NEG','math.exp':'EXP',
             'math.sqrt':'SQRT','math.absf':'ABS','math.log':'LOG',
             'math.sin':'SIN','math.cos':'COS','math.tanh':'TANH_FWD',
             'math.floor':'FLOOR','math.ceil':'CEIL','math.erf':'ERF',
             'math.tan':'TAN','math.powf':'POW','math.atan2':'ATAN2'}

# cuDNN's floating comparisons have C's ordered relational / unordered !=
# behavior. Other MLIR predicates require additional operations, not renaming.
COMPARISONS = {'oeq':'CMP_EQ','une':'CMP_NEQ','ogt':'CMP_GT',
               'oge':'CMP_GE','olt':'CMP_LT','ole':'CMP_LE'}


def reference_region(A, spec, args, extent):
    sys.path.insert(0,str(A.REPO/'scripts/correctness'))
    import mlir_lite as ML
    import kernel_match as K
    suite=A.SPEC_DIR.name.removeprefix('lib_')
    text=(A.SPEC_DIR.parent/f'kernel_library_{suite}.mlir').read_text()
    fs=[f for f in ML.Module(text).funcs if f.name==spec]
    if len(fs)!=1:raise A.AbiError('runtime adapter requires an emitted reference body')
    f=fs[0];gs=[o for o in f.ops if o.name=='linalg.generic']
    if len(gs)!=1:raise A.AbiError('pointwise runtime requires exactly one parallel reference region')
    g=gs[0];gb=K.parse_generics(g.text)[0]
    identity=lambda s:re.sub(r'\s+','',s)=='affine_map<(d0)->(d0)>'
    if g.iters!=['parallel'] or not all(identity(x) for x in g.maps) or len(g.outs)!=1:
        raise A.AbiError('pointwise reference requires identity rank-one maps and one output')
    defs={r:o for o in f.ops for r in o.results}
    def root(v):
        o=defs.get(v)
        if o is None:return v.lstrip('%')
        if o.name!='polygeist.submap' or 'map=affine_map<(d0)->(d0)>' not in re.sub(r'\s+','',o.text):
            raise A.AbiError('pointwise adapter cannot lower this reference view')
        bound=defs.get(o.operands[1])
        if not bound or bound.name!='arith.index_cast' or bound.operands!=['%'+extent]:
            raise A.AbiError('pointwise extent must bind the exact reference dimension')
        return root(o.operands[0])
    inputs=[root(x) for x in g.ins];output=root(g.outs[0])
    if len(inputs)>4 or len(set(inputs))!=len(inputs):raise A.AbiError('unsupported pointwise input interface')
    if output not in {a.name for a in args if a.intent=='out'}:
        raise A.AbiError('pointwise output must overwrite its entire reference argument')
    if set(inputs+[output])!={a.name for a in args if a.kind=='array'}:
        raise A.AbiError('unbound reference array in pointwise adapter')
    return f, g, gb, inputs, output


def pointwise_graph(A, spec, args, extent):
    f, g, gb, inputs, output = reference_region(A, spec, args, extent)
    import kernel_match as K
    # Read opcode numbers from the runtime ABI's own enum dispatch.
    runtime=(A.REPO/'runtime/polygeist_cublas_rt_cuda.c').read_text()
    modes={name:int(op) for op,name in re.findall(r'case (\d+): \*mode = CUDNN_POINTWISE_(\w+); return true;',runtime)}
    values=dict(zip(gb.ins_arg_names,range(len(inputs))))
    types={v:'f32' for v in values};scalars=[];scalar_arguments={};instructions=[]
    for name,value in K.parse_generics(f.text)[0].constants.items():
        if not math.isfinite(value):raise A.AbiError('nonfinite pointwise reference constant')
        if len(scalars)==8:raise A.AbiError('too many pointwise constants')
        values[name]=4+len(scalars);types[name]='f32';scalars.append(value)
    for arg in args:
        if arg.kind!='scalar':continue
        if arg.ctype!='float' or ('%'+arg.name,'f32') not in f.args:
            raise A.AbiError('pointwise scalar capture must preserve FP32 type')
        if '%'+arg.name not in gb.captures:raise A.AbiError('unused pointwise scalar parameter')
        if len(scalars)==8:raise A.AbiError('too many pointwise scalar operands')
        scalar_arguments[arg.name]=len(scalars)
        values['%'+arg.name]=4+len(scalars);types['%'+arg.name]='f32';scalars.append(0.0)
    def emit(mode,operands,result,result_type):
        if any(x not in values for x in operands):raise A.AbiError('unbound/recursive pointwise scalar')
        if A.enum_of_constant('cuDNN','CUDNN_POINTWISE_'+mode) is None:
            raise A.AbiError('vendor header lacks CUDNN_POINTWISE_'+mode)
        refs=[values[x] for x in operands]+[0,0,0]
        instructions.append((modes[mode]<<24)|(refs[0]<<16)|(refs[1]<<8)|refs[2])
        values[result]=12+len(instructions)-1;types[result]=result_type
    for line in gb.body_lines:
        m=re.fullmatch(r'(%[\w.]+) = ([\w.]+) (.*?) : (f32|i1)',line.strip())
        if not m:raise A.AbiError('unsupported reference pointwise instruction: '+line)
        result,opcode,raw,ty=m.groups();operands=[x.strip() for x in raw.split(',')]
        if opcode=='arith.cmpf':
            predicate=operands.pop(0)
            if predicate not in COMPARISONS:raise A.AbiError('unsupported floating comparison predicate: '+predicate)
            mode=COMPARISONS[predicate];expected=['f32','f32'];result_type='i1'
        elif opcode=='arith.select':
            mode='BINARY_SELECT';expected=['i1','f32','f32'];result_type='f32'
        elif opcode in ('arith.andi','arith.ori'):
            mode='LOGICAL_AND' if opcode=='arith.andi' else 'LOGICAL_OR'
            expected=['i1','i1'];result_type='i1'
        elif opcode in POINTWISE:
            mode=POINTWISE[opcode]
            arity=2 if opcode in ('arith.addf','arith.subf','arith.mulf','arith.divf','math.powf','math.atan2') else 1
            expected=['f32']*arity;result_type='f32'
        else:raise A.AbiError('unsupported reference pointwise instruction: '+line)
        if any(x not in values for x in operands):raise A.AbiError('unbound/recursive pointwise scalar')
        if [types[x] for x in operands]!=expected or ty!=expected[-1]:
            raise A.AbiError('pointwise instruction operand types disagree')
        emit(mode,operands,result,result_type)
    if not instructions or len(instructions)>24 or gb.yield_values!=[next(reversed(values))] or types[gb.yield_values[0]]!='f32':
        raise A.AbiError('pointwise output must be the final supported instruction')
    words=[0]*12
    for i,v in enumerate(instructions):words[i//2]|=v<<(32*(i%2))
    words=[v if v<2**63 else v-2**64 for v in words]
    return dict(inputs=inputs,output=output,words=words,scalars=scalars,
                scalar_arguments=scalar_arguments,nodes=len(instructions))


def unary_reference(A, spec, args, extent):
    """Lower one imported FP32 scalar primitive to a cuTENSOR unary enum."""
    f, g, gb, inputs, output = reference_region(A, spec, args, extent)
    if len(inputs) != 1 or len(gb.body_lines) != 1 or gb.captures:
        raise A.AbiError('unary adapter requires one scalar primitive and no captures')
    line = gb.body_lines[0].strip()
    call = re.fullmatch(r'(%[\w.]+) = func.call @(acosf|acoshf|asinf|asinhf|atanf|atanhf|coshf|sinhf|tanf)\((%[\w.]+)\) : \(f32\) -> f32', line)
    operation = re.fullmatch(r'(%[\w.]+) = math\.(acos|acosh|asin|asinh|atan|atanh|cosh|sinh|tan) (%[\w.]+) : f32', line)
    if call:
        result, symbol, operand = call.groups()
        opcode = symbol[:-1].upper()
    elif operation:
        result, symbol, operand = operation.groups()
        opcode = symbol.upper()
    else:
        raise A.AbiError('unsupported typed unary reference primitive: ' + line)
    if operand != gb.ins_arg_names[0] or gb.yield_values != [result]:
        raise A.AbiError('unary primitive must consume the input and produce the output')
    runtime_enum = 'POLYGEIST_CUTENSOR_UNARY_' + opcode
    vendor_enum = 'CUTENSOR_OP_' + opcode
    raw = (A.REPO / 'runtime/polygeist_cublas_rt_cuda.c').read_text()
    if not re.search(r'case\s+' + runtime_enum + r':\s*return\s+' + vendor_enum + r';', raw):
        raise A.AbiError('runtime does not implement the selected unary opcode')
    if A.enum_of_constant('cuTENSOR', vendor_enum) is None:
        raise A.AbiError('vendor header does not declare the selected unary opcode')
    header = (A.REPO / 'runtime/polygeist_cublas_rt.h').read_text()
    if not re.search(r'\b' + runtime_enum + r'\b', header):
        raise A.AbiError('runtime header does not declare the selected unary opcode')
    return dict(input=inputs[0], output=output, opcode=runtime_enum, vendor_opcode=vendor_enum)


def contract(A,spec,target,ret,args):
    cfg=target['runtime_adapter']
    family = ('cuTENSOR', 'cutensorPermute') if cfg['kind']=='unary_reference' else ('cuDNN', 'cudnnBackendExecute')
    if ret!='void' or target.get('backends')!=['cuda'] or (target['library'],target['call'])!=family or cfg.get('schema')!=1:
        raise A.AbiError('unsupported typed runtime adapter contract')
    A.resolve_proto(*family)
    header=A._strip_c_comments((A.REPO/'runtime/polygeist_cublas_rt.h').read_text())
    declaration=re.search(r'void\s+polygeist_cudnn_pointwise_graph_f32\s*\(([^;]+)\);',header)
    if not declaration:raise A.AbiError('pointwise runtime entry unavailable')
    prototype=[A._parse_c_param(x).ctype for x in A._split_top(declaration[1])]
    expected=['int32_t']+['int64_t']*12+['int32_t']+['float']*8+['int32_t']*5+['const float*']*4+['float*']
    if prototype!=expected:raise A.AbiError('pointwise runtime prototype changed')
    extent=cfg['extent_arg'];ints=[a for a in args if a.kind=='int']
    if len(ints)!=1 or ints[0].name!=extent or ints[0].ctype!='int':raise A.AbiError('runtime adapter requires one exact integer extent')
    params=[];binding={}
    for i,a in enumerate(args):
        if a.kind=='int':p=A.AbiParam(a.name,'int')
        elif a.kind=='scalar' and a.ctype=='float' and cfg['kind'] in ('pointwise_reference','vendor_runtime'):
            p=A.AbiParam(a.name,'scalar',elem='float')
        elif a.kind=='array' and a.ctype=='float' and a.rank==1:
            p=A.AbiParam(a.name,'array',elem='float',rank=1,layout=a.name+'_stride',written=a.intent in ('out','inout'),const=a.intent=='in')
        else:raise A.AbiError('runtime adapter supports contiguous FP32 rows only')
        params.append(p);binding[p.name]=('arg',i)
        if a.kind=='array':params.append(A.AbiParam(p.layout,'int',layout_of=p.name));binding[p.layout]=('const_num',1)
    if cfg['kind']=='pointwise_reference':
        graph=pointwise_graph(A,spec,args,extent)
        return params,binding,dict(cfg,graph=graph,entry='polygeist_cudnn_pointwise_graph_f32')
    if cfg['kind']=='unary_reference':
        declaration=re.search(r'void\s+polygeist_cutensor_unary_f32\s*\(([^;]+)\);',header)
        if not declaration:raise A.AbiError('cuTENSOR unary runtime entry unavailable')
        prototype=[A._parse_c_param(x).ctype for x in A._split_top(declaration[1])]
        if prototype!=['int32_t','int32_t','const float*','float*']:
            raise A.AbiError('cuTENSOR unary runtime prototype changed')
        primitive=unary_reference(A,spec,args,extent)
        return params,binding,dict(cfg,primitive=primitive,entry='polygeist_cutensor_unary_f32')
    if cfg['kind']=='vendor_runtime':
        raw=(A.REPO/'runtime/polygeist_cublas_rt.h').read_text()
        entry=cfg['entry']
        declaration=re.search(r'/\* POLYGEIST_VENDOR_ADAPTER '+re.escape(target['library'])+' '+re.escape(target['call'])+r' \*/\s*void\s+'+re.escape(entry)+r'\s*\(([^;]+)\);',raw)
        if not declaration:raise A.AbiError('GPU-only vendor runtime binding is unavailable')
        prototype=[A._parse_c_param(x) for x in A._split_top(declaration[1])]
        if set(cfg['arguments'])!={p.name for p in prototype}:raise A.AbiError('incomplete runtime argument mapping')
        refs={a.name:a for a in args};used=[]
        for p in prototype:
            name=cfg['arguments'][p.name];a=refs.get(name);used.append(name)
            expected='float' if (p.is_ptr and p.base=='float') or p.ctype=='float' else 'int' if p.ctype=='int32_t' else None
            if a is None or a.ctype!=expected or (a.kind=='array')!=p.is_ptr or (a.intent in ('out','inout') and p.ctype.startswith('const')):
                raise A.AbiError('runtime/reference parameter types or intents disagree')
        if set(used)!=set(refs):raise A.AbiError('unbound reference parameter in runtime adapter')
        return params,binding,dict(cfg,ordered_arguments=[cfg['arguments'][p.name] for p in prototype])
    raise A.AbiError('unsupported runtime adapter kind')


def bind_spec(A,spec,target,ret,args):
    _,binding,_=contract(A,spec,target,ret,args)
    return A.SpecAbi(spec,target['call']+'_'+spec,ret,args,binding,{})


def build_call(A,call,specs):
    if len(specs)!=1:raise A.AbiError('runtime adapter identity must select one imported body')
    sa=specs[0];target=A.manifest_entry(sa.spec)[1]['target'];params,_,cfg=contract(A,sa.spec,target,sa.ret,sa.args)
    return A.CallAbi(call,target['library'],target['call'],params,[],specs=[sa.spec],backends=['cuda'],runtime_adapter=cfg)
