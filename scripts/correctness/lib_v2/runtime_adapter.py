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


@functools.lru_cache(maxsize=4)
def _parsed_reference_module(text):
    """Reuse read-only MLIR parsing by exact contents, never by stale path.

    Every caller still reads the current library and validates its reference,
    ABI and backend contract. Only identical text reuses parser objects.
    """
    import mlir_lite as ML
    return ML.Module(text)


def reference_region(A, spec, args, extent):
    sys.path.insert(0,str(A.REPO/'scripts/correctness'))
    import mlir_lite as ML
    import kernel_match as K
    suite=A.SPEC_DIR.name.removeprefix('lib_')
    text=(A.SPEC_DIR.parent/f'kernel_library_{suite}.mlir').read_text()
    fs=[f for f in _parsed_reference_module(text).funcs if f.name==spec]
    if len(fs)!=1:raise A.AbiError('runtime adapter requires an emitted reference body')
    f=fs[0];gs=[o for o in f.ops if o.name=='linalg.generic']
    if len(gs)!=1:raise A.AbiError('pointwise runtime requires exactly one parallel reference region')
    g=gs[0];gb=K.parse_generics(g.text,infer_outputs_from_yield=True)[0]
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
    runtime=(A.REPO/'runtime/polygeist_cublas_rt_cuda.c').read_text()
    modes={name:int(op) for op,name in re.findall(r'case (\d+): \*mode = CUDNN_POINTWISE_(\w+); return true;',runtime)}
    values=dict(zip(gb.ins_arg_names,range(len(inputs))))
    types={v:'f32' for v in values};scalars=[];scalar_arguments={};instructions=[]
    booleans={name:literal=='true' for name,literal in re.findall(
        r'(%[\w.$-]+) = arith.constant (true|false)\b',f.text)}
    for name,value in K.parse_generics(f.text,infer_outputs_from_yield=True)[0].constants.items():
        if name in booleans:
            types[name]='i1'
            continue
        if not math.isfinite(value):raise A.AbiError('nonfinite pointwise reference constant')
        if len(scalars)==8:raise A.AbiError('too many pointwise constants')
        values[name]=4+len(scalars);types[name]='f32';scalars.append(value)
    scalar_parameters=set()
    for arg in args:
        if arg.kind!='scalar':continue
        if arg.ctype!='float' or ('%'+arg.name,'f32') not in f.args:
            raise A.AbiError('pointwise scalar capture must preserve FP32 type')
        if len(scalars)==8:raise A.AbiError('too many pointwise scalar operands')
        scalar_arguments[arg.name]=len(scalars)
        values['%'+arg.name]=4+len(scalars);types['%'+arg.name]='f32';scalars.append(0.0)
        scalar_parameters.add('%'+arg.name)
    definitions={result:op for op in f.ops for result in op.results}
    active=set();used_parameters=set();temporary=0
    boolean_integers={}

    def emit(mode,operands,result,result_type):
        if any(x not in values for x in operands):raise A.AbiError('unbound/recursive pointwise scalar')
        if A.enum_of_constant('cuDNN','CUDNN_POINTWISE_'+mode) is None or mode not in modes:
            raise A.AbiError('vendor/runtime lacks CUDNN_POINTWISE_'+mode)
        refs=[values[x] for x in operands]+[0,0,0]
        instructions.append((modes[mode]<<24)|(refs[0]<<16)|(refs[1]<<8)|refs[2])
        values[result]=12+len(instructions)-1;types[result]=result_type

    def alias(result,source):
        types[result]=types[source]
        if source in booleans:booleans[result]=booleans[source]
        else:values[result]=values[source]

    def boolean(mode,operands,result):
        # Boolean identities are exact, including when their predicates were
        # computed from NaNs. No floating comparison is complemented here.
        if mode=='LOGICAL_NOT' and operands[0] in booleans:
            booleans[result]=not booleans[operands[0]];types[result]='i1';return
        if mode in ('LOGICAL_AND','LOGICAL_OR'):
            neutral=mode=='LOGICAL_AND'
            for i,operand in enumerate(operands):
                if operand in booleans:
                    if booleans[operand]==neutral:alias(result,operands[1-i])
                    else:booleans[result]=not neutral;types[result]='i1'
                    return
        emit(mode,operands,result,'i1')

    def temp():
        nonlocal temporary
        temporary+=1
        return '@backend.boolean.'+str(temporary)

    def boolean_select(condition,left,right,result):
        if condition in booleans:alias(result,left if booleans[condition] else right);return
        if left==right:alias(result,left);return
        if right in booleans and not booleans[right]:
            boolean('LOGICAL_AND',[condition,left],result);return
        if left in booleans and booleans[left]:
            boolean('LOGICAL_OR',[condition,right],result);return
        inverted=temp();boolean('LOGICAL_NOT',[condition],inverted)
        if left in booleans and not booleans[left]:
            boolean('LOGICAL_AND',[inverted,right],result);return
        if right in booleans and booleans[right]:
            boolean('LOGICAL_OR',[inverted,left],result);return
        yes,no=temp(),temp()
        boolean('LOGICAL_AND',[condition,left],yes)
        boolean('LOGICAL_AND',[inverted,right],no)
        boolean('LOGICAL_OR',[yes,no],result)

    def ensure(value):
        if value in scalar_parameters:used_parameters.add(value)
        if value in values or value in booleans or value in boolean_integers:return
        operation=definitions.get(value)
        if operation is None or value in active or len(operation.results)!=1:
            raise A.AbiError('unbound/recursive pointwise scalar: '+value)
        active.add(value)
        lower_instruction(operation.text.strip())
        active.remove(value)

    def scalar_literal(value):
        # Exact FP32 constants used by predicate conversions: only 0 and +/-1.
        for name,index in values.items():
            if 4 <= index < 4+len(scalars) and index-4 not in scalar_arguments.values() and types[name]=='f32' and scalars[index-4].hex()==float(value).hex():
                return name
        if len(scalars)==8:raise A.AbiError('too many pointwise constants')
        name=temp();values[name]=4+len(scalars);types[name]='f32';scalars.append(float(value))
        return name

    def lower_conversion(line):
        m=re.fullmatch(r'(%[\w.$-]+) = arith\.(extui|extsi|uitofp|sitofp) (%[\w.$-]+) : (i\d+) to (i\d+|f32)',line)
        if not m:return False
        result,opcode,operand,source,target=m.groups();ensure(operand)
        if types.get(operand)!=source:raise A.AbiError('predicate conversion source type mismatch')
        if opcode in ('extui','extsi'):
            if source!='i1' or not target.startswith('i') or not 1<int(target[1:])<=64:
                raise A.AbiError('only exact boolean integer extensions are supported')
            boolean_integers[result]=(operand,1 if opcode=='extui' else -1)
            types[result]=target;return True
        if target!='f32':raise A.AbiError('predicate conversion must produce FP32')
        if source=='i1':predicate,value=operand,(1 if opcode=='uitofp' else -1)
        elif operand in boolean_integers:
            predicate,value=boolean_integers[operand]
            if value<0 and opcode=='uitofp':
                raise A.AbiError('unsigned conversion of sign-extended predicate is unsupported')
        else:raise A.AbiError('integer-to-FP32 conversion requires a proven boolean value')
        yes,no=scalar_literal(value),scalar_literal(0.)
        if predicate in booleans:alias(result,yes if booleans[predicate] else no)
        else:emit('BINARY_SELECT',[predicate,yes,no],result,'f32')
        return True

    def lower_instruction(line):
        if lower_conversion(line.strip()):return
        m=re.fullmatch(r'(%[\w.$-]+) = ([\w.]+) (.*?) : (f32|i1)',line.strip())
        if not m:raise A.AbiError('unsupported reference pointwise instruction: '+line)
        result,opcode,operands,ty=m.groups();operands=[x.strip() for x in operands.split(',')]
        if opcode=='arith.cmpf':
            predicate=operands.pop(0)
            if predicate not in COMPARISONS:raise A.AbiError('unsupported floating comparison predicate: '+predicate)
            mode=COMPARISONS[predicate];expected=['f32','f32'];result_type='i1'
        elif opcode=='arith.select':
            mode='BINARY_SELECT';expected=['i1',ty,ty];result_type=ty
        elif opcode=='arith.xori' or (opcode=='arith.cmpi' and operands[0] in ('eq','ne')):
            complement=False
            if opcode=='arith.cmpi':complement=operands.pop(0)=='eq'
            for operand in operands:ensure(operand)
            if ty!='i1' or [types[x] for x in operands]!=['i1','i1']:
                raise A.AbiError('only boolean xor/equality is supported')
            left,right=operands
            if right in booleans:
                if booleans[right] != complement:boolean('LOGICAL_NOT',[left],result)
                else:alias(result,left)
            else:
                inverted=temp();boolean('LOGICAL_NOT',[right],inverted)
                boolean_select(left,right if complement else inverted,inverted if complement else right,result)
            return
        elif opcode in ('arith.andi','arith.ori'):
            mode='LOGICAL_AND' if opcode=='arith.andi' else 'LOGICAL_OR'
            expected=['i1','i1'];result_type='i1'
        elif opcode in POINTWISE:
            mode=POINTWISE[opcode]
            arity=2 if opcode in ('arith.addf','arith.subf','arith.mulf','arith.divf','math.powf','math.atan2') else 1
            expected=['f32']*arity;result_type='f32'
        else:raise A.AbiError('unsupported reference pointwise instruction: '+line)
        for operand in operands:ensure(operand)
        if [types[x] for x in operands]!=expected or ty!=expected[-1]:
            raise A.AbiError('pointwise instruction operand types disagree')
        if opcode=='arith.select' and ty=='i1':boolean_select(*operands,result)
        elif opcode=='arith.select' and operands[0] in booleans:
            alias(result,operands[1] if booleans[operands[0]] else operands[2])
        elif mode in ('LOGICAL_AND','LOGICAL_OR'):boolean(mode,operands,result)
        else:emit(mode,operands,result,result_type)

    for line in gb.body_lines:lower_instruction(line)
    if used_parameters!=scalar_parameters:raise A.AbiError('unused pointwise scalar parameter')
    if (not instructions or len(instructions)>24 or len(gb.yield_values)!=1 or
            values.get(gb.yield_values[0])!=12+len(instructions)-1 or
            types.get(gb.yield_values[0])!='f32'):
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
    if cfg.get('kind') == 'typed_c_wrapper':
        from typed_wrapper import contract as typed_contract
        return typed_contract(A, spec, target, ret, args)
    alias_policy=cfg.get('alias_policy','disjoint')
    if alias_policy not in ('disjoint','exact_or_disjoint') or (
            alias_policy=='exact_or_disjoint' and cfg['kind']!='pointwise_reference'):
        raise A.AbiError('exact aliasing requires a validated pointwise reference')
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
    return A.CallAbi(call,target['library'],target['call'],params,[],returns=sa.ret if sa.ret != 'void' else None,specs=[sa.spec],backends=target['backends'],runtime_adapter=cfg)
