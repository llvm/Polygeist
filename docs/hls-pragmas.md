# HLS pragma extraction in cgeist

This document is the single source of truth for how cgeist (the Polygeist
C/C++ front end) turns Vitis HLS `#pragma HLS ...` directives into MLIR
attributes, and for the attribute contract with `hls-est` (`~/circt`).

Contents:

1. [Usage](#1-usage)
2. [How it works](#2-how-it-works)
3. [Attribute conventions](#3-attribute-conventions)
4. [Pragma reference](#4-pragma-reference)
5. [Placement and name resolution](#5-placement-and-name-resolution)
6. [Diagnostics](#6-diagnostics)
7. [Preservation through the pipeline](#7-preservation-through-the-pipeline)
8. [Contract with hls-est](#8-contract-with-hls-est)
9. [Tests](#9-tests)
10. [Known limitations](#10-known-limitations)
- [Appendix A: pre-rewrite state](#appendix-a-pre-rewrite-state)

---

## 1. Usage

```
cgeist kernel.c --function='*' -S --hls-annotate [--raise-scf-to-affine] [-O0..-O3]
```

| Flag | Effect |
|------|--------|
| `--hls-annotate` | Interpret `#pragma HLS` and emit `hls.*` attributes. Also tags local arrays with `polygeist.varname` / `hls.preserve` and functions with `hls.orig_name`. Without it, HLS pragmas are consumed silently (no attributes, no diagnostics) as before. |
| `--hls-pragma-report` | With `--hls-annotate`: after the pass pipeline, print one remark per pragma and outcome: the op(s) it was attached to, or why it was not applied. |
| `--hls-pragma-check-only` | With `--hls-annotate`: parse, place and validate the pragmas of **every** function definition in the translation unit (template instantiations included, `--function` ignored) without generating code, and report the AST-level targets (`loop 'L' in f (check-only)`). A lint mode, and a way to check designs cgeist cannot lower. Implies `--hls-pragma-report`. |

Example report:

```
k.c:12:1: remark: '#pragma HLS pipeline II=1' -> affine.for (loop 'outer') in @kernel
k.c:15:1: remark: '#pragma HLS array_partition variable=buf cyclic factor=4' -> memref.alloca (variable 'buf') in @kernel
k.c:15:1: remark: '#pragma HLS array_partition variable=buf cyclic factor=4' -> memref.alloca (variable 'buf') in @top      <- inlined copy
k.c:20:1: remark: '#pragma HLS unroll' -> not applied: the loop was removed during optimization
```

Errors (malformed or misplaced pragmas) make cgeist exit with status 1 after
printing clang-style diagnostics; warnings do not.

## 2. How it works

Code: `tools/cgeist/Lib/HLSPragma*.{h,cc}`.

| Stage | File | What happens |
|-------|------|--------------|
| Parse (preprocessor) | `HLSPragmaParse.cc` | A catch-all pragma handler (registered without a name at the root namespace) recognises `HLS` in any case, lexes the rest of the pragma **with macro expansion**, and builds a `PragmaRecord`: kind, options, locations, normalized text. Options are checked against the spec table (`HLSPragmaSpec.cc`): known names, flag vs. value, value syntax (integer expression, enum member, variable path, list), duplicates, required options. Non-HLS pragmas without a handler behave exactly as in clang (ignored; `warn_pragma_ignored`). |
| Associate (AST) | `HLSPragmaSema.cc` | When cgeist starts emitting a function body, every pragma located inside that body (excluding nested lambda / local-class bodies) is bound to its target by `SourceLocation`, integer values are evaluated in the pragma's scope, semantic rules are checked, conflicts detected, and the attribute entries built. |
| Emit | `HLSPragmaSema.cc`, `HLSPragma.cc` | Function and parameter targets get their attributes immediately (`func.func` attributes / argument attributes). Globals get them when the translation unit ends. Loops, local variables and `bind_op` operations get a **location tag** (section 7). |
| Materialize (MLIR) | `HLSPragmaMaterialize.cc` | After the pass pipeline (`driver.cc`), every op whose location carries a tag receives the tag's attributes if it is the right kind of op; tags are then stripped from all locations. Targets that disappeared are reported. |

Hooks in existing code (all small): `clang-mlir.h` (handler registration),
`clang-mlir.cc` (function entry, alloca creation, assignments, globals, end
of translation unit), `CGStmt.cc` (loop exit tests / `affine.for`),
`driver.cc` (materialization), `lib/polygeist/Passes/LoopRestructure.cpp`
(keeps tagged exit-test locations on the new `scf.while`).

## 3. Attribute conventions

* Every pragma kind `K` produces the attribute **`hls.K`**: an `ArrayAttr` of
  `DictionaryAttr`, one entry per pragma instance, in source order. Several
  pragmas on one target accumulate (e.g. partitioning two dimensions).
* Dictionary keys are the option names in lower case, except `II` (upper
  case, as hls-est reads it) and `array_partition`/`array_reshape` `type`,
  whose key is `kind` (historical hls-est contract).
* Value types: integers are `i32` `IntegerAttr`; flags are `BoolAttr`; enum
  values are lower-case `StringAttr` (`dependence` `direction` is upper-case:
  `"RAW"`); names are `StringAttr`; lists are `ArrayAttr`; floating
  `performance` targets are `f64` `FloatAttr`.
* Options with a documented default are always present (so consumers never
  guess); options without a default are omitted when not written.
* Variable/port pragmas record the name as written in `variable` (or `port`),
  including member paths and subscripts (`s.x`, `psum[0]`).
* **`off`**: for loop/function pragmas, `K off` produces the unit attribute
  `hls.K_off` instead of `hls.K`; for variable pragmas it produces
  `hls.K_off = [{variable[, dim]}]`.
* Unknown pragmas: `hls.unknown_pragmas = ["<text>", ...]` on the innermost
  enclosing loop, else the function, plus a warning.

Target ops:

| Target | Op / place |
|--------|------------|
| loop | `affine.for`, `scf.for`, `scf.while`, `scf.parallel`, `affine.parallel` (whatever the loop is after the pipeline) |
| function | the `func.func` |
| parameter | argument attribute of the `func.func` (index counts `this` for methods) |
| local variable | its `memref.alloca` / `memref.alloc` / `llvm.alloca` |
| global or static local | its `memref.global` / `llvm.mlir.global` |
| `bind_op` | the arithmetic op assigned to the variable (`arith.muli`, `arith.addf`, `math.sqrt`, ...) |

## 4. Pragma reference

Notation: `int` = integer constant expression (literals, macros, `constexpr`
variables, enumerators, template arguments, calls to `constexpr`
functions); `enum{a,b}` = one of the listed values (case-insensitive);
**bold** = required; *pos* = the enum value may be written without `name=`.

### 4.1 Loop and region pragmas

Placement "loop": innermost enclosing loop body, error otherwise.
"loop/function": innermost enclosing loop body, otherwise the function.

| Pragma | Placement | Options (type, default) | Attribute |
|--------|-----------|-------------------------|-----------|
| `pipeline` | loop/function | `II` int ≥ 1 (default 0 = not specified), `rewind` flag (false), `style` enum{stp,flp,frp} (stp), `enable_flush` flag, `off` | `hls.pipeline = [{II, rewind, style[, enable_flush]}]` / `hls.pipeline_off` |
| `unroll` | loop | `factor` int ≥ 1 (default 0 = complete), `skip_exit_check` flag (false), `off` | `hls.unroll = [{factor, skip_exit_check}]` / `hls.unroll_off` |
| `loop_flatten` | loop | `off` | `hls.loop_flatten = [{}]` / `hls.loop_flatten_off` |
| `loop_merge` | loop/function | `force` flag (false) | `hls.loop_merge = [{force}]` |
| `loop_tripcount` | loop | `min` int ≥ 0 (0), **`max`** int ≥ 0, `avg` int ≥ 0 ((min+max)/2) | `hls.loop_tripcount = [{min, max, avg}]` |
| `dependence` | loop/function | `variable`, `class` enum{array,pointer} *pos*, `type` enum{intra,inter} *pos* (inter), `direction` enum{raw,war,waw} *pos* (emitted upper-case), `distance` int ≥ 1, **`dependent`** flag (*pos* `true`/`false`) | `hls.dependence = [{...}]`, several allowed |
| `dataflow` | loop/function | `disable_start_propagation` flag (false) | `hls.dataflow = [{disable_start_propagation}]` |
| `latency` | loop/function | `min` int ≥ 0, `max` int ≥ 0 | `hls.latency = [{min?, max?}]` |
| `performance` | loop/function | `target_ti` number > 0, `target_tl` number > 0 (at least one), `unit` enum{cycle,sec} (cycle) | `hls.performance = [{target_ti?, target_tl?, unit}]` |

Example (`--raise-scf-to-affine`):

```c
for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II=2
#pragma HLS loop_tripcount max=64
  a[i] = 3 * a[i];
}
```
```mlir
affine.for %arg1 = 0 to 64 {
  ...
} {hls.loop_tripcount = [{avg = 32 : i32, max = 64 : i32, min = 0 : i32}],
   hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
```

`do`-while loops (and loops with data-dependent exits) stay `scf.while`:
`} attributes {hls.pipeline = [...]}`.

### 4.2 Function pragmas

| Pragma | Placement | Options (type, default) | Attribute |
|--------|-----------|-------------------------|-----------|
| `inline` | function | `recursive` flag (false), `off` | `hls.inline = [{recursive}]` / `hls.inline_off` |
| `top` | function | **`name`** name | `hls.top = [{name}]` |
| `function_instantiate` | parameter | **`variable`** (a parameter) | argument attribute `hls.function_instantiate = [{variable}]` |
| `expression_balance` | loop/function | `off` | `hls.expression_balance = [{}]` / `hls.expression_balance_off` |
| `allocation` | loop/function | **`instances`** name list, **`limit`** int ≥ 1, `type` enum{function,operation} *pos* (function) | `hls.allocation = [{instances = [...], limit, type}]`, several allowed |
| `occurrence` | loop/function | **`cycle`** int ≥ 1 | `hls.occurrence = [{cycle}]` |
| `protocol` | loop/function | `mode` enum{fixed,floating} *pos* (floating) | `hls.protocol = [{mode}]` |
| `stable` | parameter, local or global | **`variable`** | `hls.stable = [{variable}]` (argument attribute, alloca, or global) |
| `reset` | global / static local | **`variable`**, `off` | `hls.reset = [{variable}]` / `hls.reset_off = [{variable}]` on the `memref.global` |

```mlir
func.func @vadd(%arg0: memref<?xi32> {hls.interface = [...]}, ...)
    attributes {hls.inline_off, hls.orig_name = "vadd", ...}
```

`inline off` is also honoured: cgeist's inliner (`-O1` and up; it registers
its own func-dialect inliner interface, `cgeist::hls::registerInlinerExtension`)
does not inline a call to a function carrying `hls.inline_off`, so the call
stays a `func.call` and the callee stays a function, as Vitis keeps it as its
own module. Calls inside loops are raised to affine like any other op.
One exception: an array argument that does not fit the shape the parameter
declares (C ignores an array parameter's first dimension, so `f(a)` with
`int a[32][4]` and `void f(int p[8][4])` is legal C) gives a `memref.cast`
from `memref<32x4xi32>` to `memref<8x4xi32>`, which is only valid once
inlining folds it away. Such a call is inlined as before, with a warning
(`'#pragma HLS inline off' of 'f' not honoured at this call: ...`). A row
argument (`f(a[k])`) is passed as a `polygeist.subindex` of the array.
`--hls-ignore-inline-off` restores the earlier behaviour (every call inlined),
for comparisons. Without `--hls-annotate` no pragma is read and every call is
inlined.

Calls Vitis does not inline automatically stay calls too (`--hls-auto-inline`,
on by default with `--hls-annotate`; VITIS_NOTES N18 in the hls-est tree,
`Lib/HLSAutoInline.h`): for a caller C and a callee F without an inline
pragma, Vitis inlines all calls of F in C iff size(F) x calls(F in C) <= 100
(calls counted after `#pragma HLS unroll` and after the complete unroll a
`pipeline` pragma implies for every loop nested in the pipelined loop or
function; size = clang's -O0 IR with the `always_inline` callees inlined, after
`sroa, early-cse, instcombine, simplifycfg, dce`, the pragma loops unrolled
first, without integer casts, plus size(G) - 2 per call of a G inlined into F
first); pointer / reference returns and `always_inline` callees (the Vitis
headers mark every ap_int / ap_fixed member so) are always inlined, and the
calls in an `always_inline` body count, and are kept, in the caller it is
inlined into; dataflow
processes (the calls at the top level of a `dataflow` function; a call inside
one of its loops follows the size rule) and callees holding a dataflow region
always kept. The kept calls and their
callees carry `hls.inline_auto_off` (`{calls, size}` on the call, `{size}` on
the callee). `--hls-auto-inline=false` inlines them as before;
`--hls-auto-inline-threshold`, `--hls-auto-inline-dataflow`,
`--hls-auto-inline-unroll-budget` (a function the unroll would make larger is
not unrolled for measuring; its size and calls are estimated from the trip
counts) and `--hls-auto-inline-report` (every decision on stderr) are the
knobs.

### 4.3 Variable and port pragmas

| Pragma | Target | Options (type, default) | Attribute |
|--------|--------|-------------------------|-----------|
| `array_partition` | variable | **`variable`**, `type` enum{cyclic,block,complete} *pos* (complete; key `kind`), `factor` int ≥ 1 (required for cyclic/block, ignored with a warning for complete), `dim` int ≥ 0 (1; 0 = all dims; ≤ rank), `off` | `hls.array_partition = [{variable, kind, factor?, dim}]` / `hls.array_partition_off = [{variable, dim}]` |
| `array_reshape` | variable | as `array_partition`, plus `object` flag | `hls.array_reshape = [...]` / `hls.array_reshape_off` |
| `bind_storage` | variable | **`variable`**, `type` enum{fifo,ram_1p,ram_1wnr,ram_2p,ram_s2p,ram_t2p,rom_1p,rom_2p,rom_np,xpm_memory (accepted by Vitis 2025.1, kernel w9r22i_c3_xpm)} (optional: Vitis accepts `impl` alone; no default is invented), `impl` enum{auto,bram,bram_ecc,lutram,uram,uram_ecc,srl,memory}, `latency` int ≥ −1 (−1 = auto) | `hls.bind_storage = [{variable, type?, impl?, latency?}]` |
| `resource` (legacy) | variable | **`variable`**, **`core`** name, `latency` int ≥ −1, `ports` name, `metadata` name | a memory core (`RAM_<kind>[_<impl>]`, `ROM_<kind>[_<impl>]`, `FIFO[_<impl>]`) is the bind_storage it names: `hls.bind_storage = [{variable, type, impl?, latency?}]` (`RAM_2P_BRAM` → type ram_2p, impl bram; Vitis 2025.1 gives the same RAM_2P_BRAM_1R1W for both, kernels w9r13p_rs_res / rs_bind, CIRCT VITIS_NOTES N98); an operator core (`Mul_LUT`, `AddSub_DSP`) stays `hls.resource = [{variable, core, ...}]` |
| `bind_op` | operation | **`variable`** (local or parameter), **`op`** enum{add,sub,mul,fadd,fsub,fmul,fdiv,fexp,flog,fsqrt,frsqrt,frecip,dadd,…,hsqrt}, `impl` enum{auto,dsp,fabric,meddsp,fulldsp,maxdsp,primitivedsp,qadder,vivado_dds}, `latency` int ≥ −1 (−1 = auto) | `hls.bind_op = [{variable, op, impl?, latency?}]` on each matching op assigned to the variable |
| `interface` | parameter, or function for `port=return` | **`mode`** enum{ap_none,ap_vld,ap_ack,ap_hs,ap_ovld,ap_fifo,ap_memory,bram,axis,s_axilite,m_axi,ap_ctrl_none,ap_ctrl_hs,ap_ctrl_chain,ap_stable} *pos*, **`port`**, `bundle` name, `offset` int or enum{off,direct,slave}, `depth`, `latency`, `max_widen_bitwidth`, `max_read_burst_length`, `max_write_burst_length`, `num_read_outstanding`, `num_write_outstanding`, `channel`, `interrupt` (ints), `register` flag, `register_mode` enum{both,forward,reverse,off}, `storage_type`, `storage_impl` enums, `clock`, `name` names | `hls.interface = [{mode, port, ...}]` |
| `stream` | variable | **`variable`**, `depth` int ≥ 1 (2), `type` enum{fifo,pipo,shared,unsync} (fifo), `off` | `hls.stream = [{variable, depth, type}]` / `hls.stream_off = [{variable}]` |
| `aggregate` | variable (also `variable=this` inside an instance member function: the function's object argument) | **`variable`**, `compact` enum{bit,byte,none,auto} (auto) | `hls.aggregate = [{variable, compact}]`; on a parameter it is also carried to the variable every call passes to it (an alloca, memref.global or the caller's parameter, transitively, before cgeist's inliner) with `from = <callee>` (`--hls-propagate-aggregate`, on by default; Vitis aggregates the actual argument, hls-est VITIS_NOTES N37 S2 (b), (c)) |
| `disaggregate` | variable | **`variable`** | `hls.disaggregate = [{variable}]` |
| `alias` | function | **`ports`** name list (≥ 2 parameters), exactly one of `distance` int or `offset` int list (one per port) | `hls.alias = [{ports, distance | offset}]` on the function |
| `cache` | parameter | **`port`**, `lines` int ≥ 1, `depth` int ≥ 1 | argument attribute `hls.cache = [{port, lines?, depth?}]` |

Examples:

```c
int buf[16][8];
#pragma HLS array_partition variable=buf cyclic factor=4 dim=2
#pragma HLS array_partition variable=buf complete dim=1
```
```mlir
%alloca = memref.alloca() {hls.array_partition = [
    {dim = 2 : i32, factor = 4 : i32, kind = "cyclic", variable = "buf"},
    {dim = 1 : i32, kind = "complete", variable = "buf"}],
  hls.preserve, polygeist.varname = "buf"} : memref<16x8xi32>
```

```c
void f(int a[8][4], int *p) {
#pragma HLS array_partition variable=a complete dim=2
#pragma HLS interface m_axi port=p bundle=gmem offset=slave
#pragma HLS interface s_axilite port=p bundle=control
```
```mlir
func.func @f(%arg0: memref<?x4xi32> {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "a"}]},
             %arg1: memref<?xi32> {hls.interface = [{bundle = "gmem", mode = "m_axi", offset = "slave", port = "p"},
                                                    {bundle = "control", mode = "s_axilite", port = "p"}]})
```

```c
int c = a * b;
#pragma HLS bind_op variable=c op=mul impl=dsp latency=3
```
```mlir
%0 = arith.muli %arg0, %arg1 {hls.bind_op = [{impl = "dsp", latency = 3 : i32, op = "mul", variable = "c"}]} : i32
```

Syntax accepted everywhere: `#pragma HLS` / `#pragma hls` / any case;
`key=value`, `key = value`; `_Pragma("HLS ...")`; pragmas produced by macros
(including the `DO_PRAGMA(x) _Pragma(#x)` idiom); option values from
macros. Vivado positional spellings are accepted where listed (*pos*), e.g.
`#pragma HLS array_partition variable=a cyclic factor=2`,
`#pragma HLS interface m_axi port=x`, `#pragma HLS allocation function instances=f limit=1`,
`#pragma HLS dependence variable=a inter RAW false`.

## 5. Placement and name resolution

* A pragma belongs to the innermost function body containing it (a pragma
  inside a lambda belongs to the lambda). Outside any function body it is an
  error. Pragmas in functions that are never emitted (unused static
  functions, uninstantiated templates) are neither applied nor validated;
  the report says so.
* **Loops.** A loop's pragma region is its body, plus the space between the
  header's `)` and a brace-less body; for `do`-while, between `do` and
  `while`. The innermost loop whose region contains the pragma wins, so a
  pragma after other statements, inside an `if` in the body, or in a
  macro-generated loop, binds correctly. Loop kinds: `for`, `while`,
  `do`-while, range-based `for`. A `pipeline` placed right before a loop
  (outside it) applies to the function and warns.
* **Templates.** Each instantiation is associated separately, with template
  arguments usable in option values (`factor=F`).
* **Variables** (`variable=`, `port=`, `ports=`), looked up at the pragma's
  location:
  1. locals visible at the pragma (innermost scope first; C shadowing rules);
     if none, a local declared *later* in an enclosing scope (Vitis accepts
     pragmas before declarations);
  2. parameters of the function;
  3. globals and namespace-scope variables (through enclosing namespaces and
     `using namespace`), static locals;
  4. a local declared in a nested scope that does *not* contain the pragma
     (e.g. a buffer inside a loop, pragma at the top of the function), if it
     is the only declaration of that name in the function — with a warning;
     several such declarations are an ambiguity error.
  5. inside an instance member function (a constructor), a data member of the
     function's class or of one of its bases: `variable=m_Val` is
     `this->m_Val`, as Vitis parses the variable as an expression of the
     function. An `array_partition` / `array_reshape` is recorded on the
     `this` argument with the member path (`variable = "this.m_Val"`,
     `member_path`) and carried to the object when the function is inlined;
     any other pragma on a member is reported (warning) and dropped (CIRCT
     OPEN_ISSUES O219 C1: Vitis_Libraries blas `WindowRm::m_Val`, security
     aes `key_list`).
  A class data member named outside a member function is an error. A leading
  `&` (`variable=&s`) is dropped with Vitis' warning (HLS 207-5544). Paths: `s.a`, `p->a`,
  `a[2]`, `s.rows[1]`; members and subscripts are type-checked (index range
  for constant-size arrays). The whole variable's storage is annotated and
  the path recorded.
* **Integer values** are evaluated in the same scope: literals (dec/hex/oct/
  bin), `+ - * / % << >> & | ^ ~` and parentheses, macros, `const`/
  `constexpr` variables, enumerators, qualified names (`cfg::kII`), template
  arguments, and calls to `constexpr` functions or already-instantiated
  `constexpr` function templates (`ceildiv(N, P)`). Names follow C++ lookup:
  a `using Base<C>::B;` declaration names the base's member, a member is also
  found in the base classes, the first component may be a type template
  parameter (`CholeskyTraits::INNER_II`) or a typedef of a class, and a static
  data member's initializer is evaluated in the class template
  specialization (CIRCT OPEN_ISSUES O219 C2). An enumerator of an unscoped
  enum declared in the function body (`enum { fifo_depth = burst_len * 2 };`)
  is found when no other lookup finds the name.
* **Dead arms.** A pragma in the arm of an `if` whose condition is a constant
  (after template substitution) that the arm is not taken in is not emitted,
  as clang's codegen skips a constant dead arm: `if (USE_URAM) { bind_storage
  ... impl=URAM } else { bind_storage ... impl=BRAM }` applies one binding
  (O219 C4).

## 6. Diagnostics

All diagnostics are clang-style with file:line:col. Errors stop cgeist
(exit status 1) before code generation.

| Situation | Severity | Message (abridged) |
|-----------|----------|--------------------|
| `#pragma HLS` with no name / non-identifier | error | expected a pragma name after '#pragma HLS' |
| unknown pragma | warning | unknown HLS pragma 'X'; it is preserved in 'hls.unknown_pragmas' |
| unknown option | warning | unknown option 'X' in '#pragma HLS K'; ignored |
| bad option syntax | error | expected an option name ... / expected a value after 'X=' / option 'X' requires a value |
| bad value | error | invalid value 'V' for option 'X' in '#pragma HLS K': expected ... |
| option repeated | error | option 'X' is specified more than once (+ note) |
| missing required option | error | '#pragma HLS K' requires option 'X' (`loop_tripcount` without `max`: warning, the pragma is ignored, as Vitis ignores it: kernel w9r22i_c5_tripcount_nomax) |
| value not constant | error | cannot resolve 'N' to an integer constant / invalid integer expression |
| value out of range | error | option 'X' ... must be >= N / is out of range |
| outside a function | error | '#pragma HLS K' must be placed inside a function body |
| loop pragma outside a loop | error | ... must be placed inside a loop body (+ note if a loop follows) |
| `pipeline` right before a loop | warning | ... applies to the enclosing function ... |
| unknown variable | error | use of undeclared variable 'x' in '#pragma HLS K' |
| variable only in an unrelated nested scope | warning / error | 'x' is declared in a nested scope that does not contain ...; using that declaration / ... is ambiguous |
| bad member / subscript | error | no member named ... / subscripted value ... / index N is out of range ... |
| class data member | error / warning | ... names a class member ... (outside a member function) / ... names a data member of the object; the pragma is ignored (a pragma other than array_partition / array_reshape on a member) |
| `variable=&x` | warning | extra token before variable expression is ignored (HLS 207-5544) |
| array_partition on a scalar of class type (an `ap_uint`) | warning | ... on the scalar 'x' of type 'T' is ignored, as Vitis ignores it (kernel w9r22i_c7) |
| interface on a local variable | warning | 'x' in '#pragma HLS interface' is not a function parameter; the pragma is ignored (HLS 214-238, kernel w9r22i_c8); on a global it stays an error |
| wrong kind of variable | error | requires an array or pointer / is not a function parameter / requires a static or global variable / requires a local variable or parameter |
| partition semantics | error / warning | type=cyclic requires option 'factor'; dim=N is out of range ...; (w) 'factor' is ignored for complete partitioning; (w) factor=F exceeds the size S of dimension D |
| other semantics | error | min exceeds max; avg not within [min, max]; performance requires target_ti or target_tl; 'recursive' conflicts with 'off'; alias port count / distance-or-offset; interface mode/port=return mismatch |
| `off` plus other options | warning | option 'X' is ignored because '#pragma HLS K' has 'off' |
| conflicting pragmas on one target | error | '#pragma HLS K' conflicts with an earlier pragma on the same T (+ note) |
| two `array_partition` on one dimension that Vitis merges (same type: larger factor; `complete` over cyclic/block) | remark | '#pragma HLS array_partition' (cyclic factor=2) on dimension 1 of variable 'x' is dropped: the later pragma (cyclic factor=4) partitions it more finely, which Vitis keeps (+ note) |
| identical duplicate | warning | duplicate '#pragma HLS K' on the same T is ignored (+ note) |
| misplaced `dataflow` | warning | '#pragma HLS dataflow' is ignored: only a 'for' loop body or a function body supports the dataflow pragma (as Vitis, HLS 207-5576); e.g. in a nested `{ }` block or a `while` body |
| target removed by optimization | warning | the loop it applies to was removed ... / variable 'x' was promoted to registers or removed ... / the operation it binds was folded away ... |
| `bind_op` without a matching operation | warning | no matching operation is assigned to variable 'x' |
| global never emitted | warning | global 'g' is not used by any emitted function |

Conflict keys: single-instance pragmas (pipeline, unroll, loop_flatten,
loop_merge, loop_tripcount, dataflow, latency, performance, inline, top,
expression_balance, occurrence, protocol) conflict per target, `off`
included; array_partition / array_reshape per (variable, dim) with `dim=0`
overlapping every dim, except that two array_partition pragmas Vitis merges
are merged (the same type keeps the larger factor, `complete` wins over
cyclic or block, a `dim=0` entry is split per dimension first; cyclic against
block stays an error, as in Vitis: [HLS 214-290]; VITIS_NOTES N17 (c) / N28
in the hls-est tree; `--hls-reject-partition-conflicts` restores the error); bind_storage, stream, stable, reset,
function_instantiate, cache per variable; aggregate vs. disaggregate per
variable; bind_op per (variable, op); interface per (port, mode) — an m_axi
and an s_axilite interface on one port are the usual Vitis pattern;
dependence, allocation, alias and unknown pragmas only diagnose exact
duplicates.

## 7. Preservation through the pipeline

cgeist emits most loops as an unstructured CFG; `LoopRestructure`,
`CanonicalizeFor` and `RaiseSCFToAffine` later turn them into `scf.while`,
`scf.for` and `affine.for`, and the inliner, mem2reg, LICM, alloca
hoisting and canonicalization rewrite ops freely. Attributes set at codegen
would not survive that (and loops do not exist yet), so:

* The op that identifies the target — the loop's **exit-test `cf.cond_br`**
  (for/while/range-for/do-while) or the `affine.for` of a scop, the
  **alloca** of a local, the **arithmetic op** of a `bind_op` — gets a
  location `FusedLoc<{hls.tag = N}>[original]`.
* Every rewrite in the pipeline builds replacement ops with the old op's
  location, and inlining wraps callee locations in `CallSiteLoc`, so the tag
  follows the target, including every inlined copy.
* `LoopRestructure` is the one place where a new op takes a location from
  somewhere else (the first op of the loop header). It now fuses the
  metadata-carrying locations of the loop's exiting terminators into the new
  `scf.while`'s location (a local change; without metadata the behaviour is
  unchanged).
* After the pipeline, `cgeist::hls::materialize` gives each op the
  attributes of the tags in its location — only if the op is of the right
  kind (loop tags only on loop ops, etc.) — and strips the tags, so printed
  debug locations are unchanged.
* A tagged target that no longer exists anywhere (dead loop, fully unrolled
  loop, scalar promoted to a register, folded operation) gets a warning
  unless its whole function was removed.

`--hls-keep-invariant-loops` (default off): cgeist's LICM (`ParallelLICM`)
hoists an inner loop whose bounds and body do not depend on its enclosing
loop out of it (e.g. a reduction over `r` that does not use `j`, inside the
`j` loop). Vitis makes its pipeline / flatten decisions on the source loop
structure (OPEN_ISSUES O5 / VITIS_NOTES N14 in the hls-est tree), so with the
flag LICM does not hoist an op that is or contains a loop; other invariant
ops are still hoisted, also out of the kept loop. It needs `--parallel-licm`
(the default).

Verified configurations (tests): `-O0`..`-O3`, `--raise-scf-to-affine`,
`--memref-fullrank`, inlining into callers (`-O1` and up), alloca hoisting
out of loops, partial unrolling and canonicalization with `polygeist-opt`
afterwards.

## 8. Contract with hls-est

hls-est (`~/circt/lib/Dialect/Resource`) reads:

| Attribute | Reader | Shape |
|-----------|--------|-------|
| `hls.array_partition` | `MemoryInterface.cpp` `parsePartitionSpec`, `MemoryDS.cpp` `getPartitionSpecs`, `InferVitisSplitArray.cpp` | `[{kind: "cyclic"\|"block"\|"complete", factor?: i32, dim: i32 (1-based, 0 = all), variable: str}]` on `memref.alloca`/`memref.alloc`/`memref.global` or as a `func.func` argument attribute |
| `hls.pipeline` | `InferVitisSplitArray.cpp` | `[{II: i32, ...}]` on `affine.for`; `II > 0` is the user target, `0` = not specified |
| `hls.pipeline_off` | `InferVitisSplitArray.cpp` | unit attribute on `affine.for` |
| `polygeist.varname` | `InferVitisSplitArray.cpp` | `StringAttr` on the defining op of a memref |
| `hls.inline_off`, `hls.inline_auto_off` | `HlsInliner.cpp` (`--inline` keeps the calls), `CallInstances.cpp`, `MemrefBankClassification.cpp`, `BRAMAnalysis.cpp` (debug) | unit (`inline off`) / `{size: i64}` on the callee `func.func`, `{calls: i64, size: i64}` on each kept `func.call` (automatic inlining, section 4.2) |
| `hls.preserve` | Polygeist canonicalizer (`lib/polygeist/Ops.cpp`) | unit attribute on multi-element local `memref.alloca` |

`hls.port_pressure`, `hls.debug_unroll` and the `hls.store_enable` op
belong to hls-est itself.

### Breaking changes for hls-est

**None.** No attribute name or shape that hls-est reads changed. The
following output differences are behaviour fixes within the existing
contract, listed because they change hls-est results on the same input:

| Change | Before | After | Effect in hls-est |
|--------|--------|-------|-------------------|
| `pipeline off` | `hls.pipeline = [{II = 0, ...}]` (the loop was marked pipelined) | `hls.pipeline_off` | infer-vitis-split-array no longer treats such loops as pipelined |
| `array_partition` without `dim` | entry without `dim` | `dim = 1` always present | `getPartitionSpecs` (MemoryDS.cpp) no longer skips these entries |
| misplaced pragmas | attached by line-number heuristics to whatever op was nearest | attached to the right op, or a diagnostic | partitions/pipelines land where the source says |
| `pipeline` without `II` | `II = 0` | `II = 0` (unchanged, kept for compatibility) | none |
| no inline pragma, callee too large for Vitis's automatic inlining (N18) or a dataflow process | the callee was inlined into every caller | the calls stay `func.call` with `hls.inline_auto_off`, the callee a function (`--hls-auto-inline=false`: as before) | as for `inline off` |
| `inline off` | the callee was inlined into every caller (`-O1` and up) | the call stays a `func.call`, the callee a function with `hls.inline_off` (`--hls-ignore-inline-off`: as before) | the callee's arrays and loops are in their own function; hls-est sees calls |

Unchanged but restated: `factor` is omitted for `complete`;
`polygeist.varname` and `hls.preserve` are produced exactly as before.

Attributes not read by hls-est whose shape changed: `hls.allocation`
`instances` is now a list (`["foo<int>"]`); `hls.bind_op` is attached to
the matching arithmetic op and records `variable` (the old
`polygeist.ssa_names` helper attribute is gone).

## 9. Tests

`tools/cgeist/Test/HLS/`, run with `ninja check-cgeist` or
`llvm-lit tools/cgeist/Test/HLS`.

| Directory | Contents |
|-----------|----------|
| `parse/` | case/spelling/whitespace, `_Pragma`, macro-built pragmas and values, C++ constants, unknown pragmas/options, syntax and value errors, `--hls-annotate` off |
| one directory per pragma | options alone and together, defaults, placement, lowering (`-O0..-O3`, affine raising, inlining, unrolling), negative cases |
| `placement/` | C++ constructs, pragmas outside functions, macro-generated loops, targets removed by optimization |
| `consumer/` | exact hls-est attribute shapes (always runs); `hls-est-*.c` pipe cgeist output through hls-est (**REQUIRES: hls-est**) |
| `polybench/` | bicg, gesummv, 2mm, 3mm, mvt, syr2k, syrk from HLSFactory with their Tcl directives as pragmas; `generate.py` derives pragmas and expectations from the Tcl; `hls-est-e2e.c` runs all through the hls-est estimation pipeline |
| `hlsfactory/` | every HLSFactory design with synthesis sources, one `.test` each (see below) |
| `report/` | `--hls-pragma-report` |
| `regressions/` | one test per bug found while writing the tests |
| top level | the pre-rewrite tests, updated to the new contract |

hls-est is found through `HLS_EST`, `PATH`, or `~/circt/build/{clion/,}bin`
(`lit.local.cfg`); without it the `REQUIRES: hls-est` tests are reported
UNSUPPORTED.

### HLSFactory corpus

`hlsfactory/generate.py` writes one test per design of the HLSFactory_v2
catalogue (`D:\HLSFactory_v2\HLSFactory`, HLSFactory 1.1.0), read through the
hls-est harness's loader (`designs.load_hlsfactory_v2()` in
`~/circt/tools/hls-est-validate`), so the tests and the harness agree on every
design's key, top function, sources and include directories. It copies each
design's synthesis sources and headers into `hlsfactory/Inputs`, instantiating
HLSyn design-space placeholders (`auto{__PARA__Lk}` → 2, `auto{__TILE__Lk}` →
1, `pipeline auto{__PIPE__Lk}` → `pipeline`), and runs cgeist with the
harness's command for its default shim: the stand-in headers below with their
`cgeist_flags` (`%vitis_shim_flags`), the macros and `-std=` Vitis' front end
adds (`%vitis_frontend_cpp` / `_c99` / `_mixed`), `--raise-scf-to-affine
--memref-fullrank`. It chooses the strongest mode that works:

| Mode | Checks |
|------|--------|
| full | the MLIR (`%hlsfactory_mlir`): the top function is emitted, with its argument types, the types of the array allocations in its body and of the module's globals; and every `--hls-pragma-report` line in order, nothing extra |
| errors | the design's own pragmas are invalid (e.g. a partition of an undeclared variable); the exact diagnostics |
| check-only | cgeist cannot lower the design for reasons unrelated to pragmas (quoted in the test) or takes over 3 minutes; `--hls-pragma-check-only` placement of every pragma |
| skip | nothing can be checked (clang cannot parse the design with the stand-in headers, or the top function is not emitted); the test is UNSUPPORTED and quotes the reason |

`hlsfactory/SUMMARY.md` lists the modes, the outcome counts, every design
not compiled end to end with its reason, and every pragma not applied.
Expectations are the tool's output at generation time (regression tests
over real code); the non-applied outcomes in SUMMARY.md were reviewed when
generated, and the bugs they exposed have their own tests in
`regressions/hlsfactory-*`.

`Inputs/vitis/` holds stand-ins for the Vitis headers (`ap_int.h`,
`ap_fixed.h`, `hls_stream.h`, `hls_math.h`, `hls_vector.h`, `hls_half.h`,
`ap_axi_sdata.h`, `ap_cint.h`) so HLS C++ can be parsed: a copy of the
hls-est harness's bit-exact `exactclass` shim (see `Inputs/vitis/README.md`).
`ap_int<W>` / `ap_fixed<...>` are classes around a `_BitInt(W)`, which cgeist
lowers to `iW` with `--scalar-wrapper-abi`; every RUN line that uses
`%vitis_include` passes it. Stream operations are external calls.

## 10. Known limitations

* `goto` is an error ("'goto' is not supported by cgeist (its code would be
  wrong)"): its code was silently wrong (a forward goto out of an `if`
  dropped, a backward one running a body once, the code between a goto and
  its label executed; CIRCT OPEN_ISSUES O27, held-out chstone dfadd / dfsin).
  Test `regressions/goto-unsupported.c`.
* `inline` (on) only annotates the function; cgeist's inliner (at `-O1` and
  up) inlines every other call anyway, so it needs nothing more. Functions
  without a pragma are inlined too, including the ones Vitis keeps as their
  own module on its own (large callees, VITIS_NOTES N14 R8 in the hls-est
  tree).
* Scalars are promoted to registers by mem2reg; variable pragmas on scalars
  (e.g. `stream` on an `int`) are reported as removed.
* Class data members cannot be named by `variable=`.
* Pragmas in functions that are not emitted are not validated (use
  `--hls-pragma-check-only` to check every function body).
* Arrays of class objects passed as parameters (e.g. `hls::stream<T> s[N]`
  arguments) hit a pre-existing cgeist code generation bug (a `func.call`
  operand type mismatch), independently of pragmas; many stream-based
  HLSFactory designs are therefore tested in check-only mode.
* With `-immediate` (no pipeline) loops are still a CFG: loop pragmas are
  not attached (no warning in that mode).
* Constexpr function calls in option values need an existing definition or
  instantiation of the function (as when it is used elsewhere in the code).
* Pre-existing cgeist limitations unrelated to pragmas show up on some
  real designs (e.g. copying whole struct array elements asserts in
  `ValueCategory.cc`); the HLSFactory tests record these as expected
  failures with the reason.

---

## Appendix A: pre-rewrite state

What existed before this rewrite and why it was replaced.

### A.1 Code inventory

| File | Role |
|------|------|
| `tools/cgeist/Lib/pragmaHandlerHLS.{h,cc}` | `PragmaHLSHandler`, registered for the exact namespace `HLS`. Stored a name, the *raw source text* of the options and the **line number** of the pragma into the global `g_HLSMetadata`. |
| `tools/cgeist/Lib/HandleHLS.{h,cc}` | `ResolveScope` + `AttachMetadata`, run by `driver.cc` **after** the whole MLIR pass pipeline. Handled 7 pragmas: `array_partition`, `array_reshape`, `bind_storage`, `bind_op`, `allocation`, `unroll`, `pipeline`. |
| `tools/cgeist/Lib/clang-mlir.cc` | Under `--hls-annotate`: tagged local allocas with `polygeist.varname` (multi-element arrays also `hls.preserve`), values stored by `x = ...` with `polygeist.ssa_names`, and every `func.func` with `hls.orig_name`. |
| `tools/cgeist/Lib/utils.h` | Unused `g_HLSpragmaAnnotate` global. |
| `lib/polygeist/Ops.cpp` | Canonicalizer keeps allocas that carry `hls.preserve`. |
| `tools/cgeist/Test/HLS/*.cpp` | 7 FileCheck tests (one CHECK line each; two were already failing) + `HLSAllocation.cpp` (no RUN line). |

### A.2 How it worked

1. The preprocessor handler recorded `(name, raw text, line)`.
2. After *all* optimization passes, `ResolveScope` collected every op with a
   region, computed a `[first line, last line]` span from the
   `FileLineColLoc`s of the op and its nested ops, and assigned each pragma
   to the op with the smallest span containing the pragma's line.
3. `AttachMetadata` split the raw text on whitespace and built a
   `DictionaryAttr`, appended to an `ArrayAttr` on the chosen op (or on an
   alloca found by name inside it).

### A.3 Defects (confirmed by running the old binary)

Placement (the root problem: matching by line number after optimization):

- Line-span matching is not structural. Loops created by `LoopRestructure`
  take the location of the first op of the loop header (usually the loop
  *condition*), and spans shift as passes hoist, sink, fold and inline, so a
  pragma could land on the wrong loop, on an `scf.if`, or on the function.
- Two loops on one line were indistinguishable; pragmas before the first
  statement of a function or between functions attached to the next
  function; after inlining, `CallSiteLoc` lines pointed at the call site;
  template instantiations shared lines.

Parsing:

- Only the exact spelling `HLS` was recognised: `#pragma hls ...` was
  silently dropped (confirmed).
- Values came from the raw source text: macros were not expanded, `_Pragma`
  gave scratch-buffer text; keys were case-sensitive; `key = value` was
  mis-split; a failed integer parse produced `-1` as a real value.

Semantics / emission:

- `pipeline off` was emitted as a pipeline (`II = 0`) — confirmed.
- `array_partition` without `dim` omitted `dim`, which hls-est's
  `getPartitionSpecs` skips.
- `unroll off` was ignored; nothing was validated; 20+ Vitis pragmas were
  ignored without a diagnostic.

Robustness:

- An unknown variable in `array_partition` **segfaulted** cgeist
  (confirmed), as did a macro-valued option (`factor=F`, confirmed): null
  `AllocaOp` dereferences in `HandleArrayPartition` & co.
- `HandleBindOp` dereferenced a null op when no `polygeist.ssa_names`
  matched; `HandleAllocation` silently dropped pragmas missing options;
  parameters and globals were never considered as targets.
