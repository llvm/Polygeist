"""Numerical checking for the lib_v2 test harness.

Two test paths, both compared against the application's own C source compiled
by clang (the reference, symbol renamed to ref_<fn>):

  lowered : match_v2 (Linalg level, --no-region-match --inject-defns) ->
            polygeist-opt --lower-kernel-launch (inlines the validated netlib
            defn bodies) -> e2e_v2.lower() (test-only legaliser + LLVM) ->
            linked through a generated memref-descriptor wrapper.
            Residual parts of the function stay as raised code, so the whole
            function is checked.  A residual control (same lowering without
            matching) is built on failure to separate pipeline problems from
            matcher problems.
  region  : the region matcher's emitted launch block (memref.copy /
            memref.dim / kernel.launch) is interpreted and the *spec C source*
            (generic_solver/lib_v2/ref_c, manifest defines) is called exactly as
            the launch prescribes (the numeric_check.py approach; the raised
            body of some region specs is known to be unsound).  Valid only when
            the launch block covers the whole function; otherwise reported as
            NOT_CHECKABLE.

Tolerance (FP64): for every array argument (inputs included, plus 64 guard
elements past its end) the normwise error
        max_i |test_i - ref_i| / max(max_i |ref_i|, 1e-300)
must be <= RTOL = 1e-12, and NaN/Inf positions must agree.  This is ~4500 ulp
of the largest element; the observed errors of correct matches are ~1e-16
(re-association of a length-k reduction gives at most ~k*eps).  Bit-exactness
and the max ulp distance are recorded as well.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import common as C

RTOL = 1e-12
SEEDS = [1, 2, 3, 4, 5]
GUARD = 64
LLVM_BIN = C.REPO / "llvm-project/build/bin"
CLANG = str(LLVM_BIN / "clang")
MLIR_LIBDIR = C.REPO / "llvm-project/build/lib"
CFLAGS_NUM = ["-O1", "-w", "-ffp-contract=off", "-std=gnu11"]


# ---------------------------------------------------------------------------
# C prototypes
# ---------------------------------------------------------------------------

def preprocess(src: Path, cflags: list[str]) -> str:
    fl = [f for f in cflags if not f.startswith("-Dstatic")]
    return subprocess.run(["gcc", "-E", "-P", *fl, "-Dstatic=", str(src)], capture_output=True,
                          text=True, check=True).stdout


def c_params(text: str, fn: str, extents: dict | None = None) -> list[tuple[str, str, list[str]]]:
    """[(kind, name, dims)] with kind in int|double|array."""
    m = re.search(r"\b" + re.escape(fn) + r"\s*\(([^)]*)\)\s*\{", text, re.S)
    if not m:
        raise ValueError(f"prototype of {fn} not found")
    out = []
    for p in m.group(1).split(","):
        p = " ".join(p.replace("restrict", " ").split())
        am = re.match(r"(?:const\s+)?(double|float|int|long)\s*(\*?)\s*(\w+)\s*((?:\[[^\]]*\])*)$", p)
        if not am:
            raise ValueError(f"cannot parse parameter {p!r}")
        ty, star, name, dims = am.group(1), am.group(2), am.group(3), re.findall(r"\[([^\]]*)\]", am.group(4))
        if star:
            if not extents or name not in extents:
                raise ValueError(f"pointer parameter {name} needs an //@ extents entry")
            dims = [extents[name]]
        if dims and not dims[0].strip():
            if not extents or name not in extents:
                raise ValueError(f"unsized leading dimension of {name}")
            dims[0] = extents[name]
        kind = "array" if dims else ("int" if ty in ("int", "long") else "double")
        out.append((kind, name, [d.strip() for d in dims]))
    return out


def parse_extents(meta: dict) -> dict:
    ext = {}
    for item in meta.get("extents", "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            ext[k.strip()] = v.strip()
    return ext


def mlir_arg_types(mlir_text: str, fn: str) -> list[str]:
    m = re.search(r"func\.func\s+@" + re.escape(fn) + r"\((.*?)\)\s*(?:->|attributes|\{)", mlir_text, re.S)
    if not m:
        raise ValueError(f"func @{fn} not in MLIR")
    import mlir_lite as ML
    return [a.split(":", 1)[1].strip() for a in ML.split_top(m.group(1)) if ":" in a]


# ---------------------------------------------------------------------------
# Generated C
# ---------------------------------------------------------------------------

def wrapper_c(fn: str, params, arg_types: list[str]) -> str:
    if len(params) != len(arg_types):
        raise ValueError(f"{fn}: {len(params)} C params vs {len(arg_types)} MLIR args")
    decl, call, sig = [], [], []
    for (kind, name, dims), ty in zip(params, arg_types):
        if ty.startswith("memref<"):
            if kind != "array":
                raise ValueError(f"{name}: memref arg for scalar param")
            shape = ty[len("memref<"):].split(",")[0].rstrip(">").split("x")[:-1]
            r = len(shape)
            if r == len(dims):
                sizes = [f"(long)({d})" for d in dims]
            elif r == 1:
                sizes = ["(long)(" + ")*(".join(dims) + ")"]
            else:
                raise ValueError(f"{name}: rank {r} vs C dims {dims}")
            strides = ["*".join(sizes[i + 1:]) or "1" for i in range(r)]
            decl += ["double *", "double *", "long"] + ["long"] * (2 * r)
            call += [name, name, "0", *sizes, *strides]
            sig.append(f"double *{name}")
        elif ty == "f64":
            decl.append("double"); call.append(name); sig.append(f"double {name}")
        elif ty in ("i32",):
            decl.append("int"); call.append(name); sig.append(f"int {name}")
        elif ty in ("i64", "index"):
            decl.append("long"); call.append(f"(long){name}"); sig.append(f"int {name}")
        else:
            raise ValueError(f"unsupported MLIR arg type {ty}")
    # the wrapper body needs the int params in scope with their C names
    return ("extern void %s_impl(%s);\nvoid %s(%s) {\n  %s_impl(%s);\n}\n" %
            (fn, ", ".join(decl), fn, ", ".join(sig), fn, ", ".join(call)))


def spec_tu(spec: str) -> str:
    man = entry = None
    for mp in (C.SPEC_DIR / "manifest").glob("*.json"):
        d = json.loads(mp.read_text())
        for s in d.get("specs", []):
            if s["name"] == spec:
                man, entry = d, s
    if man is None:
        raise ValueError(f"spec {spec} not in manifests")
    lines = [f"#define NETLIB_SPEC {spec}"]
    for k, v in entry.get("defines", {}).items():
        lines.append(f"#define {k} {v}")
    lines.append(f'#include "{C.SPEC_DIR / "ref_c" / (man["routine"] + ".c")}"')
    return "\n".join(lines) + "\n"


def region_call(block: str, params) -> tuple[list[str], list[str]]:
    """Interpret a region launch block -> (ordered C statements, launched specs).

    Understands the launch idioms the region matcher has emitted so far:
    arith.constant, memref.dim %argK, memref.extract_strided_metadata %argK
    (result #i: 1 offset, 2.. sizes, 2+r.. strides of a row-major array),
    arith.index_cast / muli / addi / subi, memref.copy, kernel.launch.
    Anything else -> ValueError (reported as NOT_CHECKABLE)."""
    env: dict[str, str] = {}
    meta: dict[str, int] = {}          # extract_strided_metadata result -> param index
    stmts, specs = [], []               # ordered C statements / launched specs

    def arg(op):
        return params[op if isinstance(op, int) else int(op.strip()[4:])]

    def val(op):
        op = op.strip()
        if op in env:
            return env[op]
        m = re.match(r"(%[\w$.\-]+)#(\d+)$", op)
        if m and m.group(1) in meta:
            k, nm, dm = arg(meta[m.group(1)])
            i, r = int(m.group(2)), len(dm)
            if i == 1:
                return "0"
            if 2 <= i < 2 + r:
                return f"({dm[i - 2]})"
            if 2 + r <= i < 2 + 2 * r:
                j = i - 2 - r
                return "(" + ")*(".join(dm[j + 1:]) + ")" if dm[j + 1:] else "1"
        if op.startswith("%arg") and op[4:].isdigit():
            k, nm, dm = arg(op)
            return f"{nm}_t" if k == "array" else nm
        raise ValueError(f"cannot interpret launch operand {op}")

    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = re.match(r"(%\S+?)(?::\d+)? = (\S+) (.*?)(?: : .*)?$", line)
        if m and not line.startswith("kernel.launch"):
            res, opn, rest = m.group(1), m.group(2), m.group(3)
            ops = [x.strip() for x in rest.split(",")]
            if opn == "arith.constant":
                env[res] = rest.split(":")[0].strip()
            elif opn == "memref.dim":
                k, nm, dm = arg(ops[0])
                env[res] = f"({dm[int(val(ops[1]))]})"
            elif opn == "memref.extract_strided_metadata":
                meta[res] = int(ops[0].strip()[4:])
            elif opn == "arith.index_cast":
                env[res] = val(ops[0].split(" ")[0])
            elif opn in ("arith.muli", "arith.addi", "arith.subi"):
                sym = {"arith.muli": "*", "arith.addi": "+", "arith.subi": "-"}[opn]
                env[res] = f"({val(ops[0])}{sym}{val(ops[1])})"
            else:
                raise ValueError(f"unsupported op in launch block: {opn}")
            continue
        m = re.match(r"memref\.copy (%arg\d+), (%arg\d+)", line)
        if m:
            s, d = int(m.group(1)[4:]), int(m.group(2)[4:])
            _, dn, dd = params[d]
            stmts.append(f"  memcpy({dn}_t, {params[s][1]}_t, sizeof(double)*(size_t)({')*('.join(dd)}));")
            continue
        m = re.search(r"kernel\.launch @([\w$.\-]+)\((.*)\) :", line)
        if m:
            ops = [x.strip() for x in m.group(2).split(",") if x.strip()]
            specs.append(m.group(1))
            stmts.append(f"  {m.group(1)}({', '.join(val(o) for o in ops)});")
            continue
        raise ValueError(f"unsupported line in launch block: {line[:80]}")
    if not specs:
        raise ValueError("no kernel.launch in region block")
    return stmts, specs


def driver_c(fn: str, params, test: str, init: str = "uniform", alias: dict | None = None,
             orig_fn: str | None = None, region=None) -> str:
    """test: 'lowered' (calls <fn>, the descriptor wrapper) or 'region'.
    alias: {param: (base_param, element_offset)} -- runtime aliasing."""
    alias = alias or {}
    ints = [nm for k, nm, _ in params if k == "int"]
    L = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <math.h>",
         "#include <stdint.h>"]
    L.append(f"extern void ref_{fn}();")
    if test == "lowered":
        L.append(f"extern void {fn}();")
    if orig_fn:
        L.append(f"extern void orig_{orig_fn}();")
    if region:
        for s in sorted(set(region[1])):
            L.append(f"extern void {s}();")
    L.append("static unsigned long long S;")
    L.append("static double rnd(void){S=S*6364136223846793005ULL+1442695040888963407ULL;"
             "return (double)(S>>11)/9007199254740992.0;}")
    L.append("static uint64_t ord(double x){uint64_t u;memcpy(&u,&x,8);"
             "return (u>>63)?~u:(u|0x8000000000000000ULL);}")
    L.append("static void cmp(const char*nm,const double*r,const double*t,long n){"
             "double e=0,m=0;long nanm=0,ex=1;uint64_t mu=0;"
             "for(long q=0;q<n;q++){double a=r[q],b=t[q];"
             "if(isnan(a)||isnan(b)||isinf(a)||isinf(b)){if(!(isnan(a)&&isnan(b))&&a!=b)nanm++;"
             "if(memcmp(&a,&b,8))ex=0;continue;}"
             "double d=fabs(a-b);if(d>e)e=d;if(fabs(a)>m)m=fabs(a);if(memcmp(&a,&b,8))ex=0;"
             "uint64_t x=ord(a),y=ord(b);uint64_t u=x>y?x-y:y-x;if(u>mu)mu=u;}"
             "printf(\"{\\\"array\\\":\\\"%s\\\",\\\"max_abs\\\":%.6e,\\\"max_ref\\\":%.6e,"
             "\\\"nan_mismatch\\\":%ld,\\\"exact\\\":%ld,\\\"max_ulp\\\":%llu}\\n\","
             "nm,e,m,nanm,ex,(unsigned long long)mu);}")
    L.append("int main(int argc, char **argv) {")
    L.append("  S = (unsigned long long)atoll(argv[1]) * 0x9E3779B97F4A7C15ULL + 7; rnd();")
    for i, nm in enumerate(ints):
        L.append(f"  int {nm} = atoi(argv[{i + 2}]);")
    for k, nm, _ in params:
        if k == "double":
            L.append(f"  double {nm} = 0.5 + rnd();")
    copies = ["r", "t"] + (["o"] if orig_fn else [])
    for k, nm, dm in params:
        if k != "array" or nm in alias:
            continue
        size = "(long)(" + ")*(".join(dm) + ")"
        L.append(f"  long {nm}_n = {size} + {GUARD};")
        for c in copies:
            L.append(f"  double *{nm}_{c} = malloc(sizeof(double)*{nm}_n);")
        if init == "spd":
            L.append(f"  for (long q=0;q<{nm}_n;q++) {nm}_r[q]=rnd();")
            if len(dm) == 2:
                L.append(f"  if (({dm[0]})==({dm[1]})) {{ long N_=({dm[0]});"
                         f" for(long i=0;i<N_;i++) {nm}_r[i*N_+i]+=N_; }}")
        else:
            L.append(f"  for (long q=0;q<{nm}_n;q++) {nm}_r[q]=2.0*rnd()-1.0;")
        for c in copies[1:]:
            L.append(f"  memcpy({nm}_{c},{nm}_r,sizeof(double)*{nm}_n);")
    for nm, (base, off) in alias.items():
        for c in copies:
            L.append(f"  double *{nm}_{c} = {base}_{c} + ({off});")

    def args(c):
        return ", ".join(f"{nm}_{c}" if k == "array" else nm for k, nm, _ in params)
    L.append(f"  ref_{fn}({args('r')});")
    if test == "lowered":
        L.append(f"  {fn}({args('t')});")
    elif test == "region":
        L += region[0]
    if orig_fn:
        L.append(f"  orig_{orig_fn}({args('o')});")
    for k, nm, dm in params:
        if k != "array" or nm in alias:
            continue
        if test != "none":
            L.append(f"  cmp(\"{nm}\", {nm}_r, {nm}_t, {nm}_n);")
        if orig_fn:
            L.append(f"  cmp(\"orig:{nm}\", {nm}_r, {nm}_o, {nm}_n);")
    L.append("  return 0;\n}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Build + run
# ---------------------------------------------------------------------------

def sh(cmd, log: Path, timeout=600) -> int:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        try:
            return subprocess.run([str(c) for c in cmd], stdout=fh, stderr=fh, timeout=timeout,
                                  env=C.env()).returncode
        except subprocess.TimeoutExpired:
            fh.write("TIMEOUT\n")
            return 124


def size_configs(n_ints: int, square: bool = False) -> dict[str, list[int]]:
    if n_ints == 0:
        return {"fixed": []}
    if square:
        return {"n1": [1] * n_ints, "n2": [2] * n_ints, "n7": [7] * n_ints, "n23": [23] * n_ints}
    ns = [7, 5, 11, 3, 13, 4, 9][:n_ints]
    md = [23, 19, 29, 17, 31, 21, 27][:n_ints]
    return {"n1": [1] * n_ints, "n2": [2] * n_ints, "nonsquare": ns, "medium": md}


def ref_objects(app_src: Path, fn: str, cflags: list[str], polybench: bool, d: Path, log: Path,
                prefix: str = "ref") -> list[Path] | None:
    fl = [f for f in cflags if not f.startswith("-Dstatic")]
    o = d / f"{prefix}.o"
    extra = ["-Dmain=pb_unused_main_" + prefix] if polybench else []
    if sh([CLANG, "-c", *CFLAGS_NUM, *fl, "-Dstatic=", f"-D{fn}={prefix}_{fn}", *extra, app_src,
           "-o", o], log):
        return None
    objs = [o]
    return objs


def lower_matched(inp: Path, d: Path, tag: str, fn: str, log: Path) -> Path | None:
    import e2e_v2 as E  # read-only reuse of the lib_v2 lowering
    if "kernel.launch" in inp.read_text():
        inl = d / f"{tag}_inlined.mlir"
        if sh(["polygeist-opt", "--lower-kernel-launch", inp, "-o", inl], log):
            return None
        inp = inl
    with log.open("a") as fh:
        fh.write(f"# e2e_v2.lower({inp})\n")
    try:
        return E.lower(inp, d, tag, fn, log)
    except Exception as e:  # noqa: BLE001
        with log.open("a") as fh:
            fh.write(f"lower exception {e}\n")
        return None


def build_and_run(d: Path, tag: str, fn: str, params, test: str, ref_objs: list[Path], log: Path,
                  ll: Path | None = None, mlir_types=None, region=None, spec: str | None = None,
                  init: str = "uniform", alias=None, orig=None, configs=None,
                  polybench: bool = False, seeds=SEEDS) -> dict:
    objs = list(ref_objs)
    if test == "none":
        pass
    elif test == "lowered":
        (d / f"{tag}_wrapper.c").write_text(wrapper_c(fn, params, mlir_types))
        o = d / f"{tag}_wrapper.o"
        if sh([CLANG, "-c", *CFLAGS_NUM, d / f"{tag}_wrapper.c", "-o", o], log):
            return {"verdict": "BUILD_FAIL", "why": "wrapper"}
        objs += [o, ll]
    else:
        for s in sorted(set(region[1])):
            (d / f"{tag}_spec_{s}.c").write_text(spec_tu(s))
            o = d / f"{tag}_spec_{s}.o"
            if sh([CLANG, "-c", *CFLAGS_NUM, d / f"{tag}_spec_{s}.c", "-o", o], log):
                return {"verdict": "BUILD_FAIL", "why": f"spec TU {s}"}
            objs.append(o)
    orig_fn = None
    if orig:
        orig_fn = orig["fn"]
        oo = ref_objects(orig["src"], orig_fn, orig["cflags"], orig.get("polybench", False), d, log,
                         prefix="orig")
        if oo is None:
            return {"verdict": "BUILD_FAIL", "why": "original reference"}
        objs += oo
    (d / f"{tag}_driver.c").write_text(driver_c(fn, params, test, init, alias, orig_fn, region))
    exe = d / f"{tag}_exe"
    extra = []
    if polybench or (orig and orig.get("polybench")):
        extra.append(C.PB_UTIL / "polybench.c")
        extra.append(f"-I{C.PB_UTIL}")
    if test == "lowered":
        extra += [f"-L{MLIR_LIBDIR}", f"-Wl,-rpath,{MLIR_LIBDIR}", "-lmlir_c_runner_utils"]
    if sh([CLANG, *CFLAGS_NUM, d / f"{tag}_driver.c", *objs, *extra, "-lm", "-o", exe], log):
        return {"verdict": "BUILD_FAIL", "why": "link"}
    n_ints = sum(1 for k, _, _ in params if k == "int")
    configs = configs or size_configs(n_ints)
    runs, worst, worst_orig = [], 0.0, 0.0
    exact_all, fails, orig_fails = True, [], []
    for cname, vals in configs.items():
        for seed in seeds:
            try:
                p = subprocess.run([str(exe), str(seed), *map(str, vals)], capture_output=True,
                                   text=True, timeout=120)
            except subprocess.TimeoutExpired:
                fails.append({"config": cname, "seed": seed, "why": "timeout"})
                continue
            if p.returncode:
                fails.append({"config": cname, "seed": seed, "why": f"rc={p.returncode}",
                              "stderr": p.stderr[-300:]})
                continue
            for ln in p.stdout.splitlines():
                r = json.loads(ln)
                err = r["max_abs"] / max(r["max_ref"], 1e-300) if r["max_abs"] else 0.0
                ok = err <= RTOL and r["nan_mismatch"] == 0
                if r["array"].startswith("orig:"):
                    worst_orig = max(worst_orig, err)
                    if not ok:
                        orig_fails.append({"config": cname, "seed": seed, **r, "err": err})
                    continue
                worst = max(worst, err)
                exact_all &= bool(r["exact"])
                if not ok:
                    fails.append({"config": cname, "seed": seed, **r, "err": err})
            runs.append({"config": cname, "sizes": vals, "seed": seed})
    res = {"verdict": "PASS" if not fails else "FAIL", "runs": len(runs), "max_err": worst,
           "bit_exact": exact_all and not fails, "fails": fails[:6], "n_fail": len(fails),
           "configs": configs, "seeds": list(seeds), "rtol": RTOL}
    if orig:
        res["orig_check"] = {"verdict": "PASS" if not orig_fails else "FAIL",
                             "max_err": worst_orig, "fails": orig_fails[:4]}
    return res


def _control(mlir: Path, d: Path, tag: str, fn: str, params, ref, log, init, alias, configs,
             polybench, seeds) -> dict:
    ll = lower_matched(mlir, d, tag, fn, log)
    if ll is None:
        return {"verdict": "LOWER_FAIL"}
    types = mlir_arg_types(mlir.read_text(), fn)
    rc = build_and_run(d, tag, fn, params, "lowered", ref, log, ll=ll, mlir_types=types, init=init,
                       alias=alias, configs=configs, polybench=polybench, seeds=seeds)
    return {k: rc.get(k) for k in ("verdict", "max_err", "n_fail", "why")}


def check_lowered(app_src: Path, fn: str, cflags: list[str], matched: Path, d: Path,
                  polybench: bool = False, extents=None, init="uniform", alias=None, orig=None,
                  configs=None, residual: Path | None = None, seeds=SEEDS,
                  always_residual: bool = False) -> dict:
    """Numerically compare the Linalg-level matched + lowered function with the
    clang-compiled application source.  `residual` (the unmatched debuf IR) is
    lowered as a control when the matched build fails the check."""
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    t0 = time.time()
    text = preprocess(app_src, cflags) if polybench else app_src.read_text()
    params = c_params(text, fn, extents)
    ref = ref_objects(app_src, fn, cflags, polybench, d, log)
    if ref is None:
        return {"verdict": "BUILD_FAIL", "why": "reference"}
    ll = lower_matched(matched, d, "v2", fn, log)
    if ll is None:
        res = {"verdict": "LOWER_FAIL", "log": str(log)}
    else:
        types = mlir_arg_types(matched.read_text(), fn)
        res = build_and_run(d, "v2", fn, params, "lowered", ref, log, ll=ll, mlir_types=types,
                            init=init, alias=alias, orig=orig, configs=configs,
                            polybench=polybench, seeds=seeds)
    if residual is not None and (res["verdict"] != "PASS" or always_residual):
        res["residual_control"] = _control(residual, d, "residual", fn, params, ref, log, init,
                                           alias, configs, polybench, seeds)
        if res["residual_control"]["verdict"] != "PASS" and res["verdict"] != "PASS":
            # stage triage: which raising stage first loses the C semantics?
            tri = {}
            for stage, suffix in (("cgeist_affine", ".mlir"), ("memref_linalg", "_linalg.mlir")):
                sp = Path(str(residual).replace("_debuf.mlir", suffix))
                if sp.exists() and sp != residual:
                    tri[stage] = _control(sp, d, f"stage_{stage}", fn, params, ref, log, init,
                                          alias, configs, polybench, seeds)["verdict"]
            tri["debufferized"] = res["residual_control"]["verdict"]
            res["stage_triage"] = tri
            res["first_bad_stage"] = next((s for s in ("cgeist_affine", "memref_linalg", "debufferized")
                                           if tri.get(s) not in (None, "PASS")), None)
    res["seconds"] = round(time.time() - t0, 1)
    res["path"] = "lowered (inlined netlib defn bodies)"
    return res


def region_covers_function(matched_text: str, fn: str, block: dict) -> bool:
    lines = matched_text.splitlines()
    lo, hi = block["line_range"]
    start = next((i for i, l in enumerate(lines) if re.search(r"func\.func\s+@" + re.escape(fn) + r"\(", l)), None)
    if start is None:
        return False
    depth, end = 0, None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and i > start:
            end = i
            break
    body = [l for i, l in enumerate(lines[start + 1:end], start + 1) if not (lo <= i <= hi)]
    # pure tensor/view ops left dead by the splice (to_tensor, submap,
    # extract_slice, ...) do not matter; anything that can write memory does
    effect = re.compile(r"affine\.for|scf\.for|scf\.while|scf\.if|affine\.if|linalg\.generic|"
                        r"memref\.store|affine\.store|func\.call|kernel\.launch|memref\.copy|"
                        r"bufferization\.to_memref|materialize_in_destination|memref\.alloc|"
                        r"llvm\.store|polygeist\.submapInverse.*memref")
    return not any(effect.search(l) for l in body)


def check_region(app_src: Path, fn: str, cflags: list[str], matched: Path, block: dict, d: Path,
                 polybench: bool = False, extents=None, init="spd", alias=None, orig=None,
                 configs=None, seeds=SEEDS) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    t0 = time.time()
    mt = matched.read_text()
    if not region_covers_function(mt, fn, block):
        return {"verdict": "NOT_CHECKABLE", "why": "region launch does not cover the whole function"}
    text = preprocess(app_src, cflags) if polybench else app_src.read_text()
    params = c_params(text, fn, extents)
    ref = ref_objects(app_src, fn, cflags, polybench, d, log)
    if ref is None:
        return {"verdict": "BUILD_FAIL", "why": "reference"}
    try:
        stmts, specs = region_call(block["text"], params)
    except ValueError as e:
        return {"verdict": "NOT_CHECKABLE", "why": str(e)}
    res = build_and_run(d, "region", fn, params, "region", ref, log, region=(stmts, specs),
                        init=init, alias=alias, orig=orig, configs=configs,
                        polybench=polybench, seeds=seeds)
    res["seconds"] = round(time.time() - t0, 1)
    res["path"] = "region launch -> spec C (ref_c) vs application C"
    res["spec_call"] = " ".join(s.strip() for s in stmts)
    return res


def check_result(res: dict, app_src: Path, fn: str, cflags, d: Path, polybench=False,
                 extents=None, alias=None, orig=None, configs=None, residual=None,
                 init_region="spd", init_linalg="uniform", always_residual=False) -> dict:
    """Numerically check whatever a worker result matched (both paths)."""
    out = {}
    od = Path(res["_out_dir"])
    if res.get("linalg_specs"):
        inj = od / "linalg_only_injected.mlir"
        if inj.exists():
            try:
                out["linalg"] = check_lowered(app_src, fn, cflags, inj, d / "linalg", polybench,
                                              extents, init_linalg, alias, orig, configs, residual,
                                              always_residual=always_residual)
            except Exception as e:  # noqa: BLE001
                out["linalg"] = {"verdict": "HARNESS_ERROR", "why": f"{type(e).__name__}: {e}"}
            out["linalg"]["specs"] = res.get("linalg_only_specs", res["linalg_specs"])
        else:
            out["linalg"] = {"verdict": "NO_INJECTED_IR"}
    need_full = False
    mt = od / "matched.mlir"
    for i, blk in enumerate(res.get("region_blocks", [])):
        if not region_covers_function(mt.read_text(), fn, blk):
            need_full = True
            continue
        try:
            out[f"region{i}"] = check_region(app_src, fn, cflags, mt, blk, d / f"region{i}",
                                             polybench, extents, init_region, alias, orig, configs)
        except Exception as e:  # noqa: BLE001
            out[f"region{i}"] = {"verdict": "HARNESS_ERROR", "why": f"{type(e).__name__}: {e}"}
        out[f"region{i}"]["spec"] = blk["spec"]
    if need_full:
        try:
            out["full"] = check_full(app_src, fn, cflags, mt, d / "full", polybench, extents,
                                     init_linalg, alias, configs)
        except Exception as e:  # noqa: BLE001
            out["full"] = {"verdict": "HARNESS_ERROR", "why": f"{type(e).__name__}: {e}"}
    return out


def overall(num: dict) -> str:
    """PASS / FAIL / NOT_CHECKED from check_result output."""
    if not num:
        return "NOT_CHECKED"
    vs = [v.get("verdict") for v in num.values()]
    if any(v == "FAIL" for v in vs):
        return "FAIL"
    if all(v == "PASS" for v in vs):
        return "PASS"
    return "INCONCLUSIVE(" + ",".join(sorted(set(map(str, vs)))) + ")"


def clean_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)


def check_c_equivalence(var_src: Path, var_fn: str, orig_src: Path, orig_fn: str,
                        orig_cflags: list[str], d: Path, extents=None, init="uniform",
                        configs=None, seeds=SEEDS) -> dict:
    """Validity of a hand-written source variant: variant C vs the original
    (PolyBench) C on the same random inputs, same tolerance."""
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    params = c_params(var_src.read_text(), var_fn, extents)
    ref = ref_objects(var_src, var_fn, [], False, d, log)
    if ref is None:
        return {"verdict": "BUILD_FAIL", "why": "variant"}
    r = build_and_run(d, "equiv", var_fn, params, "none", ref, log, init=init,
                      orig={"src": orig_src, "fn": orig_fn, "cflags": orig_cflags,
                            "polybench": True}, configs=configs, seeds=seeds)
    oc = r.get("orig_check", {})
    if r["verdict"] != "PASS":      # build failure or crash
        return {"verdict": r["verdict"], "why": r.get("why"), "fails": r.get("fails")}
    return {"verdict": oc.get("verdict"), "max_err": oc.get("max_err"), "fails": oc.get("fails"),
            "runs": r["runs"]}


# ---------------------------------------------------------------------------
# Hybrid path: the full matched function (Linalg-level + region launches)
# ---------------------------------------------------------------------------

def _expanded_c(types: list[str]) -> tuple[list[str], list[str]]:
    """MLIR arg types -> (C params of the expanded LLVM ABI, C args for the spec call)."""
    params, args = [], []
    for i, ty in enumerate(types):
        ty = ty.strip()
        if ty.startswith("memref<"):
            r = len(ty[len("memref<"):].split(",")[0].rstrip(">").split("x")) - 1
            params += [f"double *a{i}", f"double *b{i}", f"long o{i}"]
            params += [f"long s{i}_{k}" for k in range(r)] + [f"long t{i}_{k}" for k in range(r)]
            args.append(f"(void *)(b{i} + o{i})")
        elif ty == "f64":
            params.append(f"double v{i}"); args.append(f"v{i}")
        elif ty == "i32":
            params.append(f"int v{i}"); args.append(f"v{i}")
        elif ty in ("i64", "index"):
            params.append(f"long v{i}"); args.append(f"v{i}")
        else:
            raise ValueError(f"unsupported region launch operand type {ty}")
    return params, args


def externalize_region_launches(text: str) -> tuple[str, dict[str, list[str]]]:
    """Replace every void (memref-form) kernel.launch by a call to an external
    shim @lv2shim_<spec>; returns the new text and {spec: types}."""
    shims: dict[str, list[str]] = {}

    def rep(m):
        spec, ops, tys = m.group(2), m.group(3), m.group(4)
        types = [x.strip() for x in tys.split(",") if x.strip()]
        if not any(x.startswith("memref<") for x in types):
            return m.group(0)
        shims[spec] = types
        return f"{m.group(1)}func.call @lv2shim_{spec}({ops}) : ({tys}) -> ()"
    text = re.sub(r"^(\s*)kernel\.launch\s+@([\w$.\-]+)\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*\(\)\s*$",
                  rep, text, flags=re.M)
    if shims:
        decl = "".join(f"  func.func private @lv2shim_{s}({', '.join(ty)})\n" for s, ty in shims.items())
        k = text.rstrip().rfind("}")
        text = text[:k] + decl + text[k:]
    return text, shims


def check_full(app_src: Path, fn: str, cflags: list[str], matched: Path, d: Path,
               polybench: bool = False, extents=None, init="uniform", alias=None,
               configs=None, seeds=SEEDS) -> dict:
    """Numerically check the complete match_v2 output (with region launches):
    tensor launches are inlined from kernel_library_v2 defns, memref (region)
    launches call the spec C source through a descriptor shim."""
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    t0 = time.time()
    text, shims = externalize_region_launches(matched.read_text())
    tensor_specs = sorted(set(re.findall(r"kernel\.launch\s+@([\w$.\-]+)", text)))
    if tensor_specs:
        import inject_kernel_library as IKL
        text = IKL.inject(text, C.LIB.read_text(), tensor_specs)
    full = d / "full_matched.mlir"
    full.write_text(text)
    src_text = preprocess(app_src, cflags) if polybench else app_src.read_text()
    params = c_params(src_text, fn, extents)
    ref = ref_objects(app_src, fn, cflags, polybench, d, log)
    if ref is None:
        return {"verdict": "BUILD_FAIL", "why": "reference"}
    ll = lower_matched(full, d, "full", fn, log)
    if ll is None:
        return {"verdict": "LOWER_FAIL", "log": str(log), "seconds": round(time.time() - t0, 1)}
    objs = list(ref)
    for spec, types in shims.items():
        ps, args = _expanded_c(types)
        sc = d / f"shim_{spec}.c"
        sc.write_text(spec_tu(spec) + f"void lv2shim_{spec}({', '.join(ps)}) {{\n"
                      f"  {spec}({', '.join(args)});\n}}\n")
        o = d / f"shim_{spec}.o"
        if sh([CLANG, "-c", *CFLAGS_NUM, sc, "-o", o], log):
            return {"verdict": "BUILD_FAIL", "why": f"shim {spec}"}
        objs.append(o)
    types = mlir_arg_types(full.read_text(), fn)
    res = build_and_run(d, "full", fn, params, "lowered", objs, log, ll=ll, mlir_types=types,
                        init=init, alias=alias, configs=configs, polybench=polybench, seeds=seeds)
    res.update(seconds=round(time.time() - t0, 1),
               path="hybrid: full matched function; tensor launches inlined from netlib defns, "
                    "region launches -> spec C shims", shims=sorted(shims),
               inlined=tensor_specs)
    return res
