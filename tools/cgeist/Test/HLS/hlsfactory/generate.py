#!/usr/bin/env python3
"""Generate the HLSFactory lit tests: one .test per HLSFactory_v2 design.

usage: generate.py <cgeist> [--hlsfactory-root DIR] [--hls-est-validate DIR]
                   [--resdir DIR] [--headers real|stubs] [-j N] [--timeout S]

The design catalogue is the one the hls-est harness uses: the HLSFactory 1.1.0
tree (default /mnt/d/HLSFactory_v2/HLSFactory), read through
load_hlsfactory_v2() of tools/hls-est-validate/designs.py in the CIRCT tree
(default ~/circt/tools/hls-est-validate). That loader also reads the tests in
this directory to match each design with its earlier copy here, so this script
loads the catalogue before it replaces anything. Design keys
(<suite>/<test stem>) and the source / include lists of the RUN lines are the
ones the harness reports for each design.

For every design:
  - its synthesis sources and headers are copied from the HLSFactory tree into
    Inputs/<design path> (text decoded with replacement, HLSyn design-space
    placeholders instantiated: auto{__PARA__Lk} -> 2, auto{__TILE__Lk} -> 1,
    `pipeline auto{__PIPE__Lk}` -> `pipeline`, as the DSE tool does before it
    runs Vitis); a design that shares its directory with other designs gets a
    directory of its own with only the files its sources include;
  - cgeist is run with the harness's command for the default shim. With
    --headers real (the default, the harness's `--shim real` since W11-adopt,
    CIRCT MEMOS #67) that is the Vitis 2025.1 headers of the install
    ($VITIS_AUTOPILOT, default the harness's) with the flags of the harness's
    shim/real/cgeist_flags, copied to real_cgeist_flags here and spelled
    %vitis_real_flags in the RUN line (lit.local.cfg); the tests REQUIRE the
    features vitis-autopilot and xlx-bitwidth. With --headers stubs (the
    tests of ba1c0d61) it is the Vitis stand-in headers (Inputs/vitis, a copy
    of the harness's exactclass shim) with their cgeist_flags
    (%vitis_shim_flags %vitis_include). Both add the macros and language
    standard Vitis' front end adds, --raise-scf-to-affine --memref-fullrank;
    --function='*' first, then the top function by name (what the harness
    does when '*' gives nothing);
  - the strongest test that works is written:
      full        cgeist compiles the design: the MLIR is checked (the top
                  function is emitted, with the types of its arguments, of the
                  array allocations in its body and of the module's globals),
                  and so is every --hls-pragma-report line, in order;
      errors      the design's own pragmas are invalid: the exact HLS
                  diagnostics are checked, and the nonzero exit;
      check-only  cgeist cannot generate code for the design for reasons
                  unrelated to pragmas (the failure is quoted), or takes over
                  --timeout seconds: --hls-pragma-check-only placement of
                  every pragma is checked;
      skip        nothing can be checked (clang cannot parse the design with
                  the headers): the test is UNSUPPORTED and quotes the
                  error.

The first lines of every test (header, mode, reason, the single `# RUN:` line
with `-o /dev/null`) keep the shape designs.py parses. The MLIR goes to %t.mlir
through %hlsfactory_mlir (lit.local.cfg), a second -o that overrides the first.

Expectations are the tool's output at generation time: these are regression
tests over real code. Every non-applied outcome and every design that is not
compiled end to end is listed in SUMMARY.md for review.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "Inputs"
STUBS = HERE.parent / "Inputs" / "vitis"
HDR_EXT = {".h", ".hh", ".hpp", ".hxx", ".inc", ".tcc", ".ipp"}

ap = argparse.ArgumentParser()
ap.add_argument("cgeist")
ap.add_argument("--hlsfactory-root", default=os.environ.get("HLSFACTORY_ROOT", "/mnt/d/HLSFactory_v2/HLSFactory"))
ap.add_argument("--hls-est-validate", default=str(Path.home() / "circt" / "tools" / "hls-est-validate"))
ap.add_argument("--resdir", default=None, help="clang resource dir (default: the one of the LLVM build)")
ap.add_argument("--headers", choices=("real", "stubs"), default="real",
                help="real: the Vitis install's headers (the harness's default shim 'real'); "
                     "stubs: the stand-ins in ../Inputs/vitis (the exactclass shim)")
ap.add_argument("-j", type=int, default=4)
ap.add_argument("--timeout", type=int, default=180)
args = ap.parse_args()
sys.path.insert(0, args.hls_est_validate)
import designs as D        # noqa: E402  (harness catalogue)
import hlsfactory_v2 as V2  # noqa: E402
import run_hlsest as RH    # noqa: E402  (find_top, the harness's top-function lookup)
import run_vitis as RV     # noqa: E402  (cache_keys: which sources a ground-truth entry was synthesised from)

RESDIR = args.resdir or str(HERE.parents[4] / "llvm-project" / "build" / "lib" / "clang" / "18")
REAL_FLAGS_FILE = HERE / "real_cgeist_flags"


def read_flags(path):
    """The flags of a cgeist_flags file ('#' starts a comment)."""
    out = []
    for line in Path(path).read_text().splitlines():
        out += line.split("#", 1)[0].split()
    return out


if args.headers == "real":
    # A copy of the harness's shim/real/cgeist_flags, read by lit.local.cfg too (%vitis_real_flags),
    # so the tests run the command the harness runs. The autopilot directory in it is the harness's
    # Vitis install; lit.local.cfg replaces it with $VITIS_AUTOPILOT when that is set.
    shutil.copyfile(Path(args.hls_est_validate) / "shim" / "real" / "cgeist_flags", REAL_FLAGS_FILE)
    SHIM_FLAGS = read_flags(REAL_FLAGS_FILE)
    HEADER_INCLUDES = []
    SHIM_SUBS = "%vitis_real_flags"
else:
    SHIM_FLAGS = read_flags(STUBS / "cgeist_flags")
    HEADER_INCLUDES = ["-I", str(STUBS)]
    SHIM_SUBS = "%vitis_shim_flags"
VITIS_MACROS = ["-D__SYNTHESIS__", "-D__VITIS_HLS__", "-D__HLS_SYN__", "-DAESL_SYN"]


def safe_name(rel):
    return rel.replace("/", "__").replace(" ", "_")


def fc_escape(s):
    """Literal text for a FileCheck pattern."""
    out = []
    for piece in re.split(r"(\{\{|\}\}|\[\[|\]\])", s):
        if piece in ("{{", "}}", "[[", "]]"):
            out.append("{{" + re.escape(piece) + "}}")
        else:
            out.append(piece)
    return "".join(out)


def instantiate(text):
    """HLSyn design-space placeholders, as the DSE tool fills them before running Vitis."""
    text = re.sub(r"(pipeline\s+)auto\{__PIPE__\w+\}", r"\1", text, flags=re.I)
    text = re.sub(r"auto\{__PARA__\w+\}", "2", text)
    text = re.sub(r"auto\{__TILE__\w+\}", "1", text)
    return text


# ----------------------------------------------------------------- catalogue --
def frontend(sources):
    """The Vitis front-end flags, and the lit substitution that stands for them: -std= only when every
    source is in one language (one flag cannot serve C and C++)."""
    langs = {"c" if s.endswith(".c") else "c++" for s in sources}
    if langs == {"c"}:
        return VITIS_MACROS + ["-std=gnu99"], "%vitis_frontend_c99"
    if langs == {"c++"}:
        return VITIS_MACROS + ["-std=gnu++14"], "%vitis_frontend_cpp"
    return VITIS_MACROS, "%vitis_frontend_mixed"


def test_stem(r):
    """The test file's name: the design key with '/' as '__' (an older key keeps its spaces)."""
    return f"{r['dataset']}__{r['benchmark']}"


def own_dir(r):
    """True when the harness compiles record r from the HLSFactory tree, not from an earlier copy here."""
    return INPUTS not in Path(r["dir"]).parents


def test_rel(r):
    """Inputs/<rel> and <suite>/<safe_name(rel)>.test for record r, chosen so that designs.py reads
    back the record's own dataset / benchmark key."""
    d = Path(r["dir"])
    try:
        rel = d.relative_to(INPUTS).as_posix()
    except ValueError:
        v2rel, suite = r["v2"]["rel"], r["dataset"]
        base = safe_name(v2rel)[len(suite) + 2:]
        if not r["benchmark"].startswith(base):
            raise SystemExit(f"cannot place {suite}/{r['benchmark']} (HLSFactory dir {v2rel})")
        rel = v2rel + r["benchmark"][len(base):]
    if rel.split("/")[0] != r["dataset"] or test_stem(r) not in (safe_name(rel), rel.replace("/", "__")):
        raise SystemExit(f"design path {rel!r} does not give the key {r['dataset']}/{r['benchmark']}")
    return rel


def reached_files(ddir, sources, includes):
    """{path relative to ddir: absolute path} of the sources and every file their #include lines reach
    (quoted: the including file's directory, then the include dirs; angle brackets: the include dirs)."""
    inc_dirs = [os.path.normpath(os.path.join(ddir, i)) for i in includes]
    out, todo = {}, []
    for s in sources:
        p = os.path.normpath(os.path.join(ddir, s))
        out[os.path.normpath(s)] = p
        todo.append(p)
    while todo:
        p = todo.pop()
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        for kind, name in V2.INCLUDE_LINE.findall(data):
            name = name.decode("utf-8", errors="replace").strip()
            for b in ([os.path.dirname(p)] if kind == b'"' else []) + inc_dirs:
                q = os.path.normpath(os.path.join(b, name))
                if os.path.isfile(q):
                    rq = os.path.relpath(q, ddir)
                    if rq not in out:
                        out[rq] = q
                        todo.append(q)
                    break
    return out


def files_to_copy(r, shared_dir):
    """What Inputs/<rel> holds for record r: the sources, every file they include and, for a design
    with a directory of its own, every header in that directory."""
    src_dir = r["v2_dir"]
    srcs = r["v2"].get("v2_sources") or r["sources"]
    incs = r["v2"].get("v2_includes") or r["includes"]
    files = reached_files(src_dir, srcs, incs)
    if not shared_dir:
        for p in Path(src_dir).rglob("*"):
            if p.is_file() and p.suffix in HDR_EXT:
                files.setdefault(os.path.relpath(p, src_dir), str(p))
    if not own_dir(r):
        # The harness compiles this design from the earlier copy here and keys its caches (the Vitis
        # key hashes every source-like file of the directory) on that copy's files: refresh every file
        # of the copy that the HLSFactory tree still has (e.g. a testbench an earlier extraction kept),
        # so that an unchanged design keeps its keys.
        for p in Path(r["dir"]).rglob("*"):
            relp = os.path.relpath(p, r["dir"])
            if p.is_file() and os.path.isfile(os.path.join(src_dir, relp)):
                files.setdefault(relp, os.path.join(src_dir, relp))
    return files


# ------------------------------------------------------------------- cgeist --
def cgeist_cmd(t, func, out, extra):
    cmd = [args.cgeist, *t["sources"], f"--function={func}", "-S", "--hls-annotate",
           "--raise-scf-to-affine", "--memref-fullrank", *SHIM_FLAGS, f"-resource-dir={RESDIR}",
           *t["fe_flags"], *HEADER_INCLUDES, "-I", "."]
    for inc in t["includes"]:
        if inc != ".":
            cmd += ["-I", inc]
    return cmd + ["-o", out, *extra]


def run(t, func, out, extra):
    try:
        p = subprocess.run(cgeist_cmd(t, func, out, extra), cwd=INPUTS / t["rel"], capture_output=True,
                           text=True, errors="replace", timeout=args.timeout)
        return p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


def header_dirs():
    """The include directories of the headers cgeist reads instead of the design's (--headers real:
    the Vitis install's autopilot tree): their own pragmas are the library's, not the design's."""
    return [SHIM_FLAGS[i + 1] for i, f in enumerate(SHIM_FLAGS[:-1]) if f == "-I"] if args.headers == "real" else []


def remarks(err):
    """The design's pragma-report lines. Under --headers real the remarks on the pragmas of the Vitis
    headers themselves (tens of thousands of 'the enclosing function is not emitted' for the members
    of ap_int_ref.h, hls_math.h ... a design does not use) are left out: they pin the install, not
    the design, and the tests' trailing NOT only rejects remarks on files given relative to the
    design directory (every design file is, since the RUN line runs in it)."""
    dirs = tuple(d.rstrip("/") + "/" for d in header_dirs())
    return [l for l in err.splitlines() if ": remark: '#pragma HLS" in l
            and not (dirs and l.startswith(dirs))]


# The trailing NOT of a test: no further remark. Under --headers real only remarks on a design file
# (a relative path) count; remarks on the Vitis headers (absolute paths) are not the design's.
REMARK_NOT = "{{^[^/[:space:]][^[:space:]]*}}: remark:" if args.headers == "real" else "remark:"


def hls_errors(err):
    """Our pragma errors (clang-style lines naming an HLS pragma)."""
    return [l for l in err.splitlines()
            if re.search(r":\d+:\d+: error: .*(#pragma HLS|'HLS|pragma HLS)", l)
            or re.search(r":\d+:\d+: error: (expected a pragma name|expected an option name|option '.*' is specified more than once)", l)]


def first_error(code, err):
    if code == "timeout":
        return f"cgeist takes over {args.timeout} s"
    for l in err.splitlines():
        if "error:" in l or "Assertion" in l:
            return re.sub(r"^.*?(error:|Assertion)", r"\1", l).strip()[:200]
    if "Stack dump" in err or (isinstance(code, int) and code < 0):
        return f"cgeist crashed (exit {code})"
    if code == 0:
        return "cgeist emitted no function"
    return f"cgeist failed (exit {code}) without an error message"


# --------------------------------------------------------------- MLIR checks --
FUNC_LINE = re.compile(r"^\s*func\.func\s+(?:private\s+)?@(?P<name>[\w$.]+)\(")
ALLOC_RE = re.compile(r"=\s*memref\.(?P<op>alloca|alloc)\(.*:\s*(?P<type>memref<.*>)\s*$")
GLOBAL_RE = re.compile(r"^\s*memref\.global\s+.*?@(?P<name>[\w$.]+)\s*:\s*(?P<type>memref<[^=]*>)(\s*=.*)?$")


def balanced_args(line, start):
    """The text from line[start] up to the ')' that closes the '(' at line[start - 1]."""
    depth = 1
    for i in range(start, len(line)):
        c = line[i]
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
            if depth == 0:
                return line[start:i]
    return line[start:]


def mlir_checks(path, top):
    """MLIR check lines: the module's globals, the top function's signature and the array allocations
    in its body. The top function is found as the harness finds it; (None, reason) when the MLIR has no
    function of that name."""
    sym, how = RH.find_top(path, top)
    lines = Path(path).read_text(errors="replace").splitlines()
    hdr = next((i for i, l in enumerate(lines)
                if (m := FUNC_LINE.match(l)) and m["name"] == sym), None)
    if sym is None or hdr is None:
        return None, "cgeist emitted no function"
    if how not in ("exact", "demangled", "mangled"):
        return None, (f"cgeist compiles the design but emits no function named {top} (the harness "
                      f"would estimate {sym} instead)")
    before, after = [], []
    for i, l in enumerate(lines):
        m = GLOBAL_RE.match(l)
        if m:
            (before if i < hdr else after).append(
                f"# MLIR-DAG: memref.global {{{{.*}}}}@{fc_escape(m['name'])} : {fc_escape(m['type'].strip())}")
    out = before
    hl = lines[hdr]
    open_at = hl.index("@" + sym + "(") + len(sym) + 2
    private = "private " if re.match(r"^\s*func\.func\s+private\s", hl) else ""
    out.append(f"# MLIR-LABEL: func.func {private}@{fc_escape(sym)}(")
    out.append(f"# MLIR-SAME: {fc_escape(balanced_args(hl, open_at))})")
    if hl.rstrip().endswith("{"):
        indent = len(hl) - len(hl.lstrip())
        for l in lines[hdr + 1:]:
            if l.strip() == "}" and len(l) - len(l.lstrip()) == indent:
                break
            m = ALLOC_RE.search(l)
            if m and re.match(r"memref<(\?|\d+)x", m["type"]):
                out.append(f"# MLIR: = memref.{m['op']}({{{{.*}}}}: {fc_escape(m['type'])}{{{{$}}}}")
    return out + after, ""


# -------------------------------------------------------------- per design --
def classify(t):
    tmpd = tempfile.mkdtemp(prefix="hlsf-gen-")
    try:
        out = os.path.join(tmpd, "out.mlir")
        first, no_top = None, None
        for func in ("*", t["top"]):
            if os.path.exists(out):
                os.unlink(out)
            code, err = run(t, func, out, ["--hls-pragma-report"])
            if first is None:
                first = (code, err)
            if code == 0 and os.path.exists(out) and \
                    re.search(r"func\.func\s", Path(out).read_text(errors="replace")):
                checks, why = mlir_checks(out, t["top"])
                if checks is not None:
                    return {"mode": "full", "func": func, "lines": remarks(err), "mlir": checks}
                no_top = why
            if code == "timeout":
                break
        if no_top:
            return {"mode": "skip", "why": no_top}
        code, err = first
        errs = hls_errors(err)
        if code == 1 and errs:
            return {"mode": "errors", "lines": errs}
        why = first_error(code, err)
        code2, err2 = run(t, "*", "/dev/null", ["--hls-pragma-check-only"])
        if code2 == 0:
            return {"mode": "check-only", "lines": remarks(err2), "why": why}
        # The reason is the failure of the compile the RUN line makes (the harness's command); the
        # pragma-check-only failure that rules out check-only mode is added when it differs (hp_fft_hls
        # UF*: the compile asserts in ValueCategory, only the check-only run reports the unroll factor 0;
        # CIRCT W11-adopt review D4).
        if code == "timeout":
            return {"mode": "skip", "why": why}
        why2 = first_error(code2, err2)
        return {"mode": "skip", "why": why if why2 == why else f"{why}; with --hls-pragma-check-only: {why2}"}
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def write_test(t, info):
    rel, mode = t["rel"], info["mode"]
    tdir = HERE / t["dataset"]
    tdir.mkdir(parents=True, exist_ok=True)
    srcs = " ".join(shlex.quote(s) for s in t["sources"])
    incs = " ".join(f"-I {shlex.quote(i)}" for i in ["."] + [i for i in t["includes"] if i != "."])
    func = info.get("func", "*")
    base = (f"cd \"%S/../Inputs/{rel}\" && cgeist {srcs} --function={shlex.quote(func)} -S --hls-annotate "
            f"--raise-scf-to-affine --memref-fullrank {SHIM_SUBS} %resourcedir {t['fe_sub']} "
            f"{'%vitis_include ' if args.headers == 'stubs' else ''}{incs} -o /dev/null")
    L = [f"# HLSFactory {rel} (top: {t['top']}), generated by generate.py.", f"# Mode: {mode}."]
    # A test whose name has whitespace (Monte-Carlo "Random 100 Paths"): %s and %t expand to paths
    # with spaces, which the shell must see quoted.
    spaced = re.search(r"\s", t["stem"]) is not None
    fs = '"%s"' if spaced else "%s"
    mlir_out, mlir_in = ('-o "%t.mlir"', '"%t.mlir"') if spaced else ("%hlsfactory_mlir", "%t.mlir")
    if mode == "full":
        L += [f"# RUN: {base} {mlir_out} --hls-pragma-report 2>&1 | "
              f"FileCheck --check-prefix=PRAGMA --allow-empty {fs} && FileCheck --check-prefix=MLIR {fs} < {mlir_in}",
              "# The hls-est harness's cgeist command for this design, plus the pragma report (which does",
              "# not change the MLIR).",
              "# PRAGMA-NOT: Stack dump"]
        L += [f"# PRAGMA: {fc_escape(l)}" for l in info["lines"]]
        L += [f"# PRAGMA-NOT: {REMARK_NOT}"]
        L += info["mlir"]
    elif mode == "errors":
        L += ["# The design's own pragmas are invalid; the diagnostics are checked.",
              f"# RUN: {base.replace('&& cgeist', '&& not cgeist', 1)} --hls-pragma-report 2>&1 | "
              f"FileCheck --allow-empty {fs}",
              "# CHECK-NOT: Stack dump"]
        L += [f"# CHECK-DAG: {fc_escape(l)}" for l in info["lines"]]
    elif mode == "check-only":
        L += [f"# cgeist cannot generate code for this design, for reasons unrelated to HLS pragmas: {info['why']}",
              "# Pragma parsing, placement and validation are checked instead.",
              f"# RUN: {base} --hls-pragma-check-only 2>&1 | FileCheck --allow-empty {fs}",
              "# CHECK-NOT: error:"]
        L += [f"# CHECK: {fc_escape(l)}" for l in info["lines"]]
        L += [f"# CHECK-NOT: {REMARK_NOT}"]
    else:
        L += [f"# cgeist cannot compile this design: {info['why']}",
              f"# RUN: {base}",
              "# UNSUPPORTED: true"]
    if args.headers == "real" and mode != "skip":
        # After the RUN line: designs.py reads the comment lines before it as the reason.
        L.append("# REQUIRES: vitis-autopilot, xlx-bitwidth")
    (tdir / f"{t['stem']}.test").write_text("\n".join(L) + "\n")


def load_vitis():
    """The harness's ground truth ({} when the file is not there)."""
    path = Path(args.hls_est_validate) / "results" / "vitis.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def vitis_outcome(t, vitis):
    """What Vitis did with the design of test t: 'ok (N BRAM)', 'ok, other sources' (the entry was
    synthesised from sources other than the ones the harness compiles now), the failure status, or
    'no result'."""
    v = vitis.get(f"{t['dataset']}/{t['benchmark']}")
    if not v:
        return "no result"
    if v.get("status") != "ok":
        return v.get("status", "no result")
    if v.get("key") not in RV.cache_keys(t["record"]).values():
        return "ok, other sources"
    return f"ok ({(v.get('totals') or {}).get('BRAM')} BRAM)"


def md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_summary(results):
    modes = Counter(res["mode"] for _, res in results.values())
    S = ["# HLSFactory tests: summary", "",
         "Generated by `generate.py` from the HLSFactory_v2 catalogue; regenerate after changing",
         "the extraction and review the diff of this file.", "",
         ("Headers: the real Vitis 2025.1 headers (`--headers real`, the hls-est harness's default shim "
          "`real`; flags in `real_cgeist_flags`)." if args.headers == "real" else
          "Headers: the stand-ins in `../Inputs/vitis` (`--headers stubs`, the exactclass shim)."),
         f"cgeist: md5 {md5(args.cgeist)}.", "", "| mode | designs |", "|---|---|"]
    S += [f"| {m} | {n} |" for m, n in sorted(modes.items())]
    outcomes = Counter()
    for _, res in results.values():
        for l in res.get("lines", []):
            m = re.search(r"' -> (.*)$", l)
            if m:
                o = m[1]
                outcomes["applied" if not o.startswith(("not applied", "rejected", "ignored"))
                         else re.sub(r"'[^']*'", "X", o.split(" (")[0])] += 1
    S += ["", "## Pragma outcomes", "", "| outcome | count |", "|---|---|"]
    S += [f"| {k} | {v} |" for k, v in outcomes.most_common()]
    S += ["", "## Designs not compiled end to end", "",
          "The Vitis column is the design's ground truth in the hls-est harness (results/vitis.json, Vitis 2025.1",
          "csynth). A design Vitis synthesises is a cgeist gap (the front end, or the stand-in headers of",
          "--headers stubs), not an error",
          "in the design, whatever the quoted error says; only a design Vitis also rejects can have an error of its",
          "own.", "",
          "| design | mode | Vitis | reason |", "|---|---|---|---|"]
    vitis = load_vitis()
    for rel, (t, res) in sorted(results.items()):
        if res["mode"] != "full":
            S.append(f"| {rel} | {res['mode']} | {vitis_outcome(t, vitis)} | {res.get('why', '').replace('|', '/')} |")
    S += ["", "## Pragmas not applied in compiled designs", ""]
    for rel, (t, res) in sorted(results.items()):
        for l in res.get("lines", []):
            if re.search(r"-> (not applied|rejected|ignored)", l) or ": error:" in l:
                S.append(f"- `{rel}`: `{l.strip()}`")
    (HERE / "SUMMARY.md").write_text("\n".join(S) + "\n")
    print(dict(modes))


def main():
    problems = []
    recs = D.load_hlsfactory_v2(args.hlsfactory_root, test_hls_dir=str(HERE.parent), problems=problems)
    if problems:
        raise SystemExit("catalogue problems:\n" + "\n".join(problems))
    dir_users = Counter(r["v2_dir"] for r in recs)
    tests = []
    for r in recs:
        fe_flags, fe_sub = frontend(r["sources"])
        tests.append({"rel": test_rel(r), "stem": test_stem(r), "dataset": r["dataset"], "benchmark": r["benchmark"], "top": r["top"],
                      "sources": r["sources"], "includes": r["includes"], "fe_flags": fe_flags,
                      "fe_sub": fe_sub, "files": files_to_copy(r, shared_dir=own_dir(r) and dir_users[r["v2_dir"]] > 1),
                      "record": r})
    dup = [k for k, n in Counter(t["rel"] for t in tests).items() if n > 1]
    if dup:
        raise SystemExit(f"two designs in one Inputs directory: {dup}")
    # Read every source before Inputs/ is replaced (the catalogue may point into it).
    contents = {t["rel"]: {k: instantiate(Path(v).read_text(errors="replace")) for k, v in t["files"].items()}
                for t in tests}
    if INPUTS.exists():
        shutil.rmtree(INPUTS)
    for old in HERE.glob("*/*.test"):
        old.unlink()
    for t in tests:
        for relp, text in contents[t["rel"]].items():
            dst = Path(os.path.normpath(INPUTS / t["rel"] / relp))
            if INPUTS not in dst.parents:
                raise SystemExit(f"{t['rel']}: {relp} would be copied outside Inputs/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.j) as ex:
        for t, res in zip(tests, ex.map(classify, tests)):
            results[t["rel"]] = (t, res)
            write_test(t, res)
    write_summary(results)


main()
