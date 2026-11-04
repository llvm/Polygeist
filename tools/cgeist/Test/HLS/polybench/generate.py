"""Generate tools/cgeist/Test/HLS/polybench/<kernel>.c from HLSFactory.

Each HLSFactory PolyBench design carries its directives as a Tcl template
(opt_template.tcl). We instantiate one configuration of it as inline
#pragma HLS directives:
  - set_directive_resource -core RAM_1P X      -> bind_storage type=ram_1p
  - set_directive_interface -mode M X          -> interface M port=X
  - set_directive_array_partition ... -dim D X -> array_partition cyclic factor=4 dim=D
  - set_directive_unroll / _pipeline "k/L"     -> unroll factor=4 / pipeline in loop L
  - loop_opt blocks                            -> row 1 (pipeline + unroll factor=2)
Expected outcomes (CHECK lines) are derived from the directives, not from
cgeist's output.
"""
import re
import sys
from pathlib import Path

# Usage: python generate.py <HLSFactory>/hlsfactory/hls_dataset_sources/polybench
SRC = Path(sys.argv[1])
DST = Path(__file__).resolve().parent
KERNELS = ["bicg", "gesummv", "k2mm", "k3mm", "mvt", "syr2k", "syrk"]
ALIASES = {"k2mm": "2mm", "k3mm": "3mm"}
# Arrays the source writes but never reads: the optimizer removes them, and
# cgeist must say so rather than drop the pragma silently.
WRITE_ONLY = {"syr2k": {"buff_B1"}}


def parse_tcl(k):
    fn_pragmas = []  # (kind, text, var)
    loop_pragmas = {}  # label -> [text]
    lines = (SRC / k / "opt_template.tcl").read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        m = re.match(r'set_directive_resource -core RAM_1P "\w+" (\w+)', ln)
        if m:
            fn_pragmas.append(("bind_storage", f"bind_storage variable={m[1]} type=ram_1p", m[1]))
        m = re.match(r'set_directive_interface -mode (\w+) "\w+" (\w+)', ln)
        if m:
            fn_pragmas.append(("interface", f"interface {m[1]} port={m[2]}", m[2]))
        m = re.match(r'set_directive_array_partition .*-dim (\d+) "\w+" (\w+)', ln)
        if m:
            fn_pragmas.append(("array_partition",
                               f"array_partition variable={m[2]} cyclic factor=4 dim={m[1]}", m[2]))
        m = re.match(r'set_directive_unroll -factor \[factor\] "\w+/(\w+)"', ln)
        if m:
            loop_pragmas.setdefault(m[1], []).append("unroll factor=4")
        m = re.match(r'set_directive_pipeline "\w+/(\w+)"', ln)
        if m:
            loop_pragmas.setdefault(m[1], []).append("pipeline")
        if ln.startswith("loop_opt,"):
            rows = []
            j = i + 1
            while j < len(lines) and re.match(r"\d+,", lines[j].strip()):
                rows.append(lines[j].strip().split(","))
                j += 1
            label = rows[1][1]
            loop_pragmas.setdefault(label, []).extend(["pipeline II=1", "unroll factor=2"])
            i = j
            continue
        i += 1
    return fn_pragmas, loop_pragmas


def generate(k):
    fn_pragmas, loop_pragmas = parse_tcl(k)
    src = (SRC / k / "src" / f"{k}.c").read_text().splitlines()
    header = (SRC / k / "src" / f"{k}.h").read_text()
    n = re.search(r"#define N (\d+)", header)[1]
    sig = next(l for l in src if l.startswith("void "))
    params = [p.strip() for p in re.search(r"\((.*)\)", sig)[1].split(",")]
    pnames = [re.match(r"DATA_TYPE (\w+)", p)[1] for p in params]

    out = []
    checks = []
    first_loop_done = False
    for ln in src:
        if re.search(r'#include "', ln) or "ap_cint" in ln:
            continue
        # Function-level pragmas go right before the first loop, after the
        # local declarations, the way Vitis users write them.
        if not first_loop_done and re.match(r"\s*\w+:\s*for", ln):
            first_loop_done = True
            for kind, text, var in fn_pragmas:
                out.append(f"#pragma HLS {text}")
                lineno = len(out)  # 1-based line of the pragma (before header)
                if var in pnames:
                    what = f"argument #{pnames.index(var)} (variable '{var}') of @{k}"
                elif var in WRITE_ONLY.get(k, ()):
                    what = f"not applied: the variable '{var}' was removed during optimization"
                else:
                    what = f"memref.alloca (variable '{var}') in @{k}"
                checks.append((lineno, text, what))
        out.append(ln)
        m = re.match(r"\s*(\w+):\s*for\b", ln)
        if m and m[1] in loop_pragmas:
            assert ln.rstrip().endswith("{"), (k, ln)
            for text in loop_pragmas[m[1]]:
                out.append(f"#pragma HLS {text}")
                checks.append((len(out), text,
                               f"{{{{affine|scf}}}}.for (loop '{m[1]}') in @{k}"))

    name = ALIASES.get(k, k)
    head = [
        f"// PolyBench {name} from HLSFactory (hls_dataset_sources/polybench/{k}),",
        "// with one configuration of its opt_template.tcl directives written as",
        "// inline pragmas (cyclic factor=4 partitions, pipeline/unroll on the",
        "// labelled loops, bind_storage for RAM_1P resources, ap_fifo outputs).",
        "// Generated; the CHECK lines follow from the directives alone.",
        f"// RUN: cgeist %s --function={k} %stdinclude -S --hls-annotate --hls-pragma-report --raise-scf-to-affine -o %t.mlir 2>&1 | FileCheck %s --check-prefix=REPORT",
        "// RUN: FileCheck %s --check-prefix=IR < %t.mlir",
        f"// RUN: cgeist %s --function={k} %stdinclude -S --hls-annotate --hls-pragma-report -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT",
        "",
        *[line for v in WRITE_ONLY.get(k, ())
          for line in (f"// {v} is written but never read in the original source, so it is",
                       "// removed (with a warning) instead of annotated.")],
        *(["// REPORT-NOT: warning", "// REPORT-NOT: not applied"]
          if not WRITE_ONLY.get(k) else []),
        "// REPORT-NOT: rejected",
        f"#define N {n}",
        "#define DATA_TYPE float",
        "",
    ]
    offset = len(head)
    lines = head + out
    lines.append("")
    for lineno, text, what in checks:
        pat = re.escape(text) if False else text
        lines.append(f"// REPORT-DAG: {k}.c:{lineno + offset}:1: remark: '#pragma HLS {pat}' -> {what}")
    # IR shape checks, derived from the partition directives.
    lines.append("")
    lines.append(f"// IR-LABEL: func.func @{k}(")
    for kind, text, var in fn_pragmas:
        if kind != "array_partition" or var in WRITE_ONLY.get(k, ()):
            continue
        dim = re.search(r"dim=(\d+)", text)[1]
        entry = (f'{{dim = {dim} : i32, factor = 4 : i32, kind = "cyclic", '
                 f'variable = "{var}"}}')
        lines.append(f"// IR-DAG: hls.array_partition = [{entry}]")
    (DST / f"{k}.c").write_text("\n".join(lines) + "\n", newline="\n")
    return len(checks)


DST.mkdir(parents=True, exist_ok=True)
for k in KERNELS:
    print(k, generate(k))


def e2e():
    """polybench/hls-est-e2e.c: every kernel through the hls-est estimation
    pipeline; bank counts follow from the partition directives."""
    lines = [
        "// The PolyBench kernels of this directory, end to end through the",
        "// hls-est BRAM estimation pipeline (as used by",
        "// ~/circt/test/hls-est/Estimation). Each cyclic factor=4 partition of a",
        "// local N=64 array yields 4 banks; the counts follow from the pragmas.",
        "// Generated alongside the kernel tests.",
        "// REQUIRES: hls-est",
        "",
    ]
    for k in KERNELS:
        fn_pragmas, _ = parse_tcl(k)
        src = (SRC / k / "src" / f"{k}.c").read_text()
        sig = next(l for l in src.splitlines() if l.startswith("void "))
        pnames = [re.match(r"DATA_TYPE (\w+)", p.strip())[1]
                  for p in re.search(r"\((.*)\)", sig)[1].split(",")]
        one_d = two_d = 0
        for kind, text, var in fn_pragmas:
            if kind != "array_partition" or var in pnames or var in WRITE_ONLY.get(k, ()):
                continue
            decl = re.search(r"DATA_TYPE[^;]*\b" + var + r"((?:\[N\])+)", src)
            dims = decl[1].count("[N]") if decl else 0
            if dims == 1:
                one_d += 4
            elif dims == 2:
                two_d += 4
        K = k.upper()
        pipeline = ("--allow-unregistered-dialect --affine-loop-normalize "
                    "--memory-banking-bram --convert-affine-to-loopschedule "
                    f"--symbol-privatize=exclude={k} --symbol-dce --bram-analysis")
        lines += [
            f"// RUN: cgeist %S/{k}.c --function={k} %stdinclude -S --hls-annotate --raise-scf-to-affine -o %t.{k}.mlir",
            f"// RUN: %hls-est %t.{k}.mlir {pipeline} -o %t.{k}.est.mlir | FileCheck %s --check-prefix={K}",
        ]
        if one_d:
            lines.append(f"// RUN: FileCheck %s --check-prefix={K}-1D < %t.{k}.est.mlir")
        if two_d:
            lines.append(f"// RUN: FileCheck %s --check-prefix={K}-2D < %t.{k}.est.mlir")
        lines.append(f"// {K}: Total BRAM:")
        if one_d:
            lines += [f"// {K}-1D-COUNT-{one_d}: memref.alloca() : memref<16xf32>",
                      f"// {K}-1D-NOT: memref.alloca() : memref<16xf32>"]
        if two_d:
            lines += [f"// {K}-2D-COUNT-{two_d}: memref.alloca() : memref<64x16xf32>",
                      f"// {K}-2D-NOT: memref.alloca() : memref<64x16xf32>"]
        lines.append("")
    (DST / "hls-est-e2e.c").write_text("\n".join(lines), newline="\n")


e2e()
