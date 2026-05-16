#!/home/arjaiswal/slacker/.venv/bin/python3
"""Build a static HTML index of PolyBench kernels where each row deep-links to
Compiler Explorer with the full Polygeist pipeline pre-wired:

  - left column:  C source editor + cgeist_aff compiler pane (shows affine MLIR)
  - right column: MLIR editor (pre-filled with affine MLIR) + popt_full compiler
                  pane + Opt Pipeline view (every internal pass clickable)

Per-kernel HTML pages with raised / debuferized / kernel.launch IR are also
rendered (uses the existing matcher pipeline).

Inputs:
  - PolyBench C sources at $POLYBENCH/tools/cgeist/Test/polybench/.../<k>.c
  - Pre-computed affine MLIR at /tmp/polybench_new/<k>.mlir
  - Pre-computed linalg MLIR at /tmp/polybench_new/<k>_linalg.mlir
  - Pre-computed debuf MLIR  at /tmp/polybench_new/<k>_debuf.mlir

Output:
  /tmp/ir_viewer/index.html   (entrypoint — open this)
  /tmp/ir_viewer/<k>.html     (per-kernel IR preview)
"""
import json
import re
import subprocess
import urllib.parse
from pathlib import Path

POLYBENCH_TEST_DIR = Path("/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench")
POLYBENCH_UTILS = POLYBENCH_TEST_DIR / "utilities"
MLIR_DIR = Path("/tmp/polybench_new")
MACHSUITE_ROOT = Path("/home/arjaiswal/Polygeist/third_party/MachSuite")
MACHSUITE_MLIR_DIR = Path("/tmp/machsuite_mlir")
NPB_ROOT = Path("/home/arjaiswal/Polygeist/third_party/NPB-polybenchified")
NPB_MLIR_DIR = Path("/tmp/npb_mlir")
OUTPUT_DIR = Path("/tmp/ir_viewer")
REWRITER = Path("/home/arjaiswal/Polygeist/scripts/correctness/kernel_match_rewrite.py")
PYTHON = "/home/arjaiswal/slacker/.venv/bin/python3"

# MachSuite tag → (relative subdir under third_party/MachSuite, kernel function).
# The tag is what the viewer uses for filenames and as the display name.
MACHSUITE_KERNELS: dict[str, tuple[str, str]] = {
    "aes":           ("aes/aes",              "aes256_encrypt_ecb"),
    "backprop":      ("backprop/backprop",    "backprop"),
    "bfs-bulk":      ("bfs/bulk",             "bfs"),
    "bfs-queue":     ("bfs/queue",            "bfs"),
    "fft-strided":   ("fft/strided",          "fft"),
    "fft-transpose": ("fft/transpose",        "fft1D_512"),
    "gemm-ncubed":   ("gemm/ncubed",          "gemm"),
    "gemm-blocked":  ("gemm/blocked",         "bbgemm"),
    "kmp":           ("kmp/kmp",              "kmp"),
    "md-grid":       ("md/grid",              "md"),
    "md-knn":        ("md/knn",               "md_kernel"),
    "nw":            ("nw/nw",                "needwun"),
    "sort-merge":    ("sort/merge",           "ms_mergesort"),
    "sort-radix":    ("sort/radix",           "ss_sort"),
    "spmv-crs":      ("spmv/crs",             "spmv"),
    "spmv-ellpack":  ("spmv/ellpack",         "ellpack"),
    "stencil2d":     ("stencil/stencil2d",    "stencil"),
    "stencil3d":     ("stencil/stencil3d",    "stencil3d"),
    "viterbi":       ("viterbi/viterbi",      "viterbi"),
}

# PolyBench-extracted NPB kernels (one .c per kernel in NPB-polybenchified/).
# These were manually carved out of the monolithic per-benchmark .c files
# in NPB3.0-omp-C; the kernel functions had their static-global dependencies
# converted to explicit array parameters so the pipeline can isolate them
# without the extraction issues the whole-file sweep hit.
NPB_KERNELS: dict[str, tuple[str, str]] = {
    "bt-add":      ("bt_add.c",      "bt_add"),
    "ft-evolve":   ("ft_evolve.c",   "ft_evolve"),
    "lu-l2norm":   ("lu_l2norm.c",   "lu_l2norm"),
    "mg-psinv":    ("mg_psinv.c",    "mg_psinv"),
    "mg-resid":    ("mg_resid.c",    "mg_resid"),
    "mg-norm2u3":  ("mg_norm2u3.c",  "mg_norm2u3"),
    "mg-rprj3":    ("mg_rprj3.c",    "mg_rprj3"),
}

# Per-NPB-kernel parallelism + characterisation notes.
NPB_NOTES: dict[str, tuple[str, str]] = {
    "bt-add":      ("highly parallel",   "BT vector add over 4D field — pure elemwise, fully parallel"),
    "ft-evolve":   ("highly parallel",   "FT timestep multiply — parallel but uses ex[indexmap[...]] gather; raise refuses indirect index"),
    "lu-l2norm":   ("highly parallel",   "LU L2 norm over 4D field — reduction over the spatial axes"),
    "mg-psinv":    ("highly parallel",   "MG smoother — 27-point stencil via per-row r1/r2 scratch arrays; outer i3/i2 hold scratch state"),
    "mg-resid":    ("highly parallel",   "MG residual r = v - Au — same 27-point stencil shape as psinv"),
    "mg-norm2u3":  ("highly parallel",   "MG L2 + L∞ combined norm — mixed sum+max reductions in one loop; raise pass can't fuse"),
    "mg-rprj3":    ("highly parallel",   "MG restriction (trilinear FE projection) — coarse-grid 2x downsample"),
}

# Per-MachSuite-kernel parallelism + characterisation notes.
MACHSUITE_NOTES: dict[str, tuple[str, str]] = {
    "gemm-ncubed":   ("highly parallel",   "textbook 3-loop gemm with flat 1D indexing — lifts to single linalg.generic"),
    "gemm-blocked":  ("highly parallel",   "tiled gemm; blocking collapses, still matches GEMM"),
    "stencil2d":     ("highly parallel",   "9-tap 2D conv (3x3 filter), not jacobi-shaped — no matcher template yet"),
    "stencil3d":     ("highly parallel",   "3D stencil — 7-tap-ish, mostly matches"),
    "backprop":      ("partial parallel",  "neural-net backprop; many small generics, body shapes outside our library"),
    "nw":            ("serial",            "Needleman-Wunsch DP; row-by-row dependencies"),
    "fft-strided":   ("serial",            "bit-reversal addressing; outer shift loop non-affine"),
    "fft-transpose": ("partial parallel",  "transpose-based FFT; some stages parallel, others not"),
    "kmp":           ("serial",            "KMP string matching; backtracking, control-flow heavy"),
    "bfs-bulk":      ("serial",            "bulk-synchronous BFS; queue-based, non-affine"),
    "bfs-queue":     ("serial",            "queue-based BFS; non-affine indirect access"),
    "spmv-crs":      ("partial parallel",  "sparse matvec CRS — indirect indexing not raisable today"),
    "spmv-ellpack":  ("partial parallel",  "sparse matvec ELLPACK — same"),
    "sort-merge":    ("serial",            "merge sort; control flow heavy"),
    "sort-radix":    ("partial parallel",  "radix sort; counting + scatter; some stages affine"),
    "aes":           ("serial",            "byte-oriented AES; bit ops + sbox lookup; not numerical"),
    "md-grid":       ("highly parallel",   "molecular dynamics with cell-grid neighbour list"),
    "md-knn":        ("highly parallel",   "molecular dynamics with k-NN neighbour list"),
    "viterbi":       ("serial",            "Viterbi DP + arg-max; sequential along time"),
}

CE_BASE = "http://localhost:10240/"
CGEIST_NAME = "cgeist_aff"
POPT_NAME = "popt_full"
POPT_DISPLAY = "polygeist-opt: full (raise + lower-submap + debuferize)"


# Per-kernel parallelism notes — how well the kernel's algorithm maps to GPU.
# Categories used in the index column:
#   highly parallel    — every iteration independent; embarrassingly parallel
#   parallel + T loop  — body parallel, but a sequential outer time/step loop remains
#   partial parallel   — significant parallel ops mixed with reductions / serial steps
#   serial             — fundamental cross-iteration dependencies; poor GPU fit
KERNEL_NOTES: dict[str, tuple[str, str]] = {
    # BLAS-shaped — fully parallel iter space.
    "gemm":          ("highly parallel",   "dense gemm, 3-loop parallel + reduction"),
    "gemver":        ("highly parallel",   "rank-2 update + gemv stages, all parallel"),
    "gesummv":       ("highly parallel",   "two gemvs + axpby, all parallel"),
    "atax":          ("highly parallel",   "y = A·x then t = Aᵀ·y, parallel"),
    "bicg":          ("highly parallel",   "s = Aᵀ·p and q = A·r, parallel"),
    "mvt":           ("highly parallel",   "x1 += A·y1; x2 += Aᵀ·y2, parallel"),
    "2mm":           ("highly parallel",   "two chained gemms, parallel"),
    "3mm":           ("highly parallel",   "three chained gemms, parallel"),
    "symm":          ("highly parallel",   "symmetric gemm (lower triangle), parallel"),
    "syrk":          ("highly parallel",   "symmetric rank-k update (lower triangle)"),
    "syr2k":         ("highly parallel",   "symmetric rank-2k update (lower triangle)"),
    "trmm":          ("highly parallel",
                      "triangular gemm — (i,j) parallel, k reduction; raise "
                      "splits the per-i body into 2 memref linalg ops which "
                      "the matcher can't see today (form-gated)"),

    # Stencils — body parallel, outer time loop is sequential.
    "jacobi-1d":     ("parallel + T loop",
                      "3-point 1D smoother; T steps sequential, inner parallel"),
    "jacobi-2d":     ("parallel + T loop",
                      "5-point 2D stencil; T steps sequential, inner parallel"),
    "heat-3d":       ("parallel + T loop",
                      "7-point 3D Laplacian; T steps sequential, inner highly parallel"),
    "fdtd-2d":       ("parallel + T loop",
                      "E/H field cross-updates; T steps sequential, inner parallel"),
    "adi":           ("parallel + T loop",
                      "alternating direction implicit; T+sweep loops sequential, "
                      "tridiagonal solves inside each sweep partially serial"),

    # Mixed: significant parallel ops plus reductions/serial constraints.
    "correlation":   ("partial parallel",
                      "mean + stddev reductions parallel; output is symmetric, "
                      "diagonal/off-diagonal phases mostly parallel"),
    "covariance":    ("partial parallel",
                      "mean reduction + centered outer product; mostly parallel "
                      "with reduction phases"),
    "doitgen":       ("partial parallel",
                      "inner contraction parallel; outer r-update sweep "
                      "has loop-carried scratch buffer"),
    "floyd-warshall":("partial parallel",
                      "all-pairs shortest path: (i,j) parallel per k, but k loop "
                      "is strictly sequential (each k uses previous k's distances)"),

    # Strictly serial / poor GPU fit.
    "cholesky":      ("serial",
                      "L·Lᵀ factorization — outer k column update carries "
                      "dependency to all later columns; small inner parallelism"),
    "lu":            ("serial",
                      "LU factorization — same column-sequential pattern as cholesky"),
    "ludcmp":        ("serial",
                      "LU + forward/back substitution — substitution phase is "
                      "strictly sequential"),
    "gramschmidt":   ("serial",
                      "modified Gram-Schmidt — each column projects against ALL "
                      "previously orthogonalized columns; strictly sequential"),
    "trisolv":       ("serial",
                      "triangular solve — y[i] depends on y[0..i-1]; sequential "
                      "row-by-row"),
    "durbin":        ("serial",
                      "Levinson-Durbin recurrence — O(N²) outer loop with full "
                      "scalar carry (α, β) between iterations; needs persistent "
                      "CUDA kernel with cooperative-groups sync"),
    "nussinov":      ("serial",
                      "RNA folding DP — sequential over diagonals, each cell "
                      "reads from prior diagonals"),
    "seidel-2d":     ("serial",
                      "Gauss-Seidel stencil — IN-PLACE writes within an inner "
                      "iteration, so each cell reads values updated earlier in "
                      "the SAME sweep; not naturally parallel"),
    "deriche":       ("serial",
                      "recursive IIR filter — output sample y[i] depends on "
                      "y[i-1..i-k]; sequential along the filter axis"),
}


def find_kernel_c(name: str, kset: str = "polybench") -> Path | None:
    """Find <name>.c. Dispatches per kernel-set."""
    if kset == "machsuite":
        info = MACHSUITE_KERNELS.get(name)
        if not info:
            return None
        subdir, _fn = info
        # The kernel .c is the only .c in the subdir that's not local_support
        # or generate (per MachSuite layout convention).
        for p in (MACHSUITE_ROOT / subdir).glob("*.c"):
            if p.name in ("local_support.c", "generate.c"):
                continue
            return p
        return None
    if kset == "npb":
        info = NPB_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = NPB_ROOT / srcname
        return p if p.exists() else None
    # polybench
    for p in POLYBENCH_TEST_DIR.rglob(f"{name}.c"):
        if "/utilities/" in str(p):
            continue
        if p.name.endswith(".orig.c"):
            continue
        return p
    return None


def discover_kernels(mlir_dir: Path = MLIR_DIR) -> list[str]:
    """Return kernel tags present in `mlir_dir`. A kernel is "present" if
    it has any of <tag>.mlir / <tag>_linalg.mlir / <tag>_debuf.mlir /
    <tag>_debuf_mr.mlir — so kernels that fail one stage still show up
    in the index with a partial set of tabs."""
    tags: set[str] = set()
    for f in mlir_dir.glob("*.mlir"):
        name = f.stem
        for suffix in ("_debuf_mr", "_debuf", "_linalg"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        tags.add(name)
    return sorted(tags)


def build_ce_state(c_src: str, c_kernel_dir: Path, mlir_src: str) -> dict:
    """3-visible-pane CE layout state.

    Visible:
      - C editor (top-left)
      - cgeist_aff compiler reading C editor (bottom-left)
      - Opt Pipeline view bound to polygeist-opt:full (right)

    Hidden (in tab stacks alongside the visible panes):
      - LLVM IR editor with affine MLIR (tab next to C editor)
      - polygeist-opt:full compiler reading MLIR editor (tab next to Opt Pipeline)
    The hidden panes still exist so the Opt Pipeline can bind to popt_full.
    """
    editor_opts = {"compileOnChange": True, "colouriseAsm": True}
    cgeist_compiler_pane = {
        "type": "component",
        "componentName": "compiler",
        "componentState": {
            "id": 1,
            "source": 1,
            "compiler": CGEIST_NAME,
            "lang": "c",
            "editorid": 1,
            "treeid": 0,
            "filters": {},
            "options": f"-I{c_kernel_dir}",
            "libs": [],
        },
    }
    popt_compiler_pane = {
        "type": "component",
        "componentName": "compiler",
        "componentState": {
            "id": 2,
            "source": 2,
            "compiler": POPT_NAME,
            "lang": "llvm",
            "editorid": 2,
            "treeid": 0,
            "filters": {},
            "options": "",
            "libs": [],
        },
    }
    opt_pipeline_pane = {
        "type": "component",
        "componentName": "optPipelineView",
        "componentState": {
            "id": 2,
            "lang": "llvm",
            "compiler": POPT_NAME,
            "compilerName": POPT_DISPLAY,
            "editorid": 2,
            "treeid": 0,
            "selectedGroup": "",
            "selectedIndex": 0,
            "sidebarWidth": 250,
        },
    }
    c_editor = {
        "type": "component",
        "componentName": "codeEditor",
        "componentState": {"id": 1, "source": c_src, "lang": "c", "options": editor_opts},
    }
    mlir_editor = {
        "type": "component",
        "componentName": "codeEditor",
        "componentState": {"id": 2, "source": mlir_src, "lang": "llvm", "options": editor_opts},
    }
    return {
        "version": 4,
        "content": [{
            "type": "row",
            "content": [
                {
                    "type": "column",
                    "width": 50,
                    "content": [
                        # Tab stack: C editor active, LLVM IR editor on a hidden tab.
                        {
                            "type": "stack",
                            "activeItemIndex": 0,
                            "content": [c_editor, mlir_editor],
                        },
                        cgeist_compiler_pane,
                    ],
                },
                # Tab stack: Opt Pipeline active, popt_full compiler on a hidden tab.
                {
                    "type": "stack",
                    "width": 50,
                    "activeItemIndex": 0,
                    "content": [opt_pipeline_pane, popt_compiler_pane],
                },
            ],
        }],
    }


def ce_link(kernel: str, mlir_dir: Path = MLIR_DIR,
            kset: str = "polybench") -> str | None:
    """Construct the CE deep-link URL for a kernel; None if sources missing."""
    c_path = find_kernel_c(kernel, kset=kset)
    mlir_path = mlir_dir / f"{kernel}.mlir"
    if not c_path or not mlir_path.exists():
        return None
    c_src = c_path.read_text()
    mlir_src = mlir_path.read_text()
    # Strip the giant dlti spec — saves a lot of URL space and CE will recompute
    # it for the popt_full pane anyway.
    mlir_src = re.sub(
        r'module attributes \{[^\}]*\}',
        'module',
        mlir_src, count=1,
    )
    state = build_ce_state(c_src, c_path.parent, mlir_src)
    payload = json.dumps(state, separators=(',', ':'))
    return CE_BASE + "#" + urllib.parse.quote(payload, safe='')


def render_html(title: str, body_html: str, css: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #ffffff; color: #1f1f1f; }}
  .header {{ background: #f4f4f4; padding: 10px 20px;
             border-bottom: 1px solid #ddd; }}
  .header a {{ color: #0366d6; text-decoration: none; margin-right: 12px; }}
  .header a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 18px; margin: 0; }}
  h2 {{ font-size: 14px; margin: 16px 20px 4px; color: #444; }}
  .container {{ padding: 8px 20px; }}
  pre {{ margin: 0; font-size: 13px; line-height: 1.4; overflow-x: auto;
         background: #fafafa; padding: 8px; border: 1px solid #eee;
         color: #1f1f1f; font-family: ui-monospace, SFMono-Regular,
         Menlo, Consolas, monospace; }}
  {css}
  table {{ border-collapse: collapse; margin: 16px 20px; }}
  td, th {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
  th {{ text-align: left; color: #555; font-weight: 600; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.5px; }}
  tr:hover td {{ background: #f8f8f8; }}
  td a.kernel {{ color: #0366d6; text-decoration: none; font-weight: 600;
                 font-size: 15px; }}
  td a.kernel:hover {{ text-decoration: underline; }}
  td a.viewer {{ color: #666; font-size: 12px; }}
  .pass {{ color: #1a7f37; font-weight: 600; }}
  .partial {{ color: #9a6700; font-weight: 600; }}
  .none {{ color: #cf222e; font-weight: 600; }}
  .nope {{ color: #888; }}
  .intro {{ padding: 12px 20px; color: #444; max-width: 900px; }}
  .intro code {{ background: #f1f1f1; padding: 1px 6px; border-radius: 3px;
                 font-size: 13px; }}
</style></head>
<body>{body_html}</body></html>
"""


def syntax_highlight(text: str, lang: str = "llvm") -> tuple[str, str]:
    """Render MLIR as plain text inside a styled <pre>. We deliberately skip
    pygments' LLVM lexer because it doesn't recognise MLIR syntax and marks
    nearly every token with an "error" class — which renders as a red box."""
    text = re.sub(r"#dlti\.dl_spec<[^>]*>", "(dlti spec hidden)", text)
    import html
    return f'<pre class="ir">{html.escape(text)}</pre>', ''


_LOOP_RE = re.compile(r"\b(affine\.for|scf\.for|scf\.while|scf\.parallel|affine\.parallel)\b")


def count_for_loops(text: str) -> int:
    """Count loop-level ops still in the IR. Each match is one loop nest level
    that the raise pipeline did NOT lift to a linalg.generic — a measure of how
    much imperative structure the kernel still carries after the pipeline."""
    return len(_LOOP_RE.findall(text))


def run_rewriter(path: Path) -> tuple[str, list[tuple]]:
    res = subprocess.run(
        [PYTHON, str(REWRITER), str(path)],
        capture_output=True, text=True, timeout=120,
    )
    out = res.stdout
    n_launch = len(re.findall(r"kernel\.launch", out))
    n_lg = len(re.findall(r"linalg\.generic", out))
    return out, [("launches", n_launch), ("residual_lg", n_lg)]


def build_kernel_page(kernel: str, mlir_dir: Path = MLIR_DIR,
                       kset: str = "polybench",
                       file_prefix: str = "") -> dict:
    raised = mlir_dir / f"{kernel}_linalg.mlir"
    debuf = mlir_dir / f"{kernel}_debuf.mlir"
    debuf_mr = mlir_dir / f"{kernel}_debuf_mr.mlir"

    pages: dict[str, str] = {}
    css = ""
    n_for = 0

    if raised.exists():
        html, css = syntax_highlight(raised.read_text())
        pages["raised"] = html
    if debuf.exists():
        debuf_text = debuf.read_text()
        n_for = count_for_loops(debuf_text)
        html, css = syntax_highlight(debuf_text)
        pages["debuf"] = html
        rewritten, report = run_rewriter(debuf)
        html, css = syntax_highlight(rewritten)
        pages["matched"] = html
    else:
        report = [("launches", 0), ("residual_lg", 0)]
    if debuf_mr.exists():
        html, css = syntax_highlight(debuf_mr.read_text())
        pages["debuf_mr"] = html

    ce_url = ce_link(kernel, mlir_dir=mlir_dir, kset=kset)
    open_link = (f'<a href="{ce_url}" target="_blank" '
                 f'style="margin-left:12px; color:#0366d6;">'
                 f'open in Compiler Explorer →</a>') if ce_url else ''

    n_launches = report[0][1]
    n_resid = report[1][1]
    summary = (
        f'<div class="summary" style="padding:8px 20px; '
        f'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
        f'<b>{n_launches}</b> kernel.launch op(s) emitted &nbsp;·&nbsp; '
        f'<b>{n_resid}</b> residual linalg.generic &nbsp;·&nbsp; '
        f'<b>{n_for}</b> residual for-loop(s) &nbsp;|&nbsp; '
        f'jump to: <a href="#raised">raised</a> · '
        f'<a href="#debuf">debuferized</a> · '
        f'<a href="#debuf_mr">debuf multi-root</a> · '
        f'<a href="#matched">kernel.launch output</a>'
        f'</div>'
    )
    header = (
        f'<div class="header"><h1><a href="index.html">← index</a> '
        f'&nbsp; {kernel}{open_link}</h1></div>'
        + summary
    )
    body_blocks = []
    for stage, title in [
        ("raised",   "raised (memref linalg, before debuferize)"),
        ("debuf",    "debuferized (tensor linalg, matcher input)"),
        ("debuf_mr", "debuferized — multi-root (--linalg-debufferize=use-multi-root=true)"),
        ("matched",  "kernel.launch (matcher output)"),
    ]:
        if stage not in pages:
            continue
        body_blocks.append(
            f'<h2 id="{stage}">{title}</h2>'
            f'<div class="container">{pages[stage]}</div>'
        )
    body = header + "\n".join(body_blocks)
    OUTPUT_DIR.joinpath(f"{file_prefix}{kernel}.html").write_text(render_html(kernel, body, css))
    return {
        "launches": report[0][1],
        "residual": report[1][1],
        "residual_for": n_for,
        "ce_url": ce_url,
        "page_filename": f"{file_prefix}{kernel}.html",
    }


def _render_section_rows(kernel_stats: dict[str, dict],
                          notes: dict[str, tuple[str, str]]) -> str:
    rows = []
    for k, s in sorted(kernel_stats.items()):
        l = s["launches"]; r = s["residual"]; f = s["residual_for"]
        if l > 0 and r == 0 and f == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        for_cls = "none" if f > 0 else "pass"

        if s["ce_url"]:
            kernel_link = f'<a class="kernel" href="{s["ce_url"]}" target="_blank">{k}</a>'
        else:
            kernel_link = f'<span class="nope">{k} (no source)</span>'

        note_tag, note_blurb = notes.get(k, ("", ""))
        tag_cls = {
            "highly parallel":   "pass",
            "parallel + T loop": "partial",
            "partial parallel":  "partial",
            "serial":            "none",
        }.get(note_tag, "")
        note_cell = (
            f'<td class="{tag_cls}" style="white-space:nowrap"><b>{note_tag}</b></td>'
            f'<td style="font-size:12px; color:#555">{note_blurb}</td>'
            if note_tag else '<td></td><td></td>'
        )

        page_file = s.get("page_filename", f"{k}.html")
        rows.append(
            f'<tr>'
            f'<td>{kernel_link}'
            f'<a class="viewer" href="{page_file}" style="margin-left:12px">[IR preview]</a>'
            f'</td>'
            f'<td>{l}</td><td>{r}</td><td class="{for_cls}">{f}</td>'
            f'<td class="{cls}">{status}</td>'
            f'{note_cell}'
            f'</tr>'
        )
    return "\n".join(rows)


def _build_section(title: str, anchor: str, blurb: str,
                    kernel_stats: dict[str, dict],
                    notes: dict[str, tuple[str, str]]) -> str:
    """Render one benchmark-suite section: a section header, blurb, then table."""
    rows_html = _render_section_rows(kernel_stats, notes)
    return (
        f'<a name="{anchor}"></a>'
        f'<div class="section-header"><h2 class="section-title">{title}</h2></div>'
        f'<div class="intro">{blurb}</div>'
        '<table><thead><tr>'
        '<th>kernel</th><th>kernel.launches</th>'
        '<th>residual linalg.generic</th>'
        '<th>residual for-loops</th>'
        '<th>match status</th>'
        '<th>parallelism</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        + rows_html +
        '</tbody></table>'
    )


def build_index(polybench_stats: dict[str, dict],
                 machsuite_stats: dict[str, dict],
                 npb_stats: dict[str, dict]) -> str:
    common_legend = (
        '  Click a kernel name to open the full Polygeist pipeline in '
        '  Compiler Explorer: C source on the left feeds cgeist; the affine '
        '  MLIR on the right feeds <code>polygeist-opt</code> with an '
        '  <em>Opt Pipeline</em> pane showing every internal pass. '
        '  The <code>[IR preview]</code> link opens a static snapshot of the '
        '  raised / debuferized / matcher-rewritten IR for that kernel.'
        '  The <em>residual for-loops</em> column counts imperative-loop ops '
        '  (<code>affine.for</code>, <code>scf.for</code>, '
        '  <code>scf.while</code>, <code>affine.parallel</code>, '
        '  <code>scf.parallel</code>) still present after raise + lower-submap '
        '  + debuferize — a measure of how much of the kernel remains '
        '  imperative rather than expressed as linalg / kernel.launch.'
        '  The <em>parallelism</em> column classifies the kernel by its GPU '
        '  suitability: <span class="pass"><b>highly parallel</b></span> '
        '  (every iter independent), <span class="partial"><b>parallel + T '
        '  loop</b></span> (body parallel, outer time loop serial — stencils), '
        '  <span class="partial"><b>partial parallel</b></span> (mixes '
        '  reductions / serial steps), <span class="none"><b>serial</b></span> '
        '  (cross-iter dependencies, poor naive GPU fit — factorizations, '
        '  recurrences, DPs).'
    )

    polybench_section = _build_section(
        title="PolyBench/C 4.2.1",
        anchor="polybench",
        blurb=(
            "30 numerical kernels from the PolyBench/C 4.2.1 benchmark — "
            "dense linear algebra, stencils, and data-mining bodies. " +
            common_legend
        ),
        kernel_stats=polybench_stats,
        notes=KERNEL_NOTES,
    )
    machsuite_section = _build_section(
        title="MachSuite",
        anchor="machsuite",
        blurb=(
            "19 kernels from the MachSuite accelerator-research benchmark — "
            "wider coverage than PolyBench (AES, sorting, FFT bit-reversal, "
            "SpMV, BFS, KMP, MD, Viterbi) at the cost of more kernels that "
            "fall outside the pipeline's affine sweet spot. Kernels marked "
            "<span class=\"nope\">(no source)</span> failed at the cgeist "
            "front-end (typically due to pointer- or bit-heavy C that cgeist "
            "doesn't model)."
        ),
        kernel_stats=machsuite_stats,
        notes=MACHSUITE_NOTES,
    )
    npb_section = _build_section(
        title="NPB (polybenchified)",
        anchor="npb",
        blurb=(
            "Selected kernels from NPB3.0-omp-C extracted into PolyBench-"
            "style single-file form (third_party/NPB-polybenchified/). The "
            "original NPB is one giant .c per benchmark with module-level "
            "static globals — cgeist can't isolate a single function from "
            "that layout. Each kernel here had its array dependencies "
            "rewritten as parameters so the pipeline can lift it. The "
            "results surface gaps that whole-file NPB didn't expose: "
            "indirect indexing (ft-evolve), scratch-row carries (MG "
            "stencils), and mixed sum+max reductions (norm2u3)."
        ),
        kernel_stats=npb_stats,
        notes=NPB_NOTES,
    )

    body = (
        '<div class="header"><h1>Polygeist IR explorer</h1>'
        '<div style="margin-top:6px; font-size:13px;">'
        '  Jump to: '
        '  <a href="#polybench">PolyBench</a> &middot; '
        '  <a href="#machsuite">MachSuite</a> &middot; '
        '  <a href="#npb">NPB (polybenchified)</a>'
        '</div></div>'
        + polybench_section
        + machsuite_section
        + npb_section
    )
    # Extra CSS for section headers.
    extra_css = (
        '.section-header { background: #eaeefa; padding: 8px 20px; '
        'border-top: 2px solid #c4cce0; border-bottom: 1px solid #c4cce0; '
        'margin-top: 24px; } '
        '.section-title { margin: 0; font-size: 16px; color: #1f2d3d; }'
    )
    return render_html("Polygeist IR explorer", body, extra_css)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PolyBench set.
    pb_kernels = discover_kernels(MLIR_DIR)
    print(f"Rendering {len(pb_kernels)} PolyBench kernels...", flush=True)
    pb_stats = {}
    for i, k in enumerate(pb_kernels, 1):
        print(f"  [PB {i:2d}/{len(pb_kernels)}] {k}", flush=True)
        pb_stats[k] = build_kernel_page(k, mlir_dir=MLIR_DIR,
                                         kset="polybench", file_prefix="")

    # MachSuite set.
    ms_kernels_from_files = discover_kernels(MACHSUITE_MLIR_DIR)
    # Also include kernels that have NO MLIR (cgeist failed) so they show as
    # "(no source)" entries with the explanatory parallelism note. We still
    # need them in the index to be honest about what the pipeline did/didn't
    # eat. They get an empty stats record below.
    ms_kernels = sorted(set(ms_kernels_from_files) | set(MACHSUITE_KERNELS.keys()))
    print(f"Rendering {len(ms_kernels)} MachSuite kernels...", flush=True)
    ms_stats = {}
    for i, k in enumerate(ms_kernels, 1):
        print(f"  [MS {i:2d}/{len(ms_kernels)}] {k}", flush=True)
        # If the kernel produced no MLIR files at all, fabricate a zero-stat
        # record so it still appears in the index (with no CE link).
        has_any = any((MACHSUITE_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            ms_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                            "ce_url": None, "page_filename": ""}
            continue
        ms_stats[k] = build_kernel_page(
            k, mlir_dir=MACHSUITE_MLIR_DIR, kset="machsuite",
            file_prefix="ms_",
        )

    # NPB-polybenchified set.
    npb_kernels_from_files = discover_kernels(NPB_MLIR_DIR)
    npb_kernels = sorted(set(npb_kernels_from_files) | set(NPB_KERNELS.keys()))
    print(f"Rendering {len(npb_kernels)} NPB kernels...", flush=True)
    npb_stats = {}
    for i, k in enumerate(npb_kernels, 1):
        print(f"  [NPB {i:2d}/{len(npb_kernels)}] {k}", flush=True)
        has_any = any((NPB_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            npb_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                             "ce_url": None, "page_filename": ""}
            continue
        npb_stats[k] = build_kernel_page(
            k, mlir_dir=NPB_MLIR_DIR, kset="npb",
            file_prefix="npb_",
        )

    OUTPUT_DIR.joinpath("index.html").write_text(
        build_index(pb_stats, ms_stats, npb_stats))
    print(f"\nDone. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
