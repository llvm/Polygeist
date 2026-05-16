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
OUTPUT_DIR = Path("/tmp/ir_viewer")
REWRITER = Path("/home/arjaiswal/Polygeist/scripts/correctness/kernel_match_rewrite.py")
PYTHON = "/home/arjaiswal/slacker/.venv/bin/python3"

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


def find_kernel_c(name: str) -> Path | None:
    """Find <name>.c under polybench/, excluding utilities and *.orig.c."""
    for p in POLYBENCH_TEST_DIR.rglob(f"{name}.c"):
        if "/utilities/" in str(p):
            continue
        if p.name.endswith(".orig.c"):
            continue
        return p
    return None


def discover_kernels() -> list[str]:
    return sorted(
        f.stem.replace("_debuf", "")
        for f in MLIR_DIR.glob("*_debuf.mlir")
    )


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


def ce_link(kernel: str) -> str | None:
    """Construct the CE deep-link URL for a kernel; None if sources missing."""
    c_path = find_kernel_c(kernel)
    mlir_path = MLIR_DIR / f"{kernel}.mlir"
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


def build_kernel_page(kernel: str) -> dict:
    raised = MLIR_DIR / f"{kernel}_linalg.mlir"
    debuf = MLIR_DIR / f"{kernel}_debuf.mlir"
    debuf_mr = MLIR_DIR / f"{kernel}_debuf_mr.mlir"

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

    ce_url = ce_link(kernel)
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
    OUTPUT_DIR.joinpath(f"{kernel}.html").write_text(render_html(kernel, body, css))
    return {
        "launches": report[0][1],
        "residual": report[1][1],
        "residual_for": n_for,
        "ce_url": ce_url,
    }


def build_index(kernel_stats: dict[str, dict]) -> str:
    rows = []
    for k, s in sorted(kernel_stats.items()):
        l = s["launches"]; r = s["residual"]; f = s["residual_for"]
        # FULL = every linalg.generic was matched AND no imperative for-loop
        # remains (so the kernel is entirely in kernel.launch + arith/SSA form).
        # PARTIAL = matcher fired at least once but something is left behind
        # (either a residual linalg.generic or an outer for-loop the raise
        # pass never lifted).
        # NONE   = matcher never fired.
        if l > 0 and r == 0 and f == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        # Highlight high-loop-residual kernels as imperative-form holdouts.
        for_cls = "none" if f > 0 else "pass"

        if s["ce_url"]:
            kernel_link = f'<a class="kernel" href="{s["ce_url"]}" target="_blank">{k}</a>'
        else:
            kernel_link = f'<span class="nope">{k} (no source)</span>'

        note_tag, note_blurb = KERNEL_NOTES.get(k, ("", ""))
        # Colour-code the parallelism tag.
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

        rows.append(
            f'<tr>'
            f'<td>{kernel_link}'
            f'<a class="viewer" href="{k}.html" style="margin-left:12px">[IR preview]</a>'
            f'</td>'
            f'<td>{l}</td><td>{r}</td><td class="{for_cls}">{f}</td>'
            f'<td class="{cls}">{status}</td>'
            f'{note_cell}'
            f'</tr>'
        )
    body = (
        '<div class="header"><h1>Polygeist — PolyBench IR explorer</h1></div>'
        '<div class="intro">'
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
        '</div>'
        '<table><thead><tr>'
        '<th>kernel</th><th>kernel.launches</th>'
        '<th>residual linalg.generic</th>'
        '<th>residual for-loops</th>'
        '<th>match status</th>'
        '<th>parallelism</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
    )
    return render_html("Polygeist IR explorer", body, "")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kernels = discover_kernels()
    print(f"Rendering {len(kernels)} kernels into {OUTPUT_DIR}...", flush=True)
    stats = {}
    for i, k in enumerate(kernels, 1):
        print(f"  [{i:2d}/{len(kernels)}] {k}", flush=True)
        stats[k] = build_kernel_page(k)
    OUTPUT_DIR.joinpath("index.html").write_text(build_index(stats))
    print(f"\nDone. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
