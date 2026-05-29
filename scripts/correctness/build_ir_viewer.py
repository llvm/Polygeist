#!/usr/bin/env python3
"""Render all PolyBench IR stages as a static-HTML browse-able site.

For each kernel we expose:
  1. raised-linalg (memref form, before debuferize)
  2. debuferized (tensor form, the input to the matcher) — default v2 path
  3. debuferized — multi-root (--linalg-debufferize=use-multi-root=true)
  4. kernel-launches (the matcher's rewritten output)

Plus an index page that links to all kernels and shows match stats.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

SCRIPT_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, str(default)))


POLYBENCH_DIR = env_path("POLYGEIST_POLYBENCH_MLIR_DIR", "/tmp/polybench_new")
OUTPUT_DIR = env_path("POLYGEIST_IR_VIEWER_OUT", "/tmp/ir_viewer")
REWRITER = env_path("POLYGEIST_KERNEL_MATCH_REWRITER", SCRIPT_DIR / "kernel_match_rewrite.py")
PYTHON = os.environ.get("PYTHON", sys.executable)


def discover_kernels() -> list[str]:
    return sorted(
        f.stem.replace("_debuf", "")
        for f in POLYBENCH_DIR.glob("*_debuf.mlir")
    )


def render_html(title: str, body_html: str, css: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #1e1e1e; color: #d4d4d4; }}
  .header {{ background: #2d2d2d; padding: 10px 20px; border-bottom: 1px solid #444; }}
  .header a {{ color: #58a6ff; text-decoration: none; margin-right: 12px; }}
  .header a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 18px; margin: 0; }}
  .tabs {{ display: flex; gap: 0; background: #252525; border-bottom: 1px solid #444; }}
  .tab {{ padding: 8px 16px; color: #aaa; text-decoration: none; }}
  .tab.active {{ background: #1e1e1e; color: #58a6ff; border-bottom: 2px solid #58a6ff; }}
  .container {{ padding: 16px 20px; }}
  pre {{ margin: 0; font-size: 13px; line-height: 1.4; overflow-x: auto; }}
  {css}
  table {{ border-collapse: collapse; }}
  td, th {{ padding: 6px 12px; border-bottom: 1px solid #333; }}
  th {{ text-align: left; color: #aaa; font-weight: normal; font-size: 12px; }}
  .pass {{ color: #4ec9b0; }} .partial {{ color: #ce9178; }} .none {{ color: #f48771; }}
</style></head>
<body>{body_html}</body></html>
"""


def syntax_highlight(text: str, lang: str = "llvm") -> tuple[str, str]:
    text = re.sub(r"#dlti\.dl_spec<[^>]*>", "(dlti spec hidden)", text)
    lexer = get_lexer_by_name(lang)
    fmt = HtmlFormatter(style="monokai", nobackground=True)
    return highlight(text, lexer, fmt), fmt.get_style_defs(".highlight")


def run_rewriter(path: Path) -> tuple[str, list[tuple]]:
    """Run the kernel-match rewriter on the file."""
    res = subprocess.run(
        [PYTHON, str(REWRITER), str(path)],
        capture_output=True, text=True, timeout=120,
    )
    out = res.stdout
    n_launch = len(re.findall(r"kernel\.launch", out))
    n_lg = len(re.findall(r"linalg\.generic", out))
    report = [("launches", n_launch), ("residual_lg", n_lg)]
    return out, report


def build_kernel_page(kernel: str) -> dict:
    """Build all four stage pages plus return summary stats."""
    raised = POLYBENCH_DIR / f"{kernel}_linalg.mlir"
    debuf = POLYBENCH_DIR / f"{kernel}_debuf.mlir"
    debuf_mr = POLYBENCH_DIR / f"{kernel}_debuf_mr.mlir"

    pages: dict[str, str] = {}
    css = ""

    if raised.exists():
        html, css = syntax_highlight(raised.read_text())
        pages["raised"] = html
    if debuf.exists():
        html, css = syntax_highlight(debuf.read_text())
        pages["debuf"] = html

        rewritten, report = run_rewriter(debuf)
        html, css = syntax_highlight(rewritten)
        pages["matched"] = html
    else:
        report = [("launches", 0), ("residual_lg", 0)]
    if debuf_mr.exists():
        html, css = syntax_highlight(debuf_mr.read_text())
        pages["debuf_mr"] = html

    # Combine into one tabs page.
    header = (
        f'<div class="header"><h1><a href="index.html">← index</a> '
        f'&nbsp; {kernel}</h1></div>'
    )
    tabs_html = '<div class="tabs">'
    body_html_blocks = []
    for stage, title in [
        ("raised",   "raised (memref linalg)"),
        ("debuf",    "debuferized (tensor linalg, matcher input)"),
        ("debuf_mr", "debuferized — multi-root"),
        ("matched",  "kernel.launch (matcher output)"),
    ]:
        if stage not in pages:
            continue
        anchor = stage
        tabs_html += f'<a class="tab" href="#{anchor}">{title}</a>'
        body_html_blocks.append(
            f'<a name="{anchor}"></a><h2 style="margin-top:24px">{title}</h2>'
            f'<div class="container">{pages[stage]}</div>'
        )
    tabs_html += '</div>'
    body = header + tabs_html + "\n".join(body_html_blocks)
    OUTPUT_DIR.joinpath(f"{kernel}.html").write_text(render_html(kernel, body, css))

    return {"launches": report[0][1], "residual": report[1][1]}


def build_index(kernel_stats: dict[str, dict]) -> str:
    rows = []
    for k, s in sorted(kernel_stats.items()):
        l = s["launches"]; r = s["residual"]
        if l > 0 and r == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        rows.append(f'<tr><td><a href="{k}.html">{k}</a></td>'
                    f'<td>{l}</td><td>{r}</td>'
                    f'<td class="{cls}">{status}</td></tr>')
    body = (
        '<div class="header"><h1>PolyBench IR explorer</h1></div>'
        '<div class="container">'
        '<p>Click a kernel to inspect its raised / debuferized / kernel.launch IRs.</p>'
        '<table><thead><tr><th>kernel</th><th>kernel.launches</th>'
        '<th>residual linalg.generic</th><th>match status</th></tr></thead>'
        '<tbody>' + "\n".join(rows) + '</tbody></table></div>'
    )
    return render_html("PolyBench IR explorer", body, "")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kernels = discover_kernels()
    print(f"Rendering {len(kernels)} kernels into {OUTPUT_DIR}...", flush=True)
    stats = {}
    for i, k in enumerate(kernels, 1):
        print(f"  [{i:2d}/{len(kernels)}] {k}", flush=True)
        stats[k] = build_kernel_page(k)
    OUTPUT_DIR.joinpath("index.html").write_text(build_index(stats))
    print(f"\nDone. Open {OUTPUT_DIR}/index.html or serve {OUTPUT_DIR} via HTTP.")


if __name__ == "__main__":
    main()
