#!/usr/bin/env python3
"""Reverse the kernel-match rewrite: restore each `kernel.launch` op back to
the original `linalg.generic` span the matcher recognized.

This is the round-trip Phase-1 lowering for `kernel.launch`. It consumes MLIR
text emitted by `kernel_match_rewrite.py --with-roundtrip-markers` and emits
MLIR with the kernel.launch ops swapped back for their pre-match form, so the
result is parseable by `polygeist-opt` and can flow on to LLVM lowering and
execution. Used by the kernel-launch e2e correctness tests.

Each rewritten site looks like

    // POLYGEIST-MATCH-BEGIN-<libop>
    //   <indented original linalg.generic(s)>
    // POLYGEIST-MATCH-END
    %X = kernel.launch @<libop>(...) : (...) -> <type>

We replace that entire region with the captured original span.

Usage:
  kernel_launch_lower.py <input.mlir>             # write to stdout
  kernel_launch_lower.py <input.mlir> -o <out>    # write to a file

Phase-2 ("canonical templates") will swap each `kernel.launch` for a fresh
linalg.generic synthesised from the library entry rather than the stashed
original, so the matcher's LABELS are also validated. Not in this script.
"""
import argparse
import re
import sys
from pathlib import Path


# (?ms): multiline + dotall. We deliberately avoid `re.M` here so the
# leading-indent group also matches across leading newlines.
_BLOCK_RE = re.compile(
    r"^([ \t]*)// POLYGEIST-MATCH-BEGIN-(\w+)\s*\n"   # marker open
    r"((?:^[ \t]*//[^\n]*\n)+?)"                       # captured comment body
    r"^[ \t]*// POLYGEIST-MATCH-END[ \t]*\n"            # marker close
    r"^[ \t]*[%\w]+\s*=\s*kernel\.launch @[^\n]*\n",  # the kernel.launch line
    re.MULTILINE,
)


def _strip_comment_prefix(body: str, indent: str) -> str:
    """Strip `<indent>// ` from each captured line, restoring the original."""
    # Each line is either `<indent>// <stuff>` or `<indent>//` for blanks.
    prefix_re = re.compile(rf"^{re.escape(indent)}//[ \t]?", re.MULTILINE)
    return prefix_re.sub("", body)


def lower_text(text: str) -> tuple[str, int]:
    """Return (lowered_text, n_blocks_restored)."""
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        indent = m.group(1)
        body = m.group(3)
        return _strip_comment_prefix(body, indent)

    return _BLOCK_RE.sub(repl, text), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="MLIR with kernel.launch + match markers.")
    ap.add_argument("-o", "--output", help="Write to file (default: stdout).")
    args = ap.parse_args()

    src = Path(args.input).read_text()
    out, n = lower_text(src)
    if n == 0:
        print(
            "kernel_launch_lower: warning — no POLYGEIST-MATCH markers found. "
            "Run kernel_match_rewrite.py with --with-roundtrip-markers.",
            file=sys.stderr,
        )

    if args.output:
        Path(args.output).write_text(out)
    else:
        sys.stdout.write(out)
    print(f"kernel_launch_lower: restored {n} kernel.launch op(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
