#!/usr/bin/env python3
"""Prepend kernel.defn ops from a kernel library file into an input module so
the kernel.launch ops it contains pass MLIR's symbol verification at parse
time. Used by the Phase-2 e2e pipeline before running --lower-kernel-launch.

Usage:
  inject_kernel_library.py <input.mlir> <library.mlir> -o <out.mlir>
"""
import argparse
import re
import sys
from pathlib import Path


def find_module_body_open(text: str) -> int:
    """Return the offset of the `{` that opens the top-level module's body.

    Handles both `module {` and `module attributes {...} {`. We scan for the
    `module` keyword, then walk braces tracking depth — the body `{` is the
    first `{` at depth 0 AFTER the keyword. Attribute-dict `{}`'s pair up
    cleanly so they cancel out and don't perturb the depth tally.
    """
    m = re.search(r"\bmodule\b", text)
    if not m:
        raise ValueError("no `module` keyword found")
    i = m.end()
    depth = 0
    while i < len(text):
        c = text[i]
        if c == '{':
            if depth == 0:
                # If this `{` is preceded (skipping ws) by `attributes`, it's
                # the attr-dict opener — descend so its matching `}` decrements.
                preceding = text[m.end():i].rstrip()
                if preceding.endswith("attributes"):
                    depth += 1
                    i += 1
                    continue
                return i
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    raise ValueError("did not find module body `{`")


def extract_module_body(text: str) -> str:
    """Return contents between module body `{` and the final `}`."""
    body_open = find_module_body_open(text)
    end = text.rindex("}")
    return text[body_open + 1 : end]


def inject(input_text: str, library_text: str) -> str:
    """Splice library defns into the input module's top-level block."""
    lib_body = extract_module_body(library_text).strip()
    insert_at = find_module_body_open(input_text) + 1
    return input_text[:insert_at] + "\n" + lib_body + "\n" + input_text[insert_at:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("library")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    inp = Path(args.input).read_text()
    lib = Path(args.library).read_text()
    Path(args.output).write_text(inject(inp, lib))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
