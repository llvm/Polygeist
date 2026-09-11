#!/usr/bin/env python3
"""Add a target-neutral repeated-call lifetime boundary to an MLIR module.

The generated function contains only an SCF loop and calls the selected void
function.  Accelerator residency passes can therefore promote inputs and
scratch across the complete repeated session without requiring CUDA code in
the source program or benchmark harness.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def split_top_level(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def module_closing_brace(text: str) -> int:
    module = re.search(r"^\s*module(?:\s+attributes\s+.*)?\s*\{", text,
                       re.MULTILINE)
    if not module:
        raise ValueError("could not find the top-level module")
    opening = text.find("{", module.start(), module.end())
    depth = 0
    last_top_level_closing = -1
    in_string = False
    escaped = False
    in_line_comment = False
    for index in range(opening, len(text)):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == "/" and following == "/":
            in_line_comment = True
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                last_top_level_closing = index
    if last_top_level_closing >= 0:
        return last_top_level_closing
    raise ValueError("module has no closing brace")


def add_session(module: str, function: str, session: str) -> str:
    if re.search(rf"^\s*func\.func\s+@{re.escape(session)}\b", module,
                 re.MULTILINE):
        raise ValueError(f"function @{session} already exists")
    match = re.search(
        rf"^\s*func\.func\s+@{re.escape(function)}\((.*?)\)"
        rf"(?P<suffix>\s*(?:attributes\s*\{{[^}}]*\}}\s*)?)\{{",
        module,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        raise ValueError(f"could not find a definition of @{function}")
    arguments = split_top_level(match.group(1))
    names: list[str] = []
    types: list[str] = []
    for argument in arguments:
        arg_match = re.fullmatch(r"\s*(%[A-Za-z0-9_.$-]+)\s*:\s*(.+?)\s*", argument,
                                 re.DOTALL)
        if not arg_match:
            raise ValueError(f"unsupported function argument: {argument}")
        names.append(arg_match.group(1))
        types.append(arg_match.group(2))

    closing = module_closing_brace(module)
    forwarded_args = ", ".join(arguments)
    if forwarded_args:
        forwarded_args = ", " + forwarded_args
    operands = ", ".join(names)
    operand_types = ", ".join(types)
    call = f"      func.call @{function}({operands}) : ({operand_types}) -> ()\n"
    generated = (
        f"\n  func.func @{session}(%repetitions: i32{forwarded_args}) {{\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        "    %count = arith.index_cast %repetitions : i32 to index\n"
        "    scf.for %iteration = %c0 to %count step %c1 {\n"
        f"{call}"
        "    }\n"
        "    return\n"
        "  }\n"
    )
    return module[:closing] + generated + module[closing:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--session")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    session = args.session or args.function + "_session"
    args.output.write_text(add_session(args.input.read_text(), args.function, session))


if __name__ == "__main__":
    main()
