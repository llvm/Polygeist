#!/usr/bin/env python3
"""Add a compiler-owned repeated-call lifetime around a void MLIR function.

The original function is marked as owned by the wrapper.  The public wrapper
adds one leading i32 repetition count and otherwise preserves its ABI.  This
is useful after separately raising a large noinline C helper: device-residency
planning can then own buffers for the complete repeated-call lifetime without
requiring CUDA types or allocation calls in the C source.
"""

import argparse
import re
from pathlib import Path


def split_arguments(arguments: str) -> list[str]:
    result: list[str] = []
    start = 0
    depths = {"<": 0, "(": 0, "[": 0, "{": 0}
    closing = {">": "<", ")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(arguments):
        if char in depths:
            depths[char] += 1
        elif char in closing:
            depths[closing[char]] -= 1
        elif char == "," and not any(depths.values()):
            result.append(arguments[start:index].strip())
            start = index + 1
    tail = arguments[start:].strip()
    if tail:
        result.append(tail)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("function")
    parser.add_argument("wrapper")
    parser.add_argument("output")
    args = parser.parse_args()

    text = Path(args.input).read_text()
    header = re.search(
        rf"(?m)^  func\.func @{re.escape(args.function)}\(([^\n]*)\)"
        rf"([^\n]*)\s*\{{$",
        text,
    )
    if header is None:
        raise SystemExit(f"function not found: {args.function}")
    suffix = header.group(2)
    if "->" in suffix:
        raise SystemExit("only void functions are supported")

    arguments = split_arguments(header.group(1))
    names: list[str] = []
    for argument in arguments:
        name = re.match(r"(%[\w.$-]+)\s*:", argument)
        if name is None:
            raise SystemExit(f"cannot parse argument: {argument}")
        names.append(name.group(1))

    owned_header = header.group(0)
    if " attributes {" in owned_header:
        owned_header = owned_header.replace(
            " attributes {",
            " attributes {polygeist.gpu_resident_callee, ",
            1,
        )
    else:
        owned_header = owned_header[:-1].rstrip() + \
            " attributes {polygeist.gpu_resident_callee} {"
    text = text[:header.start()] + owned_header + text[header.end():]

    final_brace = text.rfind("}")
    if final_brace < 0:
        raise SystemExit("module closing brace not found")
    wrapper_arguments = ", ".join(["%repetitions: i32", *arguments])
    call_arguments = ", ".join(names)
    call_types = ", ".join(argument.split(":", 1)[1].strip()
                           for argument in arguments)
    wrapper = f"""
  func.func @{args.wrapper}({wrapper_arguments}) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count = arith.index_cast %repetitions : i32 to index
    scf.for %iteration = %c0 to %count step %c1 {{
      func.call @{args.function}({call_arguments}) : ({call_types}) -> ()
    }}
    return
  }}
"""
    Path(args.output).write_text(text[:final_brace] + wrapper + text[final_brace:])


if __name__ == "__main__":
    main()
