#!/usr/bin/env python3
"""Generate a C-ABI wrapper for a PolyBench kernel.

The wrapper bridges PolyBench's C signature (int scalars, double scalars,
flat double* arrays) to the MLIR-lowered function which uses the bare
memref descriptor calling convention (each N-D memref expands to
[base, aligned, offset, sizes..., strides...] arguments).

Usage:
  gen_wrapper.py <kernel.c> <kernel_name>

Prints the wrapper C source to stdout.
"""
import re
import sys


def parse_signature(c_text: str, kernel_name: str):
    """Return list of (kind, *fields) tuples describing each argument.

    Kinds:
      ('int', name)
      ('double', name)
      ('1D', name, size_var)
      ('2D', name, d0_var, d1_var)
      ('3D', name, d0_var, d1_var, d2_var)
    """
    # The signature can be split across many lines. Find the function head.
    m = re.search(
        rf"void\s+{re.escape(kernel_name)}\s*\((.*?)\)\s*(?:\n)?\s*\{{",
        c_text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"Couldn't find function {kernel_name}")
    args_str = m.group(1)
    # Split by top-level commas (respecting nested parens).
    args, depth, cur = [], 0, []
    for c in args_str:
        if c == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        cur.append(c)
    args.append(''.join(cur).strip())

    out = []
    for a in args:
        if 'POLYBENCH_3D' in a:
            m3 = re.search(
                r"POLYBENCH_3D\s*\(\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,"
                r"\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)",
                a,
            )
            if not m3:
                raise ValueError(f"Couldn't parse 3D arg: {a}")
            out.append(('3D', m3.group(1), m3.group(2), m3.group(3), m3.group(4)))
        elif 'POLYBENCH_2D' in a:
            m2 = re.search(
                r"POLYBENCH_2D\s*\(\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*,"
                r"\s*(\w+)\s*,\s*(\w+)\s*\)",
                a,
            )
            if not m2:
                raise ValueError(f"Couldn't parse 2D arg: {a}")
            out.append(('2D', m2.group(1), m2.group(2), m2.group(3)))
        elif 'POLYBENCH_1D' in a:
            m1 = re.search(
                r"POLYBENCH_1D\s*\(\s*(\w+)\s*,\s*\w+\s*,\s*(\w+)\s*\)", a
            )
            if not m1:
                raise ValueError(f"Couldn't parse 1D arg: {a}")
            out.append(('1D', m1.group(1), m1.group(2)))
        elif re.match(r"^\s*int\b", a):
            name = a.split()[-1].strip('*')
            out.append(('int', name))
        elif re.match(r"^\s*DATA_TYPE\b", a) or re.match(r"^\s*float\b", a) \
                or re.match(r"^\s*double\b", a):
            # Scalar (alpha, beta, etc.).
            name = a.split()[-1].strip('*')
            out.append(('double', name))
        else:
            raise ValueError(f"Unrecognized arg: {a}")
    return out


def gen_wrapper(kernel_name: str, args, dtype: str = 'double'):
    """Emit wrapper C source for `kernel_name`."""
    extern_args, wrapper_args, call_args = [], [], []
    for a in args:
        k = a[0]
        if k == 'int':
            extern_args.append(f"int {a[1]}")
            wrapper_args.append(f"int {a[1]}")
            call_args.append(a[1])
        elif k == 'double':
            extern_args.append(f"{dtype} {a[1]}")
            wrapper_args.append(f"{dtype} {a[1]}")
            call_args.append(a[1])
        elif k == '1D':
            name, sz = a[1], a[2]
            extern_args.extend([
                f"{dtype} *{name}_b", f"{dtype} *{name}_a",
                f"int64_t {name}_off", f"int64_t {name}_s0", f"int64_t {name}_t0",
            ])
            wrapper_args.append(f"{dtype} *{name}")
            call_args.append(f"{name}, {name}, 0, {sz}, 1")
        elif k == '2D':
            name, d0, d1 = a[1], a[2], a[3]
            extern_args.extend([
                f"{dtype} *{name}_b", f"{dtype} *{name}_a",
                f"int64_t {name}_off",
                f"int64_t {name}_s0", f"int64_t {name}_s1",
                f"int64_t {name}_t0", f"int64_t {name}_t1",
            ])
            wrapper_args.append(f"{dtype} *{name}")
            call_args.append(f"{name}, {name}, 0, {d0}, {d1}, {d1}, 1")
        elif k == '3D':
            name, d0, d1, d2 = a[1], a[2], a[3], a[4]
            extern_args.extend([
                f"{dtype} *{name}_b", f"{dtype} *{name}_a",
                f"int64_t {name}_off",
                f"int64_t {name}_s0", f"int64_t {name}_s1", f"int64_t {name}_s2",
                f"int64_t {name}_t0", f"int64_t {name}_t1", f"int64_t {name}_t2",
            ])
            wrapper_args.append(f"{dtype} *{name}")
            # Row-major stride: t0 = d1*d2, t1 = d2, t2 = 1.
            call_args.append(
                f"{name}, {name}, 0, {d0}, {d1}, {d2}, ({d1}) * ({d2}), {d2}, 1"
            )
        else:
            raise ValueError(f"Unknown kind {k}")

    extern = (
        f"extern void {kernel_name}_impl(\n    "
        + ",\n    ".join(extern_args)
        + ");"
    )
    wrapper = (
        f"void {kernel_name}({', '.join(wrapper_args)}) {{\n"
        f"  {kernel_name}_impl(\n      "
        + ",\n      ".join(call_args)
        + ");\n}"
    )
    return f"#include <stdint.h>\n\n{extern}\n\n{wrapper}\n"


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    src, name = sys.argv[1], sys.argv[2]
    with open(src) as f:
        text = f.read()
    args = parse_signature(text, name)
    print(gen_wrapper(name, args))


if __name__ == "__main__":
    main()
