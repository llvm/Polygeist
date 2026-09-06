#!/usr/bin/env python3
"""Generate a C-ABI wrapper for a PolyBench kernel.

The wrapper bridges PolyBench's C signature (int scalars, double scalars,
flat double* arrays) to the MLIR-lowered function which uses the bare
memref descriptor calling convention (each N-D memref expands to
[base, aligned, offset, sizes..., strides...] arguments).

Usage:
  gen_wrapper.py [--cuda-timing] <kernel.c> <kernel_name>

Prints the wrapper C source to stdout.
"""
import re
import sys


def extract_macro_prelude(c_text: str) -> str:
    """Copy simple #define constants needed by fixed-size plain C arrays."""
    lines = []
    for line in c_text.splitlines():
        m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)\b(.*)$", line)
        if not m:
            continue
        name = m.group(1)
        rest = m.group(2).strip()
        if "(" in name:
            continue
        if rest:
            lines.append(f"#ifndef {name}")
            lines.append(f"#define {name} {rest}")
            lines.append("#endif")
    return "\n".join(lines)


def infer_dtype(c_text: str) -> str:
    m = re.search(r"^\s*#\s*define\s+DATA_TYPE\s+(float|double)\b",
                  c_text, re.MULTILINE)
    if m:
        return m.group(1)
    if (re.search(r"\bfloat\s+[A-Za-z_]\w*\s*\[", c_text) or
            re.search(r"\b(?:const\s+)?float\s*\*\s*[A-Za-z_]\w*", c_text)):
        return "float"
    return "double"


def parse_signature(c_text: str, kernel_name: str):
    """Return list of (kind, *fields) tuples describing each argument.

    Kinds:
      ('char', name)
      ('int', name)
      ('double', name)
      ('1D', name, size_var)
      ('2D', name, d0_var, d1_var)
      ('3D', name, d0_var, d1_var, d2_var)
      ('4D', name, d0_var, d1_var, d2_var, d3_var)
      ... (plain C arrays support arbitrary positive rank)
    """
    # The signature can be split across many lines. Find the function head.
    m = re.search(
        rf"(?:void|int|DATA_TYPE|float|double)\s+{re.escape(kernel_name)}"
        rf"\s*\((.*?)\)\s*(?:\n)?\s*\{{",
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

    # Pointer-only extraction signatures lose their C array bounds.  Allow a
    # source to preserve those bounds without changing the function ABI:
    #   // polygeist-arg-extents function_name: A=20, X=128, Y=128
    extent_map = {}
    annotation = re.search(
        rf"^\s*//\s*polygeist-arg-extents\s+{re.escape(kernel_name)}\s*:\s*(.+)$",
        c_text,
        re.MULTILINE,
    )
    if annotation:
        for item in annotation.group(1).split(','):
            name, separator, extent = item.strip().partition('=')
            if not separator or not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise ValueError(f"Malformed pointer extent annotation: {item!r}")
            extent_map[name] = extent.strip()

    out = []
    plain_array_indices = []
    # Collect scalar extents before parsing pointers because C APIs commonly
    # place pointer operands before their trailing dimension arguments.
    scalar_ints = {
        a.split()[-1].strip('*') for a in args
        if re.match(r"^\s*(?:const\s+)?int\b", a)
    }
    for a in args:
        cartesian = re.match(
            r"^\s*(?:const\s+)?struct\s+cartesian\s*\*\s*(\w+)\s*$", a)
        if cartesian:
            name = cartesian.group(1)
            size = "n1" if name == "data1" else "n2"
            out.append(('CART3', name, size))
            continue
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
        elif _is_plain_c_array(a):
            # Plain C array signature: `double A[NI][NJ]` or `int A[NI][NJ][NK]`
            # — what polybenchGpu-extracted / llama2.c-style sources use
            # instead of POLYBENCH_2D/3D macros. We need (a) the variable name
            # and (b) one runtime-size arg per dimension. The uppercase macros
            # in the brackets (NI, NJ, NK) are compile-time constants; the
            # runtime sizes by convention live in lowercase int args of the
            # same function (ni, nj, nk). Match them by lowercasing the macro.
            kind, name, dims = _parse_plain_c_array(a)
            out.append((kind, name, *dims))
            plain_array_indices.append(len(out) - 1)
        elif re.match(r"^\s*char\b", a):
            name = a.split()[-1].strip('*')
            out.append(('char', name))
        elif re.match(r"^\s*(?:const\s+)?int\s+[A-Za-z_]\w*\s*$", a):
            name = a.split()[-1].strip('*')
            out.append(('int', name))
            scalar_ints.add(name)
        elif re.match(r"^\s*logical\s+[A-Za-z_]\w*\s*$", a):
            # NPB defines logical as a two-valued C enum.  Its external ABI is
            # the same scalar integer ABI used by the lifted MLIR function.
            name = a.split()[-1]
            out.append(('int', name))
        elif _is_plain_c_pointer(a):
            # Extracted kernels often use pointer signatures instead of fixed
            # C arrays. Infer the 1D memref extent from common scalar args.
            name, is_const, pointer_type = _parse_plain_c_pointer(a)
            if name in extent_map:
                size = extent_map[name]
            elif name == "tmor":
                # UA's transfb_cor_e/f receive a pointer to one selected
                # mortar scalar; `n` describes topology, not buffer length.
                size = "1"
            elif name == "out" and "n" in scalar_ints and "k" in scalar_ints:
                size = "(n - k + 1)"
            elif name in ("filter", "kernel", "weights") and "k" in scalar_ints:
                size = "k"
            elif name == "data_bins" and "nbins" in scalar_ints:
                size = "(nbins + 2)"
            elif name == "binb" and "nbins" in scalar_ints:
                size = "(nbins + 1)"
            elif {"nx", "ny", "nz"}.issubset(scalar_ints):
                size = "(nx * ny * nz)"
            elif "n" in scalar_ints:
                size = "n"
            elif "N" in c_text:
                size = "N"
            else:
                raise ValueError(f"Couldn't infer pointer extent for arg: {a}")
            prefix = ('I' if pointer_type == 'int' else
                      'L' if pointer_type in ('long long',
                                              'unsigned long long') else '')
            out.append((f'{prefix}1D', name, size))
        elif re.match(r"^\s*float\b", a):
            name = a.split()[-1].strip('*')
            out.append(('float', name))
        elif re.match(r"^\s*DATA_TYPE\b", a) or re.match(r"^\s*double\b", a):
            # Scalar (alpha, beta, etc.).
            name = a.split()[-1].strip('*')
            out.append(('double', name))
        else:
            raise ValueError(f"Unrecognized arg: {a}")

    for idx in plain_array_indices:
        entry = out[idx]
        dims = []
        for d in entry[2:]:
            # In a function parameter, `double a[]` is the same ABI as
            # `double *a`.  Legacy numerical C commonly supplies its extent
            # in a trailing `int n`; use that just as the pointer path does.
            if not d:
                if "n" not in scalar_ints:
                    raise ValueError(
                        f"Cannot infer empty leading dimension for {entry[1]!r}"
                    )
                dims.append("n")
                continue
            lower = d.lower()
            dims.append(lower if lower in scalar_ints else d)
        out[idx] = (entry[0], entry[1], *dims)
    return out


def parse_return_type(c_text: str, kernel_name: str, dtype: str) -> str:
    m = re.search(
        rf"\b(void|int|DATA_TYPE|float|double)\s+{re.escape(kernel_name)}\s*\(",
        c_text,
    )
    if not m:
        raise ValueError(f"Couldn't find function {kernel_name}")
    ret = m.group(1)
    return dtype if ret == "DATA_TYPE" else ret


def _is_plain_c_array(a: str) -> bool:
    """True iff `a` looks like a plain C array parameter declaration
    (e.g. 'double A[NI][NJ]' or 'int A[N]' or 'short A[NI][NJ][NK]').
    Distinguishable from a pointer-to-scalar (`double *alpha`) because
    array params always have a square-bracket dim list."""
    if not re.match(r"^\s*(?:const\s+)?(?:unsigned\s+char|signed\s+char|unsigned\s+int|unsigned|double|float|int|short|long|DATA_TYPE|_Float16|__bf16)\b", a):
        return False
    return re.search(r"\[\s*[^\]]*\s*\]\s*(?:\[\s*[^\]]*\s*\])*\s*$", a) is not None


def _parse_plain_c_array(a: str):
    """Parse a plain C array parameter like 'double A[NI][NJ]' or
    'short A[N]' into (kind, name, [dim0, dim1, ...]).
    `kind` is '1D', '2D', or '3D' so downstream gen_wrapper() can handle
    it identically to the POLYBENCH macro form.
    """
    m = re.match(
        r"^\s*(?:const\s+)?(unsigned\s+char|signed\s+char|unsigned\s+int|unsigned|double|float|int|short|long|DATA_TYPE|_Float16|__bf16)"
        r"\s+(\w+)((?:\s*\[\s*[^\]]*\s*\])+)\s*$",
        a,
    )
    if not m:
        raise ValueError(f"Couldn't parse plain-C-array arg: {a!r}")
    ctype, name = m.group(1), m.group(2)
    dims = [d.strip() for d in re.findall(r"\[\s*([^\]]*)\s*\]", m.group(3))]
    if not dims:
        raise ValueError(f"Plain-C-array arg has no dimensions: {a!r}")
    dims = ["(size + 1)" if not d and name == "lhs" else d for d in dims]
    prefix = ('U' if ctype == 'unsigned char' else
              'B' if ctype == 'signed char' else
              'I' if ctype in ('unsigned int', 'unsigned', 'int', 'short',
                               'long') else '')
    return (f'{prefix}{len(dims)}D', name, dims)


def _is_plain_c_pointer(a: str) -> bool:
    return re.match(
        r"^\s*(?:const\s+)?(?:int|double|float|long\s+long|unsigned\s+long\s+long|DATA_TYPE)\s*\*\s*\w+\s*$", a
    ) is not None


def _parse_plain_c_pointer(a: str):
    m = re.match(
        r"^\s*(const\s+)?(int|double|float|long\s+long|unsigned\s+long\s+long|DATA_TYPE)\s*\*\s*(\w+)\s*$", a
    )
    if not m:
        raise ValueError(f"Couldn't parse pointer arg: {a!r}")
    return m.group(3), bool(m.group(1)), m.group(2)


def gen_wrapper(kernel_name: str, args, dtype: str = 'double',
                prelude: str = '', return_type: str = 'void',
                cuda_timing: bool = False):
    """Emit wrapper C source for `kernel_name`."""
    extern_args, wrapper_args, call_args = [], [], []
    for a in args:
        k = a[0]
        if k == 'char':
            extern_args.append(f"char {a[1]}")
            wrapper_args.append(f"char {a[1]}")
            call_args.append(a[1])
        elif k == 'int':
            extern_args.append(f"int {a[1]}")
            wrapper_args.append(f"int {a[1]}")
            call_args.append(a[1])
        elif k == 'double':
            extern_args.append(f"{dtype} {a[1]}")
            wrapper_args.append(f"{dtype} {a[1]}")
            call_args.append(a[1])
        elif k == 'float':
            extern_args.append(f"float {a[1]}")
            wrapper_args.append(f"float {a[1]}")
            call_args.append(a[1])
        elif k == 'CART3':
            name, size = a[1], a[2]
            extern_args.extend([
                f"float *{name}_b", f"float *{name}_a",
                f"int64_t {name}_off", f"int64_t {name}_s0",
                f"int64_t {name}_s1", f"int64_t {name}_t0",
                f"int64_t {name}_t1",
            ])
            wrapper_args.append(f"struct cartesian *{name}")
            call_args.append(
                f"(float *){name}, (float *){name}, 0, {size}, 3, 3, 1")
        elif re.fullmatch(r'(?:I|U|B|L)?[1-9][0-9]*D', k):
            is_integer = k.startswith('I')
            is_unsigned_byte = k.startswith('U')
            is_signed_byte = k.startswith('B')
            is_long_long = k.startswith('L')
            rank = int(k[1:-1] if (is_integer or is_unsigned_byte or
                                   is_signed_byte or is_long_long) else k[:-1])
            name = a[1]
            dims = list(a[2:])
            if len(dims) != rank:
                raise ValueError(
                    f"{k} argument {name} has {len(dims)} dimensions")
            arg_dtype = ("int" if is_integer else
                         "unsigned char" if is_unsigned_byte else dtype)
            if is_signed_byte:
                arg_dtype = "signed char"
            if is_long_long:
                arg_dtype = "long long"
            extern_args.extend([
                f"{arg_dtype} *{name}_b", f"{arg_dtype} *{name}_a",
                f"int64_t {name}_off",
                *(f"int64_t {name}_s{i}" for i in range(rank)),
                *(f"int64_t {name}_t{i}" for i in range(rank)),
            ])
            wrapper_args.append(f"{arg_dtype} *{name}")
            strides = []
            for i in range(rank):
                trailing = dims[i + 1:]
                strides.append(
                    " * ".join(f"({d})" for d in trailing) if trailing else "1"
                )
            descriptor = [name, name, "0", *dims, *strides]
            call_args.append(", ".join(descriptor))
        else:
            raise ValueError(f"Unknown kind {k}")

    extern = (
        f"extern {return_type} {kernel_name}_impl(\n    "
        + ",\n    ".join(extern_args)
        + ");"
    )
    call = (
        f"{kernel_name}_impl(\n      "
        + ",\n      ".join(call_args)
        + ")"
    )
    if cuda_timing and return_type != 'void':
        raise ValueError("CUDA timing wrapper currently requires a void kernel")
    if cuda_timing:
        body = (
            "  polygeist_cublas_time_begin();\n"
            f"  {call};\n"
            "  double polygeist_device_ms = polygeist_cublas_time_end_ms();\n"
            f'  fprintf(stderr, "POLYGEIST_DEVICE_TIMING kernel={kernel_name} device_ms=%.6f\\n", polygeist_device_ms);'
        )
    elif return_type == 'void':
        body = f"  {call};"
    else:
        body = f"  return {call};"
    wrapper = (
        f"{return_type} {kernel_name}({', '.join(wrapper_args)}) {{\n"
        f"{body}\n}}"
    )
    prefix = "#include <stdint.h>"
    if cuda_timing:
        prefix += ("\n#include <stdio.h>\n"
                   "extern void polygeist_cublas_time_begin(void);\n"
                   "extern double polygeist_cublas_time_end_ms(void);")
    if any(a[0] == 'CART3' for a in args):
        prefix += "\nstruct cartesian { float x, y, z; };"
    if prelude:
        prefix += "\n" + prelude
    return f"{prefix}\n\n{extern}\n\n{wrapper}\n"


def main():
    cuda_timing = "--cuda-timing" in sys.argv[1:]
    positional = [arg for arg in sys.argv[1:] if arg != "--cuda-timing"]
    if len(positional) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    src, name = positional
    with open(src) as f:
        text = f.read()
    dtype = infer_dtype(text)
    args = parse_signature(text, name)
    ret = parse_return_type(text, name, dtype)
    print(gen_wrapper(name, args, dtype, extract_macro_prelude(text), ret,
                      cuda_timing))


if __name__ == "__main__":
    main()
