"""Exercise typed BLAS lowering with valid and deliberately invalid contracts."""
import subprocess
import sys


optimizer = sys.argv[1]
checks = 0
for dtype, prefix in (("f64", "D"), ("f32", "S")):
    for transpose in (False, True):
        symbol = "cublas" + prefix + "gemv_alpha" + ("_T" if transpose else "")
        vector = "17" if transpose else "19"
        output = "19" if transpose else "17"
        types = [f"tensor<17x19x{dtype}>", f"tensor<{vector}x{dtype}>",
                 f"tensor<{output}x{dtype}>", dtype]
        result = types[2]
        cases = [("valid", types, result, None)]
        for index, replacement, diagnostic in (
                (0, types[0].replace(dtype, "f32" if dtype == "f64" else "f64"), "matrix type"),
                (0, f"tensor<17x19x1x{dtype}>", "matrix type"),
                (0, types[0].replace("tensor", "memref"), "matrix type"),
                (1, f"tensor<1x{vector}x{dtype}>", "vector tensors"),
                (1, f"tensor<23x{dtype}>", "extents"),
                (2, f"tensor<23x{dtype}>", "vector tensors"),
                (3, "f32" if dtype == "f64" else "f64", "alpha type mismatch")):
            changed = list(types)
            changed[index] = replacement
            cases.append((f"operand-{index}-{replacement}", changed, result, diagnostic))
        cases += [("missing-alpha", types[:3], result, "expected 4 operands"),
                  ("extra-alpha", types + [dtype], result, "expected 4 operands"),
                  ("missing-result", types, None, "one result")]
        for name, operands, result_type, diagnostic in cases:
            args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(operands))
            names = ", ".join(f"%a{i}" for i in range(len(operands)))
            result_sig = " -> " + result_type if result_type else ""
            call_result = result_type or "()"
            yield_line = f"kernel.yield %a2 : {result_type}" if result_type else "kernel.yield"
            # A mismatched output type in this case needs its own correctly
            # typed definition result, leaving the lowering to check x/y.
            if result_type and operands[2] != result_type:
                yield_line = f"%z = tensor.empty() : {result_type}\n kernel.yield %z : {result_type}"
            text = f"""module {{
 kernel.defn @{symbol}({args}){result_sig} {{ {yield_line} }}
 func.func @unrelated_name({args}){result_sig} {{
  {'%r = ' if result_type else ''}kernel.launch @{symbol}({names}) :
    ({', '.join(operands)}) -> {call_result}
  {'return %r : ' + result_type if result_type else 'return'}
 }}
}}"""
            run = subprocess.run([optimizer, "--lower-kernel-launch-to-cublas"],
                                 input=text, capture_output=True, text=True)
            if diagnostic is None:
                expected = "polygeist_cublas_" + prefix.lower() + "gemv" + ("_T" if transpose else "")
                assert run.returncode == 0 and "call @" + expected + "(" in run.stdout, (name, run.stderr)
            else:
                assert run.returncode > 0 and diagnostic in run.stderr, (name, run.returncode, run.stderr)
            checks += 1
print(f"PASS {checks} typed GEMV ABI checks")
