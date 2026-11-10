import pathlib
import re
import subprocess
import sys
import tempfile

optimizer, matcher, fixture = sys.argv[1:]
original = pathlib.Path(fixture).read_text()


def match(text, directory, label):
    path = pathlib.Path(directory) / (label + ".mlir")
    parsed = subprocess.run([optimizer], input=text, text=True,
                            capture_output=True, check=True).stdout
    path.write_text(parsed)
    return subprocess.run(["/usr/bin/python3", matcher, str(path)], text=True,
                          capture_output=True, check=True).stdout


with tempfile.TemporaryDirectory() as directory:
    first = match(original, directory, "original")
    renamed = match(original.replace("arbitrary_", "unrelated_").replace(
        "renamed_float_", "another_float_"), directory, "renamed")
    calls = lambda text: re.findall(r"kernel.launch[^\n]+", text)
    assert calls(first) == calls(renamed), "Recognition depends on function names"
    literal = original.replace("%s = arith.mulf %alpha, %p : f64",
                               "%two = arith.constant 2.0 : f64\n      %s = arith.mulf %two, %p : f64")
    scaled = match(literal, directory, "literal")
    assert "kernel.launch @cublasDgemv_alpha_T(" in scaled
    assert re.search(r"arith.constant 2\.0+(?:e[+]0+)? : f64", scaled)
    # Omitting alpha is a unit coefficient, not permission to omit an ABI
    # operand. An unscaled GEMV or a scaled GEMV with a materialized 1 is valid.
    unit = original.replace("%s = arith.mulf %alpha, %p : f64", "")
    unit = unit.replace("%out, %s : f64", "%out, %p : f64")
    unscaled = match(unit, directory, "unit")
    launches = re.findall(r"kernel.launch @cublasDgemv(?:_alpha)?_T\(([^)]*)\)", unscaled)
    assert launches, "Unit coefficient was not recognized"
    for line in calls(unscaled):
        if "@cublasDgemv_alpha" in line:
            assert len(line.split("(", 1)[1].split(")", 1)[0].split(",")) == 4
print("PASS rename invariance and literal/unit coefficient matching")
