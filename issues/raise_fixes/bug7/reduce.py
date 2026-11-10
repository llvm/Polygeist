#!/usr/bin/env python3
"""Tiny line-block delta reducer: repeatedly try deleting balanced {...}
statements / single lines while cgeist still crashes (rc 139/134)."""
import subprocess, sys, re
B = "/home/arjaiswal/Polygeist/build/bin/cgeist"
fn = sys.argv[2]
def crashes(text):
    open("cand.c", "w").write(text)
    ok = subprocess.run(["clang", "-fsyntax-only", "-Werror=implicit-function-declaration",
                         "-Wno-unused-value", "cand.c"], capture_output=True)
    if ok.returncode != 0:
        return False
    r = subprocess.run([B, "cand.c", f"--function={fn}", "--resource-dir=/usr/lib/clang/14",
                        "-O3", "--raise-scf-to-affine", "-S", "-o", "/dev/null"],
                       capture_output=True, timeout=60)
    return r.returncode in (139, -11, 134, -6)
lines = open(sys.argv[1]).read().split("\n")
assert crashes("\n".join(lines))
def block_end(ls, i):
    depth = 0
    for j in range(i, len(ls)):
        depth += ls[j].count("{") - ls[j].count("}")
        if depth <= 0 and (j > i or "{" not in ls[i]):
            return j
    return None
changed = True
while changed:
    changed = False
    i = 1
    while i < len(lines) - 1:
        j = block_end(lines, i)
        for end in ([j] if j is not None else []) + [i]:
            if end is None: continue
            cand = lines[:i] + lines[end + 1:]
            t = "\n".join(cand)
            if t.count("{") != t.count("}"):
                continue
            if crashes(t):
                lines = cand; changed = True; break
        else:
            i += 1
open(sys.argv[3], "w").write("\n".join(lines))
print("\n".join(lines))
