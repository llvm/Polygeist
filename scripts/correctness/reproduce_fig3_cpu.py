#!/usr/bin/env python3
"""Rebuild and validate the seven repaired Figure 3 CPU cases from C.

See reproduce_fig3_cpu.txt for the protocol and prerequisites. Board access
is supplied by an external, private module exposing upload(path, destination)
and run(command, timeout), returning a status and (for run) captured output.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/correctness"
POLYBENCH = ROOT / "tools/cgeist/Test/polybench"
DEFAULTS = "atax,bicg,cholesky,doitgen,gemver,gesummv,mvt"
NATIVE_FLAGS = ["-O3", "-fno-inline", "-fno-inline-functions",
                "-fsemantic-interposition", "-fno-ipa-ra", "-fno-ipa-cp",
                "-fno-ipa-cp-clone", "-fno-ipa-sra", "-fgnu89-inline",
                "-Dstatic="]


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def command(args, log, env=None, timeout=900):
    with log.open("a") as stream:
        stream.write("COMMAND " + shlex.join(map(str, args)) + "\n")
        stream.flush()
        try:
            return subprocess.run(list(map(str, args)), cwd=ROOT, env=env,
                                  stdout=stream, stderr=stream,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def build(args):
    env = dict(os.environ, PYTHON="/usr/bin/python3", POLYGEIST_KEEP_WORK="1",
               POLYGEIST_CPU_BLAS="1", POLYGEIST_CGEIST_FLAGS="--no-inline",
               POLYGEIST_POLYBENCH_REPETITIONS="0", POLYGEIST_WARMUP_RUNS="5",
               POLYGEIST_TIMED_RUNS="5",
               POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE="1")
    cc = "aarch64-linux-gnu-gcc" if args.target == "jetson-cpu" else "gcc"
    if args.openblas_root:
        base = args.openblas_root.resolve()
        library = base / "usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0"
        env["POLYGEIST_CPU_BLAS_CFLAGS"] = "-I" + str(
            base / "usr/include/aarch64-linux-gnu/openblas-pthread")
        env["POLYGEIST_CPU_BLAS_LIBS"] = str(library) + " -Wl,--allow-shlib-undefined"
    else:
        library = None
    tools = {name: Path(shutil.which(name)) for name in (cc, "objcopy")}
    tools.update({name: ROOT / "llvm-project/build/bin" / name
                  for name in ("clang", "mlir-opt", "mlir-translate")})
    inputs = ["lib/polygeist/Passes/LowerPolygeistSubmap.cpp",
              "lib/polygeist/Passes/LowerKernelLaunchToCuBLAS.cpp",
              "scripts/correctness/kernel_match.py", "scripts/correctness/kernel_match_rewrite.py",
              "generic_solver/kernel_library_phase2.mlir", "runtime/polygeist_cublas_rt_cpu.c",
              "scripts/correctness/polygeist_build.sh", "runtime/polybench_repeat_driver.c",
              "tools/cgeist/Test/polybench/utilities/polybench.c",
              "tools/cgeist/Test/polybench/utilities/polybench.h"]
    manifest = {"tool_hashes": {name: {"path": str(path), "sha256": sha(path)}
                                for name, path in tools.items()},
        "input_hashes": {name: sha(ROOT / name) for name in inputs},
        "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target": args.target, "size": args.size, "dtype": args.dtype,
        "native_compiler": subprocess.check_output([cc, "--version"], text=True),
        "native_flags": NATIVE_FLAGS, "environment": {
            k: v for k, v in env.items() if k.startswith("POLYGEIST_")},
        "compiler_hashes": {n: sha(ROOT / "build/bin" / n)
                            for n in ("cgeist", "polygeist-opt")},
        "openblas": str(library) if library else "host linker -lopenblas",
        "openblas_sha256": sha(library) if library else None,
        "driver_sha256": sha(ROOT / "runtime/polybench_repeat_driver.c"),
        "script_sha256": sha(__file__), "kernels": []}
    (args.output / "source.diff").write_bytes(subprocess.check_output(
        ["git", "diff"], cwd=ROOT))
    (args.output / "host-load.txt").write_bytes(subprocess.check_output(["uptime"]))
    for name in args.kernels.split(","):
        matches = list(POLYBENCH.rglob(name + ".c"))
        if len(matches) != 1:
            raise ValueError(f"Expected one PolyBench source for {name}")
        original = matches[0]
        dest = args.output / name
        dest.mkdir(exist_ok=True)
        source = dest / original.name
        shutil.copyfile(original, source)  # Computational C is unchanged.
        header = original.with_suffix(".h")
        # Only output formatting changes. Preserve all dtype/shape definitions.
        precise_header = re.sub(r'(#\s*define DATA_PRINTF_MODIFIER )"%0\.2l?f "',
                                r'\1"%.17g "', header.read_text())
        (dest / header.name).write_text(precise_header)
        flags = ["-O3", "-I" + str(POLYBENCH / "utilities"),
                 "-I" + str(dest), "-D" + args.size + "_DATASET",
                 "-DDATA_TYPE_IS_" + args.dtype.upper(), "-DPOLYBENCH_TIME",
                 "-DPOLYBENCH_DUMP_ARRAYS"]
        row = {"kernel": name, "source": str(original.relative_to(ROOT)),
               "source_sha256": sha(source), "original_source_sha256": sha(original),
               "header_sha256": sha(dest / header.name),
               "original_header_sha256": sha(header), "flags": flags}
        log = dest / "build.log"
        log.write_text("")
        native_commands = [
            [cc, *NATIVE_FLAGS, *flags,
             "-Dmain=polygeist_polybench_iteration",
             "-include", ROOT / "runtime/polybench_repeat_entry.h",
             "-c", source, "-o", dest / "native.o"],
            [cc, *flags, "-c", POLYBENCH / "utilities/polybench.c",
             "-o", dest / "polybench.o"],
            [cc, "-O2", ROOT / "runtime/polybench_repeat_driver.c",
             dest / "native.o", dest / "polybench.o", "-lm", "-o", dest / "native"]]
        row["native_build_rc"] = 0
        for cmd in native_commands:
            row["native_build_rc"] = command(cmd, log, env)
            if row["native_build_rc"]:
                break
        env["POLYGEIST_BUILD_WORK_DIR"] = str(dest / "work")
        env["POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE"] = (
            "0" if name in args.preserve_views.split(",") else "1")
        row["lower_submap_before_debufferize"] = env["POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE"]
        cmd = [SCRIPTS / "polygeist_build.sh", "--target=" + args.target,
               "--whole-program", "--function=kernel_" + name,
               "-o", dest / "raised", source, *flags]
        if name in args.memory_form.split(","):
            cmd.insert(1, "--no-debuf")
        row["raised_build_rc"] = command(cmd, log, env)
        if row["raised_build_rc"] == 0:
            # Rename only the complete program's entry after compilation.
            # Kernel bodies and source helpers remain compiler-generated.
            objcopy = "aarch64-linux-gnu-objcopy" if args.target == "jetson-cpu" else "objcopy"
            shutil.copyfile(dest / "work/kernel.o", dest / "raised.o")
            relink = [
                [objcopy, "--redefine-sym", "main=polygeist_polybench_iteration", dest / "raised.o"],
                [cc, "-O2", ROOT / "runtime/polybench_repeat_driver.c", dest / "raised.o",
                 *[dest / "work" / (x + ".o") for x in
                   ("wrapper", "harness", "rt", "mlir_runner_utils", "polybench")],
                 *shlex.split(env.get("POLYGEIST_CPU_BLAS_LIBS", "-lopenblas")),
                 "-lm", "-lpthread", "-Wl,--gc-sections", "-o", dest / "raised"]]
            for link in relink:
                row["raised_build_rc"] = command(link, log, env)
                if row["raised_build_rc"]:
                    break
        if row["raised_build_rc"] == 0:
            row["calls"] = re.findall(r"kernel.launch @([^ (]+)",
                                      (dest / "work/matched.mlir").read_text())
        for variant in ("native", "raised"):
            if row[variant + "_build_rc"] == 0:
                row[variant + "_sha256"] = sha(dest / variant)
        manifest["kernels"].append(row)
        write_json(args.output / "manifest.json", manifest)
        print(name, "build", row["native_build_rc"], row["raised_build_rc"], flush=True)
    return manifest


def stats(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return {"samples_ms": samples, "min_ms": min(samples),
            "median_ms": statistics.median(samples), "max_ms": max(samples),
            "iqr_ms": quartiles[2] - quartiles[0]}


def run(args, manifest):
    remote = None
    destination = "/tmp/" + args.output.name
    if args.target == "jetson-cpu":
        if not args.remote_helper:
            raise ValueError("--remote-helper is required for board execution")
        spec = importlib.util.spec_from_file_location("board", args.remote_helper)
        remote = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(remote)
        rc, output = remote.run("mkdir -p " + shlex.quote(destination), timeout=30)
        if rc:
            raise RuntimeError("Could not prepare board directory")
        probe = (SCRIPTS / "fig3_board_preflight.py").read_text()
        rc, output = remote.run("python3 - <<'PY'\n" + probe + "\nPY", timeout=40)
        (args.output / "deployment-preflight.log").write_text(output)
        if rc:
            raise RuntimeError("Board is not ready; no binaries or libraries deployed")
        if remote.upload(Path(manifest["openblas"]).resolve(),
                         destination + "/libopenblas.so.0"):
            raise RuntimeError("Library deployment failed")
    write_json(args.output / "run-metadata.json", {
        "script_sha256": sha(__file__), "cpu": args.cpu,
        "rtol": args.rtol, "atol": args.atol, "started": time.time(),
        "target": args.target, "warmups": 5, "samples": 5,
        "statistic": "minimum of five", "thread_count": 1})
    shutil.copyfile(__file__, args.output / "run-script.py")
    rows = []
    for index, built in enumerate(manifest["kernels"]):
        name = built["kernel"]
        dest = args.output / name
        row = dict(kernel=name, correctness="not run")
        if built["native_build_rc"] or built["raised_build_rc"]:
            row["correctness"] = "build failed"
            rows.append(row)
            continue
        # Alternate order across workloads; all runs remain sequential.
        order = ("native", "raised") if index % 2 == 0 else ("raised", "native")
        for variant in order:
            exe = dest / variant
            if sha(exe) != built[variant + "_sha256"]:
                raise RuntimeError("Executable changed since build")
            if remote:
                # The probe samples /proc over two seconds, rather than using
                # process-lifetime CPU averages to decide whether the board is idle.
                probe = (SCRIPTS / "fig3_board_preflight.py").read_text()
                rc, output = remote.run("python3 - <<'PY'\n" + probe + "\nPY", timeout=40)
                (dest / (variant + "-preflight.log")).write_text(output)
                if rc:
                    raise RuntimeError("Board preflight failed; no benchmark started")
                target = destination + "/" + name + "-" + variant
                if remote.upload(exe, target):
                    raise RuntimeError("Executable deployment failed")
                cmd = ("chmod +x " + shlex.quote(target) + "; "
                       "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLYGEIST_DUMP_LAST=1 "
                       "LD_LIBRARY_PATH=" + shlex.quote(destination) + ":/usr/lib/aarch64-linux-gnu "
                       "timeout " + str(args.timeout) + " taskset -c " + str(args.cpu) +
                       " " + shlex.quote(target))
                write_json(args.output / "state.json", {"state": "running", "kernel": name,
                                                        "variant": variant, "started": time.time()})
                monitor = shlex.quote(target + "-thermal.log")
                measured_cmd = ("tegrastats --interval 100 > " + monitor +
                    " 2>&1 & monitor_pid=$!; " + cmd +
                    "; bench_status=$?; kill $monitor_pid; wait $monitor_pid 2>/dev/null; "
                    "exit $bench_status")
                rc, output = remote.run(measured_cmd, timeout=args.timeout + 30)
                (dest / (variant + ".log")).write_text(output)
                thermal_rc, thermal = remote.run("cat " + monitor, timeout=30)
                (dest / (variant + "-thermal.log")).write_text(thermal)
                if thermal_rc:
                    rc = rc or thermal_rc
                cpu_lines = re.findall(r"CPU \[([^]]+)\]", thermal)
                frequencies = {int(v) for line in cpu_lines
                               for v in re.findall(r"@(\d+)", line)}
                temperatures = [float(v) for v in re.findall(
                    r"(?:cpu|tj)@([0-9.]+)C", thermal)]
                before = json.loads((dest / (variant + "-preflight.log")).read_text())
                expected = {int(c["scaling_min_freq"]) // 1000 for c in before["clocks"]}
                stable = frequencies.issubset(expected) and all(t < 85 for t in temperatures)
                row[variant + "_thermal"] = {
                    "samples": len(cpu_lines), "cpu_mhz": sorted(frequencies),
                    "max_temperature_c": max(temperatures, default=None),
                    "stable": stable, "short_run_preflight_only": not cpu_lines}
                if not stable:
                    rc = rc or 1
            else:
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           POLYGEIST_DUMP_LAST="1")
                cmd = ["taskset", "-c", str(args.cpu), str(exe)]
                rc = command(cmd, dest / (variant + ".log"), env, args.timeout)
                output = (dest / (variant + ".log")).read_text()
            samples = [float(v) * 1000 for v in re.findall(r"^([0-9]+\.[0-9]+)$", output, re.M)]
            row[variant] = {"rc": rc, "command": cmd, "warmups_ms": samples[:5]}
            if rc == 0 and len(samples) == 10:
                row[variant].update(stats(samples[5:]))
            print(name, variant, "rc", rc, "samples", len(samples), flush=True)
        compare = ["/usr/bin/python3", SCRIPTS / "compare_polybench_dumps.py",
                   dest / "native.log", dest / "raised.log",
                   "--rtol", str(args.rtol), "--atol", str(args.atol)]
        rc = command(compare, dest / "compare.log")
        comparison = (dest / "compare.log").read_text()
        count_match = re.search(r"PASS values=(\d+)", comparison)
        row["output_values"] = int(count_match[1]) if count_match else 0
        dumps_complete = row["output_values"] > 0 and all("==END   DUMP_ARRAYS==" in (dest / (v + ".log")).read_text()
                             for v in ("native", "raised"))
        accepted = rc == 0 and dumps_complete and all(
            "min_ms" in row[v] for v in ("native", "raised"))
        row["correctness"] = "PASS" if accepted else "FAIL"
        if accepted and row["raised"]["min_ms"] > 0:
            row["speedup"] = row["native"]["min_ms"] / row["raised"]["min_ms"]
        row["rtol"], row["atol"] = args.rtol, args.atol
        rows.append(row)
        write_json(args.output / "results.json", rows)
        print(name, row["correctness"], row.get("speedup"), flush=True)
    write_json(args.output / "results.json", rows)
    write_json(args.output / "state.json", {"state": "finished"})
    with (args.output / "RESULTS.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kernel", "native_ms", "raised_ms", "speedup", "correctness"])
        for row in rows:
            writer.writerow([row["kernel"], row.get("native", {}).get("min_ms", ""),
                             row.get("raised", {}).get("min_ms", ""),
                             row.get("speedup", ""), row["correctness"]])
    return all(row["correctness"] == "PASS" for row in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("build", "run", "all"), default="all")
    p.add_argument("--target", choices=("jetson-cpu", "host"), default="jetson-cpu")
    p.add_argument("--openblas-root", type=Path)
    p.add_argument("--remote-helper", type=Path)
    p.add_argument("--kernels", default=DEFAULTS)
    p.add_argument("--memory-form", default="cholesky")
    p.add_argument("--preserve-views", default="")
    p.add_argument("--size", choices=("MINI", "SMALL", "MEDIUM", "LARGE"), default="LARGE")
    p.add_argument("--dtype", choices=("double", "float"), default="double")
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()
    if args.stage != "run" and args.target == "jetson-cpu" and not args.openblas_root:
        p.error("--openblas-root is required for an AArch64 build")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build(args) if args.stage != "run" else json.loads(
        (args.output / "manifest.json").read_text())
    if args.stage == "build":
        return int(any(r["native_build_rc"] or r["raised_build_rc"] for r in manifest["kernels"]))
    return 0 if run(args, manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
