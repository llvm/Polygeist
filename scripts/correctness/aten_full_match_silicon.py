#!/usr/bin/env python3
"""Build the unmeasured ATen FULL-raise/FULL-match Jetson batch.

The generated sources use large, compile-time shapes, because both cgeist and
the library matcher recover dimensions from the C array types.  This script
does not mutate the small canonical extraction fixtures.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ATEN = ROOT / "issues/aten_c_kernels"
HARNESS = ATEN / "benchmarks/aten_full_match_raised_harness.c"
LEGACY_HARNESS = ATEN / "benchmarks/aten_raised_jetson_harness.c"
RESIDENT = ATEN / "benchmarks/aten_full_match_resident_baseline.c"
BUILDER = ROOT / "scripts/correctness/polygeist_build.sh"


def case(kind: str, dims: dict[str, int], problem: str,
         extra: tuple[str, ...] = ()) -> dict:
    return {"kind": kind, "dims": dims, "problem": problem, "extra": extra}


CASES = {
    "aten_as_complex_cpu": case("AS_COMPLEX", {"N": 8_388_608}, "N=8388608"),
    "aten_bf16_dot_cpu": case("DOT", {"M": 1, "K": 16_777_216}, "K=16777216 scalarized-f32"),
    "aten_bf16_gemv_trans_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192 scalarized-f32 trans", ("GEMV_TRANS",)),
    "aten_blas_copy_cpu": case("COPY1", {"N": 16_777_216}, "N=16777216"),
    "aten_blas_dot_naive_cpu": case("DOT", {"N": 16_777_216, "K": 16_777_216}, "N=K=16777216"),
    "aten_blas_gemv_generic_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192"),
    "aten_cat_serial_cpu": case("CAT", {"R": 4096, "M": 4096, "K": 2048, "TOP": 2048}, "R=4096 M=4096 K=2048"),
    "aten_complex_scalarized": case("TWO_COPY", {"N": 8_388_608}, "N=8388608"),
    "aten_conv3d": case("CONV3D_BIAS", {"B": 1, "IC": 8, "OC": 16, "D": 48, "H": 48, "W": 48, "K": 3}, "B1 IC8 OC16 D48 H48 W48 K3"),
    "aten_conv_transpose3d_backward_cpu": case("CONV3D_TRANSPOSE_BACKWARD", {"C": 8, "O": 16, "D": 48, "H": 48, "W": 48, "K": 3}, "C8 O16 D48 H48 W48 K3"),
    "aten_copy_cpu": case("COPY1", {"N": 16_777_216}, "N=16777216"),
    "aten_copy_tensor_array_cpu": case("COPY2", {"B": 4096, "N": 4096}, "B=4096 N=4096"),
    "aten_fast_cat_dim0_cpu": case("COPY2", {"B": 4096, "N": 4096}, "B=4096 N=4096"),
    "aten_flatten_nd_linear_cpu": case("BATCH_GEMM", {"B": 16, "M": 256, "K": 256, "N": 256}, "B16 M256 N256 K256"),
    "aten_fp16_dot_cpu": case("DOT", {"M": 1, "K": 16_777_216}, "K=16777216 scalarized-f32"),
    "aten_fp16_gemv_f16arith_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192 scalarized-f32"),
    "aten_fp16_gemv_f32arith_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192 scalarized-f32"),
    "aten_fp16_gemv_notrans_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192 scalarized-f32"),
    "aten_fp16_gemv_trans_cpu": case("GEMV", {"M": 4096, "K": 8192}, "M=4096 K=8192 scalarized-f32 trans", ("GEMV_TRANS",)),
    "aten_gelu_cpu_tanh": case("GELU", {"N": 8_388_608}, "N=8388608"),
    "aten_linear_combination_cpu": case("LINEAR_COMB", {"N": 8_388_608}, "N=8388608 terms=4"),
    "aten_narrow_copy_dense_cpu": case("NARROW", {"R": 4096, "C": 4096, "S": 1024, "L": 2048}, "R=4096 C=4096 S=1024 L=2048"),
    "aten_nested_clone_cpu": case("COPY2", {"B": 4096, "N": 4096}, "B=4096 N=4096"),
    "aten_nested_matmul_broadcast_cpu": case("BATCH_GEMM", {"B": 16, "M": 256, "K": 256, "N": 256}, "B16 M256 N256 K256"),
    "aten_nested_squeeze_cpu": case("COPY2", {"B": 4096, "N": 4096}, "B=4096 N=4096"),
    "aten_outer": case("OUTER", {"M": 4096, "N": 4096}, "M=4096 N=4096 f64"),
    "aten_slow_conv3d_forward_cpu": case("CONV3D", {"C": 8, "O": 16, "D": 48, "H": 48, "W": 48, "K": 3}, "C8 O16 D48 H48 W48 K3"),
    "aten_unbind_copy_cpu": case("COPY2", {"B": 4096, "N": 4096}, "B=4096 N=4096"),
    "aten_zeros_cpu": case("ZERO", {"N": 16_777_216}, "N=16777216"),
}


def legacy_case(bench: str, dims: dict[str, int], problem: str,
                cudnn: bool = False) -> dict:
    return {"legacy": True, "bench": bench, "dims": dims,
            "problem": problem, "cudnn": cudnn, "extra": ()}


# The first eleven large comparisons predate the exhaustive FULL/FULL harness.
# Keep them in the same driver so direct-buffer and device-resident validation
# covers every executed row in the ATen CE dataset.
CASES.update({
    "aten_add": legacy_case("ADD", {"B": 32, "C": 64, "H": 64, "W": 64},
                            "B32 C64 H64 W64", True),
    "aten_addmm": legacy_case("ADDMM", {"M": 512, "N": 512, "K": 512},
                              "M512 N512 K512"),
    "aten_batch_norm": legacy_case(
        "BATCH_NORM", {"B": 32, "C": 64, "H": 64, "W": 64},
        "B32 C64 H64 W64", True),
    "aten_conv2d": legacy_case(
        "CONV2D", {"B": 8, "IC": 32, "OC": 64, "H": 64, "W": 64,
                   "KH": 3, "KW": 3},
        "B8 IC32 OC64 H64 W64 KH3 KW3", True),
    "aten_dot": legacy_case("DOT", {"N": 8_388_608}, "N8388608"),
    "aten_gelu": legacy_case("GELU", {"N": 8_388_608}, "N8388608", True),
    "aten_max_pool2d": legacy_case(
        "MAX_POOL2D", {"B": 32, "C": 64, "H": 64, "W": 64,
                       "K": 2, "S": 2},
        "B32 C64 H64 W64 K2 S2", True),
    "aten_mm": legacy_case("MM", {"M": 512, "N": 512, "K": 512},
                           "M512 N512 K512"),
    "aten_mv": legacy_case("MV", {"M": 4096, "K": 4096}, "M4096 K4096"),
    "aten_rms_norm": legacy_case("RMS_NORM", {"N": 8_388_608},
                                  "N8388608", True),
    "aten_softmax": legacy_case("SOFTMAX", {"N": 8_388_608},
                                 "N8388608", True),
})


def scaled_source(kernel: str, spec: dict, out: Path) -> None:
    text = (ATEN / f"{kernel}.c").read_text()
    for name, value in spec["dims"].items():
        pattern = rf"(^\s*#\s*define\s+{re.escape(name)}\s+)[^\n]+"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)
        if count == 0:
            text = f"#define {name} {value}\n" + text
    # Preserve the same contiguous flattening semantics while presenting the
    # output at its natural rank.  The flat spelling otherwise introduces a
    # rank-changing submap after the copy launch and currently trips the
    # generic affine write-back fallback during executable lowering.
    if kernel == "aten_fast_cat_dim0_cpu":
        text = text.replace("float out[B*N]", "float out[B][N]")
        text = text.replace("out[b*N+i]", "out[b][i]")
    out.write_text(text)


def run(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log}")


def build_one(kernel: str, spec: dict, out: Path) -> dict:
    work = out / kernel
    work.mkdir(parents=True, exist_ok=True)
    source = work / f"{kernel}_large.c"
    scaled_source(kernel, spec, source)
    reference = f"{kernel}_reference"
    ref_obj = work / "reference.o"
    defs = [f"-D{k}={v}" for k, v in spec["dims"].items()]
    defs += [f"-D{x}" for x in spec.get("extra", ())]
    run(["aarch64-linux-gnu-gcc", "-O3", f"-D{kernel}={reference}",
         "-c", str(source), "-o", str(ref_obj)], work / "reference.build.log")
    exe = work / kernel
    env = os.environ.copy()
    env["POLYGEIST_CUSTOM_CUDA_OBJ"] = str(ref_obj)
    # The Slack-bot virtualenv intentionally has no compiler dependencies;
    # the system Python carries the locally installed egglog package.
    env["PYTHON"] = "/usr/bin/python3"
    env["POLYGEIST_MINIMAL_CUDA_RUNTIME"] = "1"
    if spec.get("legacy") and spec.get("cudnn"):
        env["POLYGEIST_MINIMAL_CUDNN_RUNTIME"] = "1"
    if spec.get("kind") in {
            "CONV3D_BIAS", "CONV3D", "CONV3D_TRANSPOSE_BACKWARD", "GELU"}:
        env["POLYGEIST_MINIMAL_CUDNN_RUNTIME"] = "1"
    cutensornet = Path("/tmp/polygeist_cutensornet_flat")
    if (cutensornet.exists() and
            os.environ.get("POLYGEIST_ATEN_DISABLE_CUTENSORNET", "0") == "0"):
        # This mode also drops unused cuFFT/cuSPARSE DT_NEEDED entries.  The
        # attached Jetson intentionally carries only the libraries exercised
        # by this ATen batch.
        env["POLYGEIST_CUTENSORNET_ROOT"] = str(cutensornet)
        env["POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME"] = "1"
    if spec.get("legacy"):
        harness = LEGACY_HARNESS
        bench_defs = [f"-DBENCH_ATEN_{spec['bench']}"]
    else:
        harness = HARNESS
        bench_defs = [f"-DFUNCTION={kernel}", f"-DREFERENCE={reference}",
                      f"-DBENCH_KIND_{spec['kind']}"]
    cmd = [str(BUILDER), "--target=jetson", f"--function={kernel}",
           f"--harness={harness}", "-o", str(exe), str(source),
           *bench_defs, "-DBENCH_ITERS=5", *defs]
    run(cmd, work / "raised.build.log", env)
    raised_device = str(work / f"{kernel}_raised_device")
    run([
        str(BUILDER), "--target=jetson", f"--function={kernel}",
        f"--harness={harness}", "-o", raised_device,
        str(source), *bench_defs,
        "-DDEVICE_RESIDENT", "-DBENCH_ITERS=20", *defs,
        f"-I{Path('/usr/local/cuda-12.6/targets/sbsa-linux/include')}",
    ], work / "raised_device.build.log", env)
    resident = ""
    if not spec.get("legacy") and spec["kind"] != "GELU":
        resident = str(work / f"{kernel}_resident")
        resident_defs = [f"-DATEN_{k}={v}" for k, v in spec["dims"].items()]
        resident_defs += [f"-D{x}" for x in spec["extra"]]
        cuda = Path("/usr/local/cuda-12.6/targets/sbsa-linux")
        run([
            "aarch64-linux-gnu-gcc", "-O3", str(RESIDENT), str(ref_obj),
            f"-DFUNCTION={kernel}", f"-DREFERENCE={reference}",
            f"-DBENCH_KIND_{spec['kind']}", "-DBENCH_ITERS=20",
            *resident_defs, f"-I{cuda / 'include'}", "-I/usr/include/aarch64-linux-gnu",
            f"-L{cuda / 'lib'}", f"-L{cuda / 'lib/stubs'}",
            "-L/usr/lib/aarch64-linux-gnu", "-lcudnn", "-lcublasLt",
            "-lcublas", "-lcudart", "-lm", "-ldl", "-o", resident,
        ], work / "resident.build.log")
    return {"kernel": kernel, "problem": spec["problem"],
            "kind": spec.get("kind", spec.get("bench", "")),
            "executable": str(exe),
            "raised_device_executable": raised_device,
            "resident_executable": resident,
            "status": "BUILT"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/aten_full_match_large"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--kernel", action="append", choices=sorted(CASES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = args.kernel or sorted(CASES)
    rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_one, k, CASES[k], args.output): k for k in selected}
        for future in concurrent.futures.as_completed(futures):
            kernel = futures[future]
            try:
                row = future.result(); rows.append(row)
                print(f"[BUILT] {kernel}", flush=True)
            except Exception as exc:
                failures.append({"kernel": kernel, "error": str(exc)})
                print(f"[FAIL] {kernel}: {exc}", file=sys.stderr, flush=True)
    manifest = {"cases": sorted(rows, key=lambda x: x["kernel"]), "failures": failures}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built={len(rows)} failed={len(failures)} output={args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
