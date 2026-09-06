#!/usr/bin/env python3
"""Build a static HTML index of PolyBench kernels where each row deep-links to
Compiler Explorer with the full Polygeist pipeline pre-wired:

  - left column:  C source editor + cgeist_aff compiler pane (shows affine MLIR)
  - right column: MLIR editor (pre-filled with affine MLIR) + popt_full compiler
                  pane + Opt Pipeline view (every internal pass clickable)

Per-kernel HTML pages with raised / debuferized / kernel.launch IR are also
rendered (uses the existing matcher pipeline).

Inputs:
  - PolyBench C sources at $POLYBENCH/tools/cgeist/Test/polybench/.../<k>.c
  - Pre-computed affine MLIR at /tmp/polybench_new/<k>.mlir
  - Pre-computed linalg MLIR at /tmp/polybench_new/<k>_linalg.mlir
  - Pre-computed debuf MLIR  at /tmp/polybench_new/<k>_debuf.mlir

Output:
  /tmp/ir_viewer/index.html   (entrypoint — open this)
  /tmp/ir_viewer/<k>.html     (per-kernel IR preview)
"""
import csv
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def env_path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, str(default)))


POLYBENCH_TEST_DIR = env_path(
    "POLYGEIST_POLYBENCH_TEST_DIR",
    REPO_ROOT / "tools/cgeist/Test/polybench",
)
POLYBENCH_UTILS = POLYBENCH_TEST_DIR / "utilities"
MLIR_DIR = env_path("POLYGEIST_POLYBENCH_MLIR_DIR", "/tmp/polybench_new")
MACHSUITE_ROOT = env_path("POLYGEIST_MACHSUITE_ROOT", REPO_ROOT / "third_party/MachSuite")
MACHSUITE_MLIR_DIR = env_path("POLYGEIST_MACHSUITE_MLIR_DIR", "/tmp/machsuite_mlir")
NPB_ROOT = env_path("POLYGEIST_NPB_ROOT", REPO_ROOT / "third_party/NPB-polybenchified")
NPB_MLIR_DIR = env_path("POLYGEIST_NPB_MLIR_DIR", "/tmp/npb_mlir")
LLAMA2C_ROOT = env_path("POLYGEIST_LLAMA2C_ROOT", REPO_ROOT / "third_party/llama2.c")
LLAMA2C_MLIR_DIR = env_path("POLYGEIST_LLAMA2C_MLIR_DIR", "/tmp/llama2c_mlir")
LLAMA_FORWARD_ROOT = env_path(
    "POLYGEIST_LLAMA_FORWARD_ROOT",
    REPO_ROOT / "third_party/cnn-extracted",
)
LLAMA_FORWARD_MLIR_DIR = env_path(
    "POLYGEIST_LLAMA_FORWARD_MLIR_DIR",
    "/tmp/llama_forward_ops_mlir",
)
WHISPER_OPS_ROOT = env_path(
    "POLYGEIST_WHISPER_OPS_ROOT",
    REPO_ROOT / "third_party/cnn-extracted",
)
WHISPER_OPS_MLIR_DIR = env_path(
    "POLYGEIST_WHISPER_OPS_MLIR_DIR",
    "/tmp/whisper_ops_mlir",
)
ATEN_C_ROOT = env_path(
    "POLYGEIST_ATEN_C_ROOT",
    REPO_ROOT / "issues/aten_c_kernels",
)
ATEN_C_MLIR_DIR = env_path(
    "POLYGEIST_ATEN_C_MLIR_DIR",
    ATEN_C_ROOT / "results",
)
ATEN_SILICON_RESULTS = env_path(
    "POLYGEIST_ATEN_SILICON_RESULTS",
    ATEN_C_ROOT / "silicon_results/large_problem_comparison.csv",
)
ATEN_DEVICE_RESIDENCY_RESULTS = env_path(
    "POLYGEIST_ATEN_DEVICE_RESIDENCY_RESULTS",
    ATEN_C_ROOT / "silicon_results/device_residency_comparison.csv",
)
ATEN_CUDA_LIBRARY_AUDIT = env_path(
    "POLYGEIST_ATEN_CUDA_LIBRARY_AUDIT",
    ATEN_C_ROOT / "cuda_library_audit.csv",
)
ATEN_UPSTREAM_ROOT = env_path(
    "POLYGEIST_ATEN_UPSTREAM_ROOT",
    REPO_ROOT / "third_party/pytorch",
)
ATEN_UPSTREAM_COMMIT = "d7af122d81a49b1fa7a31ba52bd57c026f092646"
MFEM_C_ROOT = env_path(
    "POLYGEIST_MFEM_C_ROOT",
    REPO_ROOT / "issues/mfem_c_kernels",
)
MFEM_RESULTS_DIR = env_path(
    "POLYGEIST_MFEM_RESULTS_DIR",
    MFEM_C_ROOT / "results",
)
MFEM_MATCH_RESULTS_DIR = env_path(
    "POLYGEIST_MFEM_MATCH_RESULTS_DIR",
    MFEM_C_ROOT / "match_results",
)
MFEM_SILICON_RESULTS_DIR = env_path(
    "POLYGEIST_MFEM_SILICON_RESULTS_DIR",
    MFEM_C_ROOT / "silicon_results",
)
MFEM_APPLICATIONS_DIR = env_path(
    "POLYGEIST_MFEM_APPLICATIONS_DIR",
    MFEM_C_ROOT / "applications",
)
MFEM_APPLICATION_EXTRACTIONS_DIR = env_path(
    "POLYGEIST_MFEM_APPLICATION_EXTRACTIONS_DIR",
    MFEM_C_ROOT / "application_extractions",
)
MFEM_APPLICATION_EXTRACTION_RESULTS_DIR = env_path(
    "POLYGEIST_MFEM_APPLICATION_EXTRACTION_RESULTS_DIR",
    MFEM_APPLICATION_EXTRACTIONS_DIR / "results",
)
MFEM_UPSTREAM_ROOT = env_path(
    "POLYGEIST_MFEM_UPSTREAM_ROOT",
    REPO_ROOT / "third_party/mfem",
)
MFEM_UPSTREAM_COMMIT = "951cf8886b9c0c33fb36a2f0ede268c8d6a0d8b5"

# Correctness-gated application runs on the attached Jetson.  These are kept
# separate from the structural matcher counts: a candidate launch is not a
# performance result until the complete application agrees with its -O3 CPU
# reference.  `process_wall_s` includes CUDA/cuTensorNet initialization and
# is diagnostic only.  The harness reports timing only after correctness;
# all ten harness-supported executable paths now pass this gate.
MFEM_APPLICATION_JETSON_RUNS: dict[str, dict[str, str]] = {
    "mfem_app_mtop_iso_elasticity_dfem_2d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "NE=1024 max_abs=max_rel=5.551115e-17",
        "runtime": "NE=1024 apples-to-apples: CPU -O3 935.686403 us; warm raised Jetson 17465.791991 us; speedup 0.053573x (raised is 18.666x slower)",
        "calls": "12 cuTensorNet launches/application",
        "params": "f64; NE=1024; D1D=4; Q1D=5; B/G=20; x/y=32768; lambda/mu=25600; J=102400; weights=25; median of warm process runs 2-4",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_ex35p_hcurl_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=4.857226e-17",
        "runtime": "CPU -O3 62.732794 us; warm raised Jetson 58184.332796 us; speedup 0.001078x",
        "calls": "29 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; Bo/Bot=15; Bc/Bct/G/Gt=20; operators=1500; x/y=288",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_dfem_minimal_surface_2d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=8.673617e-19",
        "runtime": "CPU -O3 0.876817 us; warm raised Jetson 11099.043209 us; speedup 0.000079x",
        "calls": "6 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; B/G=20; field/y=32; Jacobian=100; weights=25",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_ex35p_h1_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=6.938894e-18",
        "runtime": "CPU -O3 5.427189 us; warm raised Jetson 36668.323190 us; speedup 0.000148x",
        "calls": "19 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; B/G/Bt/Gt=20; diffusion=1500; mass=250; x/y=128",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_ex35p_hdiv_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=4.857226e-17",
        "runtime": "CPU -O3 42.969594 us; warm raised Jetson 22730.832011 us; speedup 0.001890x",
        "calls": "12 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; Bo/Bot=15; Bc/Bct/G/Gt=20; div=250; mass=1500; x/y=216",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_ex9p_mass_convection_2d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=6.938894e-18",
        "runtime": "CPU -O3 0.671996 us; warm raised Jetson 14588.060789 us; speedup 0.000046x",
        "calls": "9 library launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; B/G/Bt=20; mass=50; convection=100; vector extents=32",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_grad_div_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=4.857226e-17",
        "runtime": "CPU -O3 42.947195 us; warm raised Jetson 22724.332800 us; speedup 0.001890x",
        "calls": "12 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; Bo/Bot=15; Bc/Bct/G/Gt=20; div=250; mass=1500; x/y=216",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_abs_l1_mass_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=6.938894e-18",
        "runtime": "CPU -O3 1.407997 us; warm raised Jetson 9253.155184 us; speedup 0.000152x",
        "calls": "5 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; B/Bt=20; D=250; x/y=128; one untimed warm-up; 10 timed warm iterations; final Y += contraction remains residual Linalg",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_abs_l1_diffusion_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=2.710505e-20",
        "runtime": "CPU -O3 4.047994 us; warm raised Jetson 26162.835187 us; speedup 0.000155x",
        "calls": "14 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; B/G/Bt/Gt=20; operator=1500; x/y=128",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_abs_l1_curlcurl_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "max_abs=max_rel=4.857226e-17",
        "runtime": "CPU -O3 60.716807 us; warm raised Jetson 57165.744016 us; speedup 0.001062x",
        "calls": "29 cuTensorNet launches/application",
        "params": "f64; NE=2; D1D=4; Q1D=5; Bo/Bot=15; Bc/Bct/G/Gt=20; operators=1500; x/y=288",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 CPU reference",
    },
    "mfem_app_navier_tgv_pa_operators_3d": {
        "outcome": "CORRECTNESS PASS",
        "correctness": "NE=2 max_abs=max_rel=4.163336e-17; NE=1024 max_abs=max_rel=8.326673e-17; residual loops 26 -> 0",
        "runtime": "NE=2: CPU 0.947853 ms, warm Jetson 142.371238 ms, 0.006658x. NE=1024: CPU 491.086496 ms, warm Jetson 1402.250182 ms, 0.350213x",
        "calls": "70 cuTensorNet launches/application",
        "params": "f64; D1D=4; Q1D=5; NE compile-time scalable. NE=1024: velocity=196608, pressure=65536, largest operator=2304000 doubles; vector mass + vector diffusion + nonlinear convection + pressure diffusion + divergence + gradient; one untimed raised warm-up; 5 timed warm iterations",
        "hardware": "Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet; independent aarch64 -O3 direct C reference",
    },
}
STENCIL_CONV2D_ROOT = env_path(
    "POLYGEIST_STENCIL_CONV2D_ROOT",
    REPO_ROOT / "third_party/cnn-extracted",
)
STENCIL_CONV2D_MLIR_DIR = env_path(
    "POLYGEIST_STENCIL_CONV2D_MLIR_DIR",
    "/tmp/stencil_conv2d_mlir",
)
LLMC_ROOT = env_path("POLYGEIST_LLMC_ROOT", REPO_ROOT / "third_party/llm.c")
LLMC_MLIR_DIR = env_path("POLYGEIST_LLMC_MLIR_DIR", "/tmp/llmc_mlir")
DARKNET_ROOT = env_path("POLYGEIST_DARKNET_ROOT", REPO_ROOT / "third_party/darknet")
DARKNET_MLIR_DIR = env_path("POLYGEIST_DARKNET_MLIR_DIR", "/tmp/darknet_mlir")
EXTRACTED_DARKNET_ROOT = env_path(
    "POLYGEIST_EXTRACTED_DARKNET_ROOT",
    REPO_ROOT / "third_party/cnn-extracted",
)
EXTRACTED_DARKNET_MLIR_DIR = env_path(
    "POLYGEIST_EXTRACTED_DARKNET_MLIR_DIR",
    "/tmp/extracted_darknet_mlir",
)
OUTPUT_DIR = env_path("POLYGEIST_IR_VIEWER_OUT", "/tmp/ir_viewer")
SECTION42_RESULTS_DIR = env_path(
    "POLYGEIST_SECTION42_RESULTS_DIR",
    REPO_ROOT / "issues/polybench_section42",
)
GINSBACH_SUMMARY = env_path(
    "POLYGEIST_GINSBACH_SUMMARY",
    REPO_ROOT / "issues/ginsbach_asplos18/program_summary_2026-09-05.csv",
)
GINSBACH_SILICON = env_path(
    "POLYGEIST_GINSBACH_SILICON",
    REPO_ROOT / "issues/ginsbach_asplos18/silicon_results_2026-09-05.csv",
)
GINSBACH_AUDIT_ROOT = env_path(
    "POLYGEIST_GINSBACH_AUDIT_ROOT",
    "/tmp/ginsbach_external_hist_fact",
)
REWRITER = env_path("POLYGEIST_KERNEL_MATCH_REWRITER", SCRIPT_DIR / "kernel_match_rewrite.py")
PYTHON = os.environ.get("PYTHON", sys.executable)
POLYGEIST_OPT = env_path("POLYGEIST_OPT", REPO_ROOT / "build/bin/polygeist-opt")
KERNEL_LIBRARY = env_path(
    "POLYGEIST_KERNEL_LIBRARY",
    REPO_ROOT / "generic_solver/kernel_library_phase2.mlir",
)
INJECT_KERNEL_LIBRARY = SCRIPT_DIR / "inject_kernel_library.py"
CPU_RUNTIME = REPO_ROOT / "runtime/polygeist_cublas_rt_cpu.c"
CUDA_RUNTIME = REPO_ROOT / "runtime/polygeist_cublas_rt_cuda.c"

# Runtime calls that the host shim can route to an optimized CBLAS provider
# when POLYGEIST_CPU_BLAS=1. Other implemented host calls use reference C.
CPU_CBLAS_CALLS = {
    "polygeist_cublas_dgemm",
    "polygeist_cublas_sgemm",
    "polygeist_cublas_dgemv",
    "polygeist_cublas_sgemv",
    "polygeist_cublas_dgemv_T",
    "polygeist_cublas_sgemv_T",
    "polygeist_cublas_daxpby",
    "polygeist_cublas_daxpy_unit",
    "polygeist_cublas_dger_rank2",
    "polygeist_cublas_dscal_2d",
    "polygeist_cublas_sgemm_1x1conv",
    "polygeist_cublas_dsyrk",
    "polygeist_cublaslt_matmul_bias_relu",
}

# MachSuite tag → (relative subdir under third_party/MachSuite, kernel function).
# The tag is what the viewer uses for filenames and as the display name.
MACHSUITE_KERNELS: dict[str, tuple[str, str]] = {
    "aes":           ("aes/aes",              "aes256_encrypt_ecb"),
    "backprop":      ("backprop/backprop",    "backprop"),
    "bfs-bulk":      ("bfs/bulk",             "bfs"),
    "bfs-queue":     ("bfs/queue",            "bfs"),
    "fft-strided":   ("fft/strided",          "fft"),
    "fft-transpose": ("fft/transpose",        "fft1D_512"),
    "gemm-ncubed":   ("gemm/ncubed",          "gemm"),
    "gemm-blocked":  ("gemm/blocked",         "bbgemm"),
    "kmp":           ("kmp/kmp",              "kmp"),
    "md-grid":       ("md/grid",              "md"),
    "md-knn":        ("md/knn",               "md_kernel"),
    "nw":            ("nw/nw",                "needwun"),
    "sort-merge":    ("sort/merge",           "ms_mergesort"),
    "sort-radix":    ("sort/radix",           "ss_sort"),
    "spmv-crs":      ("spmv/crs",             "spmv"),
    "spmv-ellpack":  ("spmv/ellpack",         "ellpack"),
    "stencil2d":     ("stencil/stencil2d",    "stencil"),
    "stencil3d":     ("stencil/stencil3d",    "stencil3d"),
    "viterbi":       ("viterbi/viterbi",      "viterbi"),
}

# PolyBench-extracted NPB kernels (one .c per kernel in NPB-polybenchified/).
# These were manually carved out of the monolithic per-benchmark .c files
# in NPB3.0-omp-C; the kernel functions had their static-global dependencies
# converted to explicit array parameters so the pipeline can isolate them
# without the extraction issues the whole-file sweep hit.
NPB_KERNELS: dict[str, tuple[str, str]] = {
    "bt-add":      ("bt_add.c",      "bt_add"),
    "ft-evolve":   ("ft_evolve.c",   "ft_evolve"),
    "lu-l2norm":   ("lu_l2norm.c",   "lu_l2norm"),
    "mg-psinv":    ("mg_psinv.c",    "mg_psinv"),
    "mg-resid":    ("mg_resid.c",    "mg_resid"),
    "mg-norm2u3":  ("mg_norm2u3.c",  "mg_norm2u3"),
    "mg-rprj3":    ("mg_rprj3.c",    "mg_rprj3"),
}

# llama2.c hot numeric functions in run.c. All three live in the same file.
LLAMA2C_KERNELS: dict[str, tuple[str, str]] = {
    "rmsnorm":  ("run.c", "rmsnorm"),
    "softmax":  ("run.c", "softmax"),
    "matmul":   ("run.c", "matmul"),
}

# Standalone Llama-forward operation fixtures plus the fuller one-token
# one-layer forward fixture. These live in third_party/cnn-extracted/ and are
# intentionally source-level C benchmarks that our pipeline raises.
LLAMA_FORWARD_KERNELS: dict[str, tuple[str, str]] = {
    "token_embedding":        ("llama_forward_ops.c", "kernel_llama_token_embedding"),
    "attention_rmsnorm":      ("llama_forward_ops.c", "kernel_llama_attention_rmsnorm"),
    "qkv_projection":         ("llama_forward_ops.c", "kernel_llama_qkv_projection"),
    "rope_interleaved":       ("llama_forward_ops.c", "kernel_llama_rope"),
    "rope_split":             ("llama_forward_ops.c", "kernel_llama_rope_split"),
    "kv_cache_rw":            ("llama_forward_ops.c", "kernel_llama_kv_cache_rw"),
    "attention_scores":       ("llama_forward_ops.c", "kernel_llama_attention_scores"),
    "attention_mask_if":      ("llama_forward_ops.c", "kernel_llama_attention_mask"),
    "attention_mask_select":  ("llama_forward_ops.c", "kernel_llama_attention_mask_select"),
    "attention_softmax":      ("llama_forward_ops.c", "kernel_llama_attention_softmax"),
    "attention_output":       ("llama_forward_ops.c", "kernel_llama_attention_output"),
    "output_projection":      ("llama_forward_ops.c", "kernel_llama_output_projection"),
    "residual_add":           ("llama_forward_ops.c", "kernel_llama_residual_add"),
    "ffn_rmsnorm":            ("llama_forward_ops.c", "kernel_llama_ffn_rmsnorm"),
    "gate_up_projection":     ("llama_forward_ops.c", "kernel_llama_gate_up_projection"),
    "swiglu":                 ("llama_forward_ops.c", "kernel_llama_swiglu"),
    "down_projection":        ("llama_forward_ops.c", "kernel_llama_down_projection"),
    "final_rmsnorm":          ("llama_forward_ops.c", "kernel_llama_final_rmsnorm"),
    "lm_head_projection":     ("llama_forward_ops.c", "kernel_llama_lm_head_projection"),
    "extended_forward":       ("llama2_extended_forward_bench.c", "kernel_llama2_extended_forward"),
}

LLAMA_FORWARD_ORDER = list(LLAMA_FORWARD_KERNELS.keys())

LLAMA_FORWARD_DISPLAY_NAMES: dict[str, str] = {
    "token_embedding":        "token embedding",
    "attention_rmsnorm":      "attention RMSNorm",
    "qkv_projection":         "QKV projection",
    "rope_interleaved":       "RoPE, interleaved",
    "rope_split":             "RoPE, split",
    "kv_cache_rw":            "KV cache read/write",
    "attention_scores":       "attention scores",
    "attention_mask_if":      "causal mask, if-form",
    "attention_mask_select":  "causal mask, select-form",
    "attention_softmax":      "attention softmax",
    "attention_output":       "attention output",
    "output_projection":      "output projection",
    "residual_add":           "residual add",
    "ffn_rmsnorm":            "FFN RMSNorm",
    "gate_up_projection":     "gate/up projection",
    "swiglu":                 "SwiGLU",
    "down_projection":        "down projection",
    "final_rmsnorm":          "final RMSNorm",
    "lm_head_projection":     "LM head projection",
    "extended_forward":       "extended forward benchmark",
}

WHISPER_OPS_KERNELS: dict[str, tuple[str, str]] = {
    "whisper_vec_dot":      ("whisper_ops.c", "kernel_whisper_vec_dot"),
    "whisper_vec_softmax":  ("whisper_ops.c", "kernel_whisper_vec_softmax"),
    "whisper_softmax_full": ("whisper_ops.c", "kernel_whisper_softmax_full"),
    "whisper_rms_norm":     ("whisper_ops.c", "kernel_whisper_rms_norm"),
    "whisper_gelu":         ("whisper_ops.c", "kernel_whisper_gelu"),
    "whisper_conv1d":       ("whisper_ops.c", "kernel_whisper_conv1d"),
    "whisper_quantize_q4_0_ref": (
        "../whisper.cpp/ggml/src/ggml-quants.c",
        "quantize_row_q4_0_ref",
    ),
    "whisper_decode_residue": (
        "../whisper.cpp/examples/stb_vorbis.c",
        "decode_residue",
    ),
    "whisper_inverse_mdct": (
        "../whisper.cpp/examples/stb_vorbis.c",
        "inverse_mdct",
    ),
}

WHISPER_OPS_ORDER = list(WHISPER_OPS_KERNELS.keys())

WHISPER_OPS_DISPLAY_NAMES: dict[str, str] = {
    "whisper_vec_dot":      "vector dot",
    "whisper_vec_softmax":  "vector softmax",
    "whisper_softmax_full": "full softmax",
    "whisper_rms_norm":     "RMSNorm",
    "whisper_gelu":         "GELU",
    "whisper_conv1d":       "1D convolution",
    "whisper_quantize_q4_0_ref": "q4_0 quantize ref",
    "whisper_decode_residue":    "Vorbis residue decode",
    "whisper_inverse_mdct":      "Vorbis inverse MDCT",
}

ATEN_C_KERNELS: dict[str, tuple[str, str]] = {
    p.stem: (p.name, p.stem) for p in sorted(ATEN_C_ROOT.glob("aten_*.c"))
}

ATEN_C_ORDER = list(ATEN_C_KERNELS.keys())

# Pinned provenance for the standalone numerical C fixtures.  The second
# tuple member is a source token used only to add a useful line anchor; when a
# stable token is unavailable, the link intentionally targets the whole file.
# These are implementation-family links, not a claim that the C fixtures are
# textual copies: ATen dispatch/TensorIterator/template machinery was removed.
ATEN_C_PROVENANCE: dict[str, tuple[str, str | None]] = {
    "aten_adaptive_avg_pool2d": ("aten/src/ATen/native/AdaptiveAveragePooling.cpp", "adaptive_avg_pool2d"),
    "aten_adaptive_avg_pool3d": ("aten/src/ATen/native/AdaptiveAveragePooling3d.cpp", "adaptive_avg_pool3d"),
    "aten_add": ("aten/src/ATen/native/CPUBlas.cpp", "void axpy"),
    "aten_addmm": ("aten/src/ATen/native/LinearAlgebra.cpp", "static void addmm_impl_cpu_"),
    "aten_avg_pool2d": ("aten/src/ATen/native/AveragePool2d.cpp", "avg_pool2d"),
    "aten_avg_pool3d": ("aten/src/ATen/native/AveragePool3d.cpp", "avg_pool3d"),
    "aten_batch_norm": ("aten/src/ATen/native/cpu/batch_norm_kernel.cpp", "batch_norm_cpu_kernel"),
    "aten_binary_cross_entropy": ("aten/src/ATen/native/Loss.cpp", "binary_cross_entropy"),
    "aten_bmm": ("aten/src/ATen/native/LinearAlgebra.cpp", "bmm"),
    "aten_channel_shuffle": ("aten/src/ATen/native/ChanelShuffle.cpp", "channel_shuffle"),
    "aten_clamp": ("aten/src/ATen/native/TensorCompare.cpp", "clamp"),
    "aten_conv1d": ("aten/src/ATen/native/Convolution.cpp", "conv1d"),
    "aten_conv2d": ("aten/src/ATen/native/ConvolutionMM2d.cpp", "slow_conv2d"),
    "aten_conv3d": ("aten/src/ATen/native/Convolution.cpp", "conv3d"),
    "aten_conv_transpose2d": ("aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp", "slow_conv_transpose2d"),
    "aten_cross": ("aten/src/ATen/native/Cross.cpp", "cross"),
    "aten_cumsum": ("aten/src/ATen/native/cpu/ReduceOpsKernel.cpp", "cumsum_cpu_kernel"),
    "aten_dot": ("aten/src/ATen/native/Blas.cpp", "Tensor dot"),
    "aten_elu": ("aten/src/ATen/native/cpu/Activation.cpp", "elu_kernel"),
    "aten_embedding": ("aten/src/ATen/native/Embedding.cpp", "embedding"),
    "aten_gelu": ("aten/src/ATen/native/Activation.cpp", "gelu"),
    "aten_hardsigmoid": ("aten/src/ATen/native/cpu/Activation.cpp", "hardsigmoid_kernel"),
    "aten_hardswish": ("aten/src/ATen/native/cpu/Activation.cpp", "hardswish_kernel"),
    "aten_hardtanh": ("aten/src/ATen/native/Activation.cpp", "Tensor hardtanh"),
    "aten_im2col": ("aten/src/ATen/native/Im2Col.cpp", "im2col"),
    "aten_l1_loss": ("aten/src/ATen/native/Loss.cpp", "l1_loss"),
    "aten_layer_norm": ("aten/src/ATen/native/layer_norm.cpp", "layer_norm"),
    "aten_leaky_relu": ("aten/src/ATen/native/cpu/Activation.cpp", "leaky_relu_kernel"),
    "aten_lerp": ("aten/src/ATen/native/cpu/LerpKernel.cpp", "lerp_kernel"),
    "aten_max_pool2d": ("aten/src/ATen/native/cpu/MaxPoolKernel.cpp", "max_pool2d"),
    "aten_mean": ("aten/src/ATen/native/ReduceOps.cpp", "mean"),
    "aten_mm": ("aten/src/ATen/native/LinearAlgebra.cpp", "TORCH_IMPL_FUNC(mm_out_cpu)"),
    "aten_mse_loss": ("aten/src/ATen/native/Loss.cpp", "mse_loss"),
    "aten_mv": ("aten/src/ATen/native/Blas.cpp", "Tensor &mv_out"),
    "aten_outer": ("aten/src/ATen/native/LinearAlgebra.cpp", "outer"),
    "aten_pixel_shuffle": ("aten/src/ATen/native/PixelShuffle.cpp", "pixel_shuffle"),
    "aten_prod": ("aten/src/ATen/native/cpu/ReduceOpsKernel.cpp", "prod_kernel"),
    "aten_reflection_pad2d": ("aten/src/ATen/native/ReflectionPad.cpp", "reflection_pad2d"),
    "aten_relu": ("aten/src/ATen/native/Activation.cpp", "relu"),
    "aten_replication_pad2d": ("aten/src/ATen/native/ReplicationPadding.cpp", "replication_pad2d"),
    "aten_rms_norm": ("aten/src/ATen/native/layer_norm.cpp", "rms_norm_composite"),
    "aten_sigmoid": ("aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "sigmoid_kernel"),
    "aten_silu": ("aten/src/ATen/native/Activation.cpp", "silu"),
    "aten_softmax": ("aten/src/ATen/native/cpu/SoftMaxKernel.cpp", "softmax"),
    "aten_softplus": ("aten/src/ATen/native/cpu/Activation.cpp", "softplus_kernel"),
    "aten_sum": ("aten/src/ATen/native/ReduceOps.cpp", "sum"),
    "aten_tanh": ("aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "tanh_kernel"),
    "aten_transpose_copy": ("aten/src/ATen/native/TensorShape.cpp", "transpose"),
    "aten_upsample_bilinear2d": ("aten/src/ATen/native/UpSampleBilinear2d.cpp", "upsample_bilinear2d"),
    "aten_upsample_nearest2d": ("aten/src/ATen/native/UpSampleNearest2d.cpp", "upsample_nearest2d"),
}

# Mechanically generated scalar specializations keep their provenance in a
# CSV so adding an extraction does not require editing this viewer by hand.
for _aten_generated_provenance in sorted(
    ATEN_C_ROOT.glob("generated*_provenance.csv")
):
    with _aten_generated_provenance.open(newline="") as _stream:
        for _row in csv.DictReader(_stream):
            ATEN_C_PROVENANCE[_row["kernel"]] = (
                _row["source"], _row.get("token") or None
            )

ATEN_C_MATCH_ASSESSMENT: dict[str, str] = {
    "aten_adaptive_avg_pool2d": (
        "generic regular-window match lowered to depthwise cuDNN convolution"
    ),
    "aten_adaptive_avg_pool2d_cpu": "semantic adaptive-pool match; regular shape executes cuDNN Resample",
    "aten_adaptive_avg_pool2d_backward_cpu": "semantic adaptive-pool match; regular shape executes cuDNN Resample backward",
    "aten_adaptive_avg_pool3d": "semantic regular 3D adaptive-pool match lowered to cuDNN Resample",
    "aten_adaptive_avg_pool3d_cpu": "semantic adaptive-pool match; variable-window shape uses exact fallback",
    "aten_adaptive_avg_pool3d_backward_cpu": "semantic adaptive-pool match; variable-window shape uses exact fallback",
    "aten_adaptive_max_pool1d_cpu": "semantic adaptive-max match; variable-window shape uses exact ATen-index fallback",
    "aten_adaptive_max_pool2d_cpu": "hybrid cuDNN max values plus exact ATen absolute-index materialization",
    "aten_adaptive_max_pool2d_backward_cpu": "semantic saved-index scatter; exact ATen-index fallback",
    "aten_adaptive_max_pool3d_cpu": "semantic adaptive-max match; variable-window shape uses exact fallback",
    "aten_adaptive_max_pool3d_backward_cpu": "semantic saved-index scatter; exact ATen-index fallback",
    "aten_adaptive_max_pool3d_legacy_cpu": "semantic legacy adaptive-max match; variable-window shape uses exact fallback",
    "aten_adaptive_max_pool3d_legacy_backward_cpu": "semantic legacy saved-index scatter; exact fallback",
    "aten_avg_pool2d": "fixed-window average pooling lowered to cuDNN Resample forward",
    "aten_avg_pool2d_cpu": "fixed-window average pooling lowered to cuDNN Resample forward",
    "aten_avg_pool2d_backward_cpu": "fixed-window average pooling lowered to cuDNN Resample backward",
    "aten_avg_pool3d": "fixed-window 3D average pooling lowered to cuDNN Resample forward",
    "aten_avg_pool3d_cpu": "fixed-window 3D average pooling lowered to cuDNN Resample forward",
    "aten_avg_pool3d_backward_cpu": "fixed-window 3D average pooling lowered to cuDNN Resample backward",
    "aten_batch_norm_backward_cpu": "saved-statistics derivative lowered to cuDNN BatchNormalizationBackward",
    "aten_batch_norm_backward_template_cpu": "input-gradient-only derivative lowered to cuDNN BatchNormalizationBackward",
    "aten_binary_cross_entropy": "external logf calls retain a residual loop",
    "aten_bmm": "no true batched-GEMM definition",
    "aten_channel_shuffle": "layout transform lowered through generic cuTENSOR modes/strides",
    "aten_clamp": "no standalone clamp definition",
    "aten_conv1d": "no 1D-convolution definition",
    "aten_conv3d": "shape/template gap in current 3D-convolution definitions",
    "aten_conv_transpose2d": "no transposed-convolution definition",
    "aten_cross": "three elementwise stages; no cross-product composition",
    "aten_cumsum": "loop-carried inclusive sum lowered to CUB DeviceScan",
    "aten_elu": "no standalone ELU definition",
    "aten_embedding": "indexed gather retains residual loops",
    "aten_gelu": "valid custom CUDA GELU route",
    "aten_hardsigmoid": "no standalone hard-sigmoid definition",
    "aten_hardswish": "no standalone hard-swish definition",
    "aten_hardtanh": "no standalone hard-tanh definition",
    "aten_im2col": "affine window materialization lowered through generic cuTENSOR strides",
    "aten_l1_loss": "no L1 reduction composition",
    "aten_layer_norm": "mean/variance/affine composition not in library",
    "aten_leaky_relu": "no standalone leaky-ReLU definition",
    "aten_lerp": "no linear-interpolation definition",
    "aten_mean": "reduction-plus-scale composition not in library",
    "aten_mse_loss": "square-difference plus mean composition not in library",
    "aten_outer": "partial: only output zero-initialization matched",
    "aten_pixel_shuffle": "rank-reduced reshape/permutation validated on Jetson",
    "aten_prod": "no product-reduction definition",
    "aten_reflection_pad2d": "no reflection-padding definition",
    "aten_relu": "no standalone ReLU definition",
    "aten_replication_pad2d": "no replication-padding definition",
    "aten_sigmoid": "no standalone sigmoid definition",
    "aten_silu": "no standalone SiLU definition",
    "aten_softplus": "external logf/expf and branch retain a residual loop",
    "aten_sum": "partial: only output zero-initialization matched",
    "aten_tanh": "no standalone tanh definition",
    "aten_transpose_copy": "permuted indexing map lowered through generic cuTENSOR modes",
    "aten_upsample_bilinear2d": "fixed cuDNN FP32 2x half-pixel bilinear resample",
    "aten_upsample_nearest2d": "no nearest-neighbor resampling definition",
}

ATEN_C_UNSAFE_MATCHES: set[str] = set()

# These probes come from large upstream source files. Keep them in the IR
# explorer, but avoid embedding the full original files in Compiler Explorer
# deep links, which makes the generated index impractically large.
WHISPER_OPS_IR_ONLY: set[str] = {
    "whisper_quantize_q4_0_ref",
    "whisper_decode_residue",
    "whisper_inverse_mdct",
}

STENCIL_CONV2D_KERNELS: dict[str, tuple[str, str]] = {
    "box3x3":          ("stencil_conv2d_3x3.c", "kernel_stencil_box3x3"),
    "gaussian3x3":     ("stencil_conv2d_3x3.c", "kernel_stencil_gaussian3x3"),
    "sobel_x3x3":      ("stencil_conv2d_3x3.c", "kernel_stencil_sobel_x3x3"),
    "sobel_y3x3":      ("stencil_conv2d_3x3.c", "kernel_stencil_sobel_y3x3"),
    "laplacian4_3x3":  ("stencil_conv2d_3x3.c", "kernel_stencil_laplacian4_3x3"),
    "laplacian8_3x3":  ("stencil_conv2d_3x3.c", "kernel_stencil_laplacian8_3x3"),
    "sharpen3x3":      ("stencil_conv2d_3x3.c", "kernel_stencil_sharpen3x3"),
    "emboss3x3":       ("stencil_conv2d_3x3.c", "kernel_stencil_emboss3x3"),
    "box5x5":          ("stencil_conv2d_3x3.c", "kernel_stencil_box5x5"),
    "gaussian5x5":     ("stencil_conv2d_3x3.c", "kernel_stencil_gaussian5x5"),
    "sobel_x5x5":      ("stencil_conv2d_3x3.c", "kernel_stencil_sobel_x5x5"),
    "sobel_y5x5":      ("stencil_conv2d_3x3.c", "kernel_stencil_sobel_y5x5"),
    "laplacian5x5":    ("stencil_conv2d_3x3.c", "kernel_stencil_laplacian5x5"),
    "sharpen5x5":      ("stencil_conv2d_3x3.c", "kernel_stencil_sharpen5x5"),
    "emboss5x5":       ("stencil_conv2d_3x3.c", "kernel_stencil_emboss5x5"),
    "box7x7":          ("stencil_conv2d_3x3.c", "kernel_stencil_box7x7"),
}

STENCIL_CONV2D_ORDER = list(STENCIL_CONV2D_KERNELS.keys())

STENCIL_CONV2D_DISPLAY_NAMES: dict[str, str] = {
    "box3x3":          "box blur 3x3",
    "gaussian3x3":     "Gaussian blur 3x3",
    "sobel_x3x3":      "Sobel X 3x3",
    "sobel_y3x3":      "Sobel Y 3x3",
    "laplacian4_3x3":  "Laplacian 4-neighbor 3x3",
    "laplacian8_3x3":  "Laplacian 8-neighbor 3x3",
    "sharpen3x3":      "sharpen 3x3",
    "emboss3x3":       "emboss 3x3",
    "box5x5":          "box blur 5x5",
    "gaussian5x5":     "Gaussian blur 5x5",
    "sobel_x5x5":      "Sobel X 5x5",
    "sobel_y5x5":      "Sobel Y 5x5",
    "laplacian5x5":    "Laplacian 5x5",
    "sharpen5x5":      "sharpen 5x5",
    "emboss5x5":       "emboss 5x5",
    "box7x7":          "box blur 7x7",
}

# llm.c (karpathy/llm.c) leaf forward/backward kernels in train_gpt2.c. These
# are the building blocks of GPT-2 inference + training. Skip the tiled
# matmul_forward in favour of matmul_forward_naive (the 4-loop reference).
LLMC_KERNELS: dict[str, tuple[str, str]] = {
    "encoder-fwd":              ("train_gpt2.c", "encoder_forward"),
    "encoder-bwd":              ("train_gpt2.c", "encoder_backward"),
    "layernorm-fwd":            ("train_gpt2.c", "layernorm_forward"),
    "layernorm-bwd":            ("train_gpt2.c", "layernorm_backward"),
    "matmul-fwd-naive":         ("train_gpt2.c", "matmul_forward_naive"),
    "matmul-bwd":               ("train_gpt2.c", "matmul_backward"),
    "attention-fwd":            ("train_gpt2.c", "attention_forward"),
    "attention-bwd":            ("train_gpt2.c", "attention_backward"),
    "gelu-fwd":                 ("train_gpt2.c", "gelu_forward"),
    "gelu-bwd":                 ("train_gpt2.c", "gelu_backward"),
    "residual-fwd":             ("train_gpt2.c", "residual_forward"),
    "residual-bwd":             ("train_gpt2.c", "residual_backward"),
    "softmax-fwd":              ("train_gpt2.c", "softmax_forward"),
    "crossentropy-fwd":         ("train_gpt2.c", "crossentropy_forward"),
    "crossentropy-softmax-bwd": ("train_gpt2.c", "crossentropy_softmax_backward"),
}

# darknet (pjreddie) — CPU reference implementation of CNN layers used by
# YOLO + ResNet configurations. We bake every .c file in src/ with
# cgeist --function='*' and inlining enabled; the matcher then runs against
# each file's debuferized output. Most files are framework code (parser, list,
# image, network) with no compute bodies. The actual numerical hot spot
# is src/gemm.c which contains the naive C gemm_nn/nt/tn/tt variants;
# everything else either fails to lift (struct-heavy code, IfStmt
# limitations in cgeist) or produces linalg.generic ops the matcher's
# current library doesn't recognise (pooling, batchnorm, RNN gates, ...).
#
# This is intentionally a "matcher coverage survey" rather than a
# silicon-target list — its purpose is to enumerate which deep-learning
# layer kernels we'd need new matcher templates to cover. See the per-
# file notes for which pattern each unmatched file has.
DARKNET_KERNELS: dict[str, tuple[str, str]] = {
    "activation_layer":      ("src/activation_layer.c",      "*"),
    "activations":           ("src/activations.c",           "*"),
    "avgpool_layer":         ("src/avgpool_layer.c",         "*"),
    "batchnorm_layer":       ("src/batchnorm_layer.c",       "*"),
    "blas":                  ("src/blas.c",                  "*"),
    "box":                   ("src/box.c",                   "*"),
    "col2im":                ("src/col2im.c",                "*"),
    "compare":               ("src/compare.c",               "*"),
    "connected_layer":       ("src/connected_layer.c",       "*"),
    "convolutional_layer":   ("src/convolutional_layer.c",   "*"),
    "cost_layer":            ("src/cost_layer.c",            "*"),
    "crnn_layer":            ("src/crnn_layer.c",            "*"),
    "crop_layer":            ("src/crop_layer.c",            "*"),
    "data":                  ("src/data.c",                  "*"),
    "deconvolutional_layer": ("src/deconvolutional_layer.c", "*"),
    "demo":                  ("src/demo.c",                  "*"),
    "detection_layer":       ("src/detection_layer.c",       "*"),
    "dropout_layer":         ("src/dropout_layer.c",         "*"),
    "gemm":                  ("src/gemm.c",                  "*"),
    "gru_layer":             ("src/gru_layer.c",             "*"),
    "im2col":                ("src/im2col.c",                "*"),
    "image":                 ("src/image.c",                 "*"),
    "iseg_layer":            ("src/iseg_layer.c",            "*"),
    "l2norm_layer":          ("src/l2norm_layer.c",          "*"),
    "layer":                 ("src/layer.c",                 "*"),
    "list":                  ("src/list.c",                  "*"),
    "local_layer":           ("src/local_layer.c",           "*"),
    "logistic_layer":        ("src/logistic_layer.c",        "*"),
    "lstm_layer":            ("src/lstm_layer.c",            "*"),
    "matrix":                ("src/matrix.c",                "*"),
    "maxpool_layer":         ("src/maxpool_layer.c",         "*"),
    "network":               ("src/network.c",               "*"),
    "normalization_layer":   ("src/normalization_layer.c",   "*"),
    "option_list":           ("src/option_list.c",           "*"),
    "parser":                ("src/parser.c",                "*"),
    "region_layer":          ("src/region_layer.c",          "*"),
    "reorg_layer":           ("src/reorg_layer.c",           "*"),
    "rnn_layer":             ("src/rnn_layer.c",             "*"),
    "route_layer":           ("src/route_layer.c",           "*"),
    "shortcut_layer":        ("src/shortcut_layer.c",        "*"),
    "softmax_layer":         ("src/softmax_layer.c",         "*"),
    "tree":                  ("src/tree.c",                  "*"),
    "upsample_layer":        ("src/upsample_layer.c",        "*"),
    "utils":                 ("src/utils.c",                 "*"),
    "yolo_layer":            ("src/yolo_layer.c",            "*"),
}

DARKNET_NOTES: dict[str, tuple[str, str]] = {
    # The 1 file that produces matches today
    "gemm":                  ("highly parallel",   "Classic dense gemm + axpy variants; gemm_nt/tt match @cublasDgemm_alpha_only; gemm_nn/tn match @cublasDaxpy (inner-loop scalar-hoisted form not composed up to gemm)"),
    # Compute-pattern files that raise OK but don't match — the matcher templates we're missing
    "activation_layer":      ("pointwise",         "Activation forward (ReLU/leaky/etc.) — pointwise; no template"),
    "activations":           ("pointwise",         "Activation primitives — pointwise; no template"),
    "avgpool_layer":         ("partial parallel",  "Average pooling — windowed reduction; no template"),
    "col2im":                ("pointwise",         "Column-to-image reshape — strided scatter; no template"),
    "connected_layer":       ("highly parallel",   "Dense (fully-connected) layer — gemv shape with bias; 16 generics but matcher's gemv composition isn't firing"),
    "cost_layer":            ("partial parallel",  "Loss computation — pointwise + reduction; no template"),
    "crop_layer":            ("pointwise",         "Image crop — pointwise; no template"),
    "deconvolutional_layer": ("highly parallel",   "Transposed conv via col2im — 20 generics; same matcher gap as conv (im2col-based gemm)"),
    "dropout_layer":         ("pointwise",         "Dropout mask multiply — pointwise; no template"),
    "gru_layer":             ("partial parallel",  "GRU RNN gates — 9 generics; matcher has no recurrent-cell composition"),
    "im2col":                ("pointwise",         "Image-to-column reshape — strided gather; raised but no compute body to match"),
    "l2norm_layer":          ("partial parallel",  "L2 normalization — reduction + divide; no template (similar to rmsnorm)"),
    "local_layer":           ("highly parallel",   "Locally-connected (per-position weights) — 6 generics; matcher gap (no shared filter)"),
    "logistic_layer":        ("pointwise",         "Sigmoid + binary cross-entropy — pointwise + reduction; no template"),
    "maxpool_layer":         ("partial parallel",  "Max pooling — windowed reduction (3 generics); matcher has no pooling composition"),
    "normalization_layer":   ("partial parallel",  "Local response normalization — reduction + divide (4 generics); no template"),
    "reorg_layer":           ("pointwise",         "Spatial reorganisation — pointwise reshape; no template"),
    "route_layer":           ("pointwise",         "Concatenation across feature maps — strided memcpy; no template"),
    "shortcut_layer":        ("pointwise",         "Residual add (x += shortcut) — pointwise; matcher-gap (same as llmc residual-fwd)"),
    "softmax_layer":         ("partial parallel",  "Softmax — 3-step composition; the llama2/llmc softmax template exists but this layer has different surrounding control flow"),
    "upsample_layer":        ("pointwise",         "Nearest-neighbour upsample — strided broadcast; no template"),
    # cgeist failures — framework code, no compute to match anyway
    "blas":                  ("",                   "cgeist failure — header includes choke (math.h + glibc-specific intrinsics)"),
    "box":                   ("",                   "Raise pass fails on memref-of-memref shape from box-list operations"),
    "compare":               ("",                   "cgeist failure — variadic ranking helpers"),
    "convolutional_layer":   ("highly parallel",   "Raise fails — body is mostly external-call dispatch (im2col_cpu + gemm); the actual compute lives in gemm.c which DOES match"),
    "crnn_layer":            ("",                   "cgeist failure — recurrent layer struct uses function pointers"),
    "data":                  ("",                   "cgeist failure — pthread + libc-heavy data-loading code"),
    "demo":                  ("",                   "cgeist failure — OpenCV display loop (requires cv::Mat headers)"),
    "detection_layer":       ("",                   "cgeist failure — IfStmt lowering bug on the per-anchor confidence branches"),
    "image":                 ("",                   "cgeist failure — stbi-style image loaders"),
    "iseg_layer":            ("",                   "cgeist failure — IfStmt lowering bug (instance-segmentation post-processing)"),
    "lstm_layer":            ("",                   "cgeist failure — recurrent-cell struct + function pointers"),
    "list":                  ("",                   "cgeist failure — linked-list manipulation; no compute"),
    "matrix":                ("",                   "cgeist failure — IfStmt on shape validation"),
    "network":               ("",                   "cgeist failure — FunctionDecl issue (function-pointer-of-layer.forward_layer dispatch)"),
    "option_list":           ("",                   "cgeist failure — header includes"),
    "parser":                ("",                   "cgeist failure — sscanf-heavy .cfg parser, header includes"),
    "region_layer":          ("",                   "cgeist failure — BinaryOperator on the YOLO grid-cell branching"),
    "rnn_layer":             ("",                   "cgeist failure — recurrent-cell struct"),
    "utils":                 ("",                   "cgeist failure — exits + abort macros, no compute"),
    "yolo_layer":            ("",                   "cgeist failure — IfStmt on YOLO loss-mask branches"),
    # files that raise OK and produce zero linalg.generic — no compute
    "activation_layer":      ("pointwise",         "Activation forward (ReLU/leaky/etc.) — pointwise; no template"),
    "layer":                 ("",                   "Layer-struct allocator + free — no compute"),
    "tree":                  ("",                   "Hierarchical-class tree manipulation — no compute"),
}

DARKNET_BLOCKERS: dict[str, tuple[str, str]] = {
    "gemm":                  ("none",              ""),
    "activation_layer":      ("matcher-gap",       "pointwise activation; no axpy-like template fires"),
    "activations":           ("matcher-gap",       "pointwise"),
    "avgpool_layer":         ("matcher-gap",       "pooling composition not in library"),
    "col2im":                ("matcher-gap",       "strided scatter"),
    "connected_layer":       ("matcher-gap",       "gemv composition gap (matrix index has bias term)"),
    "cost_layer":            ("matcher-gap",       "loss = reduction over pointwise body"),
    "crop_layer":            ("matcher-gap",       "pointwise"),
    "deconvolutional_layer": ("matcher-gap",       "transposed conv (col2im+gemm)"),
    "dropout_layer":         ("matcher-gap",       "pointwise"),
    "gru_layer":             ("matcher-gap",       "RNN gates"),
    "im2col":                ("none",              "Strided gather raises but has no compute body"),
    "l2norm_layer":          ("matcher-gap",       "norm + divide"),
    "local_layer":           ("matcher-gap",       "per-position weights"),
    "logistic_layer":        ("matcher-gap",       "sigmoid+BCE"),
    "maxpool_layer":         ("matcher-gap",       "pooling"),
    "normalization_layer":   ("matcher-gap",       "LRN"),
    "reorg_layer":           ("matcher-gap",       "spatial reshape"),
    "route_layer":           ("matcher-gap",       "concat"),
    "shortcut_layer":        ("matcher-gap",       "residual add"),
    "softmax_layer":         ("matcher-gap",       "softmax (this layer's surrounding control flow defeats the existing softmax template)"),
    "upsample_layer":        ("matcher-gap",       "upsample"),
    "blas":                  ("cgeist-gap",        "header inclusion failure"),
    "box":                   ("debuf-bug",         "memref-of-memref shape"),
    "compare":               ("cgeist-gap",        "variadic ranking"),
    "convolutional_layer":   ("matcher-gap",       "body is mostly external calls; real compute is in gemm.c"),
    "crnn_layer":            ("cgeist-gap",        "RNN struct + function pointers"),
    "data":                  ("cgeist-gap",        "pthread + libc"),
    "demo":                  ("cgeist-gap",        "OpenCV"),
    "detection_layer":       ("cgeist-gap",        "IfStmt bug"),
    "image":                 ("cgeist-gap",        "stbi-style loader"),
    "iseg_layer":            ("cgeist-gap",        "IfStmt bug"),
    "lstm_layer":            ("cgeist-gap",        "RNN struct"),
    "list":                  ("none",              "linked list, no compute"),
    "matrix":                ("cgeist-gap",        "IfStmt"),
    "network":               ("cgeist-gap",        "function-pointer dispatch"),
    "option_list":           ("cgeist-gap",        "header includes"),
    "parser":                ("cgeist-gap",        "sscanf-heavy"),
    "region_layer":          ("cgeist-gap",        "BinaryOperator on grid branches"),
    "rnn_layer":             ("cgeist-gap",        "RNN struct"),
    "utils":                 ("none",              "no compute"),
    "yolo_layer":            ("cgeist-gap",        "IfStmt bug"),
    "layer":                 ("none",              "allocator only"),
    "tree":                  ("debuf-bug",         "no compute pattern"),
}

# Per-NPB-kernel parallelism + characterisation notes.
NPB_NOTES: dict[str, tuple[str, str]] = {
    "bt-add":      ("highly parallel",   "BT vector add over 4D field — pure elemwise, fully parallel"),
    "ft-evolve":   ("highly parallel",   "FT timestep multiply — parallel but uses ex[indexmap[...]] gather; raise refuses indirect index"),
    "lu-l2norm":   ("highly parallel",   "LU L2 norm over 4D field — reduction over the spatial axes"),
    "mg-psinv":    ("highly parallel",   "MG smoother — 27-point stencil via per-row r1/r2 scratch arrays; outer i3/i2 hold scratch state"),
    "mg-resid":    ("highly parallel",   "MG residual r = v - Au — same 27-point stencil shape as psinv"),
    "mg-norm2u3":  ("highly parallel",   "MG L2 + L∞ combined norm — mixed sum+max reductions in one loop; raise pass can't fuse"),
    "mg-rprj3":    ("highly parallel",   "MG restriction (trilinear FE projection) — coarse-grid 2x downsample"),
}

# llama2.c numeric kernels — the building blocks of LLM forward pass.
LLAMA2C_NOTES: dict[str, tuple[str, str]] = {
    "matmul":   ("highly parallel",   "dense gemv (W·x = xout); single linalg.generic after raise"),
    "rmsnorm":  ("highly parallel",   "ss = mean(x²) + eps then o = weight·x/√ss; reduction + parallel scale"),
    "softmax":  ("partial parallel",  "max-shift then exp + sum then divide; three reduction/parallel phases"),
}

LLAMA_FORWARD_NOTES: dict[str, tuple[str, str]] = {
    "token_embedding":        ("highly parallel",  "embedding row copy for one token"),
    "attention_rmsnorm":      ("highly parallel",  "attention RMSNorm; mean-square reduction + weighted scale"),
    "qkv_projection":         ("highly parallel",  "Q/K/V dense projections from normalized hidden state"),
    "rope_interleaved":       ("partial parallel", "exact interleaved RoPE layout; still leaves loops today"),
    "rope_split":             ("highly parallel",  "raise-friendly split even/odd RoPE form"),
    "kv_cache_rw":            ("highly parallel",  "KV cache write at current position plus full cache read"),
    "attention_scores":       ("highly parallel",  "Q·K score reduction over per-head dimensions"),
    "attention_mask_if":      ("partial parallel", "branchy causal mask; still contains an if/loop shape"),
    "attention_mask_select":  ("highly parallel",  "branchless select-form causal mask"),
    "attention_softmax":      ("partial parallel", "max-shift softmax over the active sequence row"),
    "attention_output":       ("highly parallel",  "weighted sum over V cache"),
    "output_projection":      ("highly parallel",  "attention output projection GEMV"),
    "residual_add":           ("highly parallel",  "elementwise residual add"),
    "ffn_rmsnorm":            ("highly parallel",  "FFN RMSNorm; same shape as attention RMSNorm"),
    "gate_up_projection":     ("highly parallel",  "gate/up FFN projections"),
    "swiglu":                 ("highly parallel",  "elementwise SiLU(gate) * up"),
    "down_projection":        ("highly parallel",  "FFN down projection GEMV"),
    "final_rmsnorm":          ("highly parallel",  "final RMSNorm before logits"),
    "lm_head_projection":     ("highly parallel",  "lm_head GEMV to logits"),
    "extended_forward":       ("partial parallel", "one-token, one-layer Llama-style forward fixture combining the raised pieces"),
}

WHISPER_OPS_NOTES: dict[str, tuple[str, str]] = {
    "whisper_vec_dot":      ("highly parallel",  "dot-product reduction used by ggml vec_dot / matvec-style projection kernels"),
    "whisper_vec_softmax":  ("partial parallel", "inner softmax exp+sum loop with caller-provided max; reduction plus output write"),
    "whisper_softmax_full": ("partial parallel", "max-reduce, exp+sum, and normalize phases for attention softmax"),
    "whisper_rms_norm":     ("partial parallel", "mean-square reduction followed by parallel scale; RMSNorm-style normalization"),
    "whisper_gelu":         ("highly parallel",  "elementwise transformer activation; raises through math.tanh into tensor linalg"),
    "whisper_conv1d":       ("highly parallel",  "valid 1D convolution shape representing Whisper encoder-side audio conv"),
    "whisper_quantize_q4_0_ref": (
        "partial parallel",
        "GGML q4_0 reference quantizer: blockwise max reduction plus fp16 scale and packed 4-bit stores",
    ),
    "whisper_decode_residue": (
        "partial parallel",
        "Vorbis residue decode: codebook-driven channel residue reconstruction with dynamic classifications",
    ),
    "whisper_inverse_mdct": (
        "partial parallel",
        "Vorbis inverse MDCT transform with twiddle-factor pointer loops and staged butterfly helpers",
    ),
}

STENCIL_CONV2D_NOTES: dict[str, tuple[str, str]] = {
    "box3x3":          ("highly parallel", "uniform 3x3 box blur written as a shifted-neighbour stencil; tensor path uses generalized ntap"),
    "gaussian3x3":     ("highly parallel", "separable-looking 3x3 Gaussian coefficient stencil, matched by the tensor ntap path"),
    "sobel_x3x3":      ("highly parallel", "horizontal image-gradient stencil; unit coefficients are recovered by the matcher"),
    "sobel_y3x3":      ("highly parallel", "vertical image-gradient stencil; same 9 shifted input views as Sobel X"),
    "laplacian4_3x3":  ("highly parallel", "4-neighbour Laplacian finite-difference stencil embedded in a 3x3 kernel"),
    "laplacian8_3x3":  ("highly parallel", "8-neighbour Laplacian finite-difference stencil"),
    "sharpen3x3":      ("highly parallel", "classic image sharpen filter, center-heavy 3x3 stencil"),
    "emboss3x3":       ("highly parallel", "asymmetric emboss filter; still maps to cross-correlation semantics"),
    "box5x5":          ("highly parallel", "25-tap box filter; tensor path packs W[25] for the generalized ntap cuDNN route"),
    "gaussian5x5":     ("highly parallel", "separable 5x5 Gaussian coefficient stencil, matched by the generalized ntap path"),
    "sobel_x5x5":      ("highly parallel", "wider horizontal-gradient stencil with zero center column coefficients"),
    "sobel_y5x5":      ("highly parallel", "wider vertical-gradient stencil with zero center row coefficients"),
    "laplacian5x5":    ("highly parallel", "5x5 Laplacian / LoG-style finite-difference stencil"),
    "sharpen5x5":      ("highly parallel", "wider sharpen filter with center-heavy positive weights"),
    "emboss5x5":       ("highly parallel", "asymmetric 5x5 emboss filter mapped to cross-correlation semantics"),
    "box7x7":          ("highly parallel", "49-tap box filter; matched by the generalized packed-weight ntap cuDNN path"),
}

# llm.c kernel notes — GPT-2 building blocks. Most fwd kernels are highly
# parallel (B·T·OC or B·T·C parallel iter spaces); attention has a per-query
# softmax that introduces a reduction phase; encoder/gelu/crossentropy have
# data-dependent indexing or math.h ext-calls that block raise.
LLMC_NOTES: dict[str, tuple[str, str]] = {
    "encoder-fwd":              ("partial parallel",  "lookup wte[token]+wpe[pos]; data-dependent index blocks raise"),
    "encoder-bwd":              ("partial parallel",  "scatter-accumulate gradients into wte/wpe; indirect-index scatter"),
    "layernorm-fwd":            ("highly parallel",   "per-(B,T) row: mean + variance reductions then normalize + scale + bias"),
    "layernorm-bwd":            ("partial parallel",  "per-(B,T) row: 2 reductions for dnorm/dnorm_mean then accumulate dweight/dbias/dinp"),
    "matmul-fwd-naive":         ("highly parallel",   "4-loop reference matmul out[b,t,o] = sum_i inp[b,t,i]*weight[o,i] + bias[o]"),
    "matmul-bwd":               ("highly parallel",   "transpose matmuls for dinp, dweight, dbias"),
    "attention-fwd":            ("partial parallel",  "Q·Kᵀ → softmax → ·V; per-(B,T,h) parallel with two reductions (max, sum-exp)"),
    "attention-bwd":            ("partial parallel",  "backward through Q·Kᵀ/softmax/·V; gradient accumulation across heads"),
    "gelu-fwd":                 ("highly parallel",   "elementwise tanh-based gelu; calls tanhf — math.h ext call blocks raise"),
    "gelu-bwd":                 ("highly parallel",   "elementwise gelu derivative; calls tanhf + coshf — math.h ext calls"),
    "residual-fwd":             ("highly parallel",   "elementwise out = inp1 + inp2; single fully-parallel generic"),
    "residual-bwd":             ("highly parallel",   "elementwise dinp1 += dout; dinp2 += dout; two parallel generics"),
    "softmax-fwd":              ("partial parallel",  "per-(B,T) row softmax with max-shift; same 3-phase shape as llama2 softmax"),
    "crossentropy-fwd":         ("highly parallel",   "elementwise -log(probs[target[b,t]]); calls logf — math.h ext blocks raise"),
    "crossentropy-softmax-bwd": ("highly parallel",   "elementwise dlogits = (probs - onehot(target)) * dlosses"),
}

# Per-MachSuite-kernel parallelism + characterisation notes.
MACHSUITE_NOTES: dict[str, tuple[str, str]] = {
    "gemm-ncubed":   ("highly parallel",   "textbook 3-loop gemm with flat 1D indexing — lifts to single linalg.generic"),
    "gemm-blocked":  ("highly parallel",   "tiled gemm; blocking collapses, still matches GEMM"),
    "stencil2d":     ("highly parallel",   "9-tap 2D conv (3x3 filter), not jacobi-shaped — no matcher template yet"),
    "stencil3d":     ("highly parallel",   "3D stencil — 7-tap-ish, mostly matches"),
    "backprop":      ("partial parallel",  "neural-net backprop; many small generics, body shapes outside our library"),
    "nw":            ("serial",            "Needleman-Wunsch DP; row-by-row dependencies"),
    "fft-strided":   ("serial",            "bit-reversal addressing; outer shift loop non-affine"),
    "fft-transpose": ("partial parallel",  "transpose-based FFT; some stages parallel, others not"),
    "kmp":           ("serial",            "KMP string matching; backtracking, control-flow heavy"),
    "bfs-bulk":      ("serial",            "bulk-synchronous BFS; queue-based, non-affine"),
    "bfs-queue":     ("serial",            "queue-based BFS; non-affine indirect access"),
    "spmv-crs":      ("partial parallel",  "sparse matvec CRS — indirect indexing not raisable today"),
    "spmv-ellpack":  ("partial parallel",  "sparse matvec ELLPACK — same"),
    "sort-merge":    ("serial",            "merge sort; control flow heavy"),
    "sort-radix":    ("partial parallel",  "radix sort; counting + scatter; some stages affine"),
    "aes":           ("serial",            "byte-oriented AES; bit ops + sbox lookup; not numerical"),
    "md-grid":       ("highly parallel",   "molecular dynamics with cell-grid neighbour list"),
    "md-knn":        ("highly parallel",   "molecular dynamics with k-NN neighbour list"),
    "viterbi":       ("serial",            "Viterbi DP + arg-max; sequential along time"),
}

CE_BASE = "http://localhost:10240/"
CGEIST_NAME = "cgeist_aff"
POPT_NAME = "popt_full"
POPT_DISPLAY = "polygeist-opt: full (raise + lower-submap + debuferize)"


# =====================================================================
# Algorithm-blocker taxonomy: WHY each kernel ends up at FULL / PARTIAL /
# NONE. Derived from the per-kernel investigations done across sessions
# (see memory: scratch-row-carries, row-scratch-privatization-attempt,
# raise-to-linalg-gaps, raise-status-after-privatize). Each kernel below
# is tagged with one primary blocker. Tags:
#
#   none              — kernel fully lifts and matches; no blocker.
#   matcher-gap       — lifts to linalg.generic cleanly but the body
#                       shape isn't in the matcher library (fixable:
#                       add a CompositionEntry + kernel.defn).
#   runtime-gap       — matcher emits a kernel.launch form, but ABI lowering
#                       or the runtime shim for that exact symbol is pending.
#   t-loop            — body is parallel; outer "for t = 0..T" timestep
#                       loop is genuinely serial (stencils — body of one
#                       timestep reads the previous timestep's output).
#                       Correct partial-lift; no fix needed.
#   serial-recurrence — outer k/i loop carries data across iterations
#                       (factorizations, DPs, recurrences). Fundamentally
#                       non-parallel; can't be lifted further.
#   scratch-carry     — hand-CSE'd rank-1 scratch row used to share
#                       cross-axis arithmetic between two sibling inner
#                       loops within one outer iteration. The outer
#                       loops are parallel in principle; the shared
#                       scratch hides that from the raise pass. FIXABLE
#                       — see docs/row_scratch_privatization_failures.md.
#   indirect-index    — data-dependent array index (e.g.
#                       `ex[t * indexmap[k]]`). Needs gather semantics
#                       in linalg.generic; not supported today.
#   mixed-reductions  — single loop computes two reductions with
#                       different operators (e.g. sum + max). The
#                       raise pass currently rejects.
#   non-affine        — bit-shift loops, sparse indirect indexing,
#                       backtracking, control-flow-heavy code.
#                       Genuinely outside the affine model.
#   cgeist-frontend   — cgeist itself fails to parse / emit MLIR. Out
#                       of pipeline scope.
#   debuf-bug         — known dominance-class bug in the debufferize
#                       pass (gramschmidt-class).
# =====================================================================

BLOCKER_TAXONOMY: dict[str, tuple[str, str]] = {
    # tag → (one-liner label, longer explanation)
    "none":              ("clean lift",
                          "fully lifts to kernel.launch (or to linalg.generic + matched library entry)"),
    "matcher-gap":       ("matcher library gap",
                          "lifts to linalg.generic, but the body shape isn't in the matcher library yet"),
    "runtime-gap":       ("runtime ABI gap",
                          "matches to a kernel.launch symbol, but ABI lowering or the runtime shim for that exact symbol is still pending"),
    "t-loop":            ("serial T loop",
                          "stencil-style: body parallel, outer time/step loop must be sequential"),
    "serial-recurrence": ("serial recurrence",
                          "factorization / DP / recurrence — outer iterations have genuine cross-iter data dependencies"),
    "scratch-carry":     ("scratch row carry (FIXABLE)",
                          "hand-CSE'd rank-1 row scratch shared between sibling inner loops; needs the row-privatization pass to land"),
    "indirect-index":    ("data-dependent index (FIXABLE)",
                          "indirect array index like ex[t*indexmap[i]]; needs gather support in linalg.generic"),
    "mixed-reductions":  ("mixed sum+max reductions",
                          "outer loop computes two reductions with different operators in one nest"),
    "non-affine":        ("non-affine access",
                          "bit-shift loop / sparse indirect / control-flow heavy — genuinely outside the affine model"),
    "cgeist-frontend":   ("cgeist front-end limit",
                          "cgeist itself doesn't parse the C cleanly (bit-heavy / struct-heavy / fn-pointer code)"),
    "debuf-bug":         ("debuf dominance bug",
                          "raise OK but debufferize hits the gramschmidt-class tensor.empty dominance issue"),
    "raise-fail":        ("raise pipeline failure",
                          "cgeist emits MLIR, but polygeist-opt fails before producing a raised linalg artifact"),
    "raise-crash":       ("polygeist-opt crash during raise",
                          "polygeist-opt segfaults in the raise pipeline; needs deeper investigation"),
    "no-linalg":         ("no linalg form",
                          "the function compiles, but the current raise path leaves imperative/control-flow or low-level LLVM structure instead of producing linalg.generic"),
    "ext-math-call":     ("math.h ext call in body (FIXABLE)",
                          "loop body calls tanhf / logf / coshf etc.; raise refuses to lift a generic whose body contains an external call. Fixable by teaching the frontend or a pre-pass to rewrite known math.h calls to math.* dialect ops"),
    "cudnn-dtype-gap":   ("cuDNN dtype not supported",
                          "MLIR pipeline (raise / match / ABI lowering / runtime shim ABI) is correct end-to-end, but the underlying library doesn't expose the requested dtype on this hardware. Today's hit: cuDNN's cudnnConvolutionForward does not support a pure INT32 input+filter+compute configuration on Ampere/Orin (returns CUDNN_STATUS_BAD_PARAM at descriptor setup); CUDNN_DATA_INT32 is only available as an accumulator type for INT8 inputs via the bias+activation API. Real fixes are out-of-pipeline: hand-written CUDA kernel via nvcc, INT8 quantisation path, or swap cuDNN for cutlass/CUB"),
    "cgeist-dtype-gap":  ("cgeist frontend dtype assert",
                          "cgeist itself can't parse the source dtype: BuiltinType `_Float16` / `__bf16` hits an `unhandled type` assertion in tools/cgeist/Lib/clang-mlir.cc:5830. Affects FP16 and BF16 conv2d sources — we never get an MLIR file to feed the rest of the pipeline. Fix is a small addition to the BuiltinType switch that maps clang's Half / BFloat16 to MLIR's f16 / bf16"),
    "partial-pipeline":  ("partial pipeline (matcher OK, downstream incomplete)",
                          "matcher + rewriter produce a clean kernel.launch op for this kernel, but the canonical defn / ABI lowering / runtime shim for the new library symbol haven't landed yet. Distinct from cudnn-dtype-gap (where the library is fundamentally unwilling) or matcher-gap (where the linalg body doesn't fingerprint). This is a 'in progress, scope-limited' state; the linalg → kernel.launch step is validated, the kernel.launch → func.call step is pending"),
}

# Per-kernel parallelism notes — how well the kernel's algorithm maps to GPU.
# Categories used in the index column:
#   highly parallel    — every iteration independent; embarrassingly parallel
#   parallel + T loop  — body parallel, but a sequential outer time/step loop remains
#   partial parallel   — significant parallel ops mixed with reductions / serial steps
#   serial             — fundamental cross-iteration dependencies; poor GPU fit
KERNEL_NOTES: dict[str, tuple[str, str]] = {
    # BLAS-shaped — fully parallel iter space.
    "gemm":          ("highly parallel",   "dense gemm, 3-loop parallel + reduction"),
    "gemver":        ("highly parallel",   "rank-2 update + gemv stages, all parallel"),
    "gesummv":       ("highly parallel",   "two gemvs + axpby, all parallel"),
    "atax":          ("highly parallel",   "y = A·x then t = Aᵀ·y, parallel"),
    "bicg":          ("highly parallel",   "s = Aᵀ·p and q = A·r, parallel"),
    "mvt":           ("highly parallel",   "x1 += A·y1; x2 += Aᵀ·y2, parallel"),
    "2mm":           ("highly parallel",   "two chained gemms, parallel"),
    "3mm":           ("highly parallel",   "three chained gemms, parallel"),
    "symm":          ("highly parallel",   "symmetric gemm (lower triangle), parallel"),
    "syrk":          ("highly parallel",   "symmetric rank-k update (lower triangle)"),
    "syr2k":         ("highly parallel",   "symmetric rank-2k update (lower triangle)"),
    "trmm":          ("highly parallel",
                      "triangular gemm — (i,j) parallel, k reduction; raise "
                      "splits the per-i body into 2 memref linalg ops which "
                      "the matcher can't see today (form-gated)"),

    # Stencils — body parallel, outer time loop is sequential.
    "jacobi-1d":     ("parallel + T loop",
                      "3-point 1D smoother; T steps sequential, inner parallel"),
    "jacobi-2d":     ("parallel + T loop",
                      "5-point 2D stencil; T steps sequential, inner parallel"),
    "heat-3d":       ("parallel + T loop",
                      "7-point 3D Laplacian; T steps sequential, inner highly parallel"),
    "fdtd-2d":       ("parallel + T loop",
                      "E/H field cross-updates; T steps sequential, inner parallel"),
    "adi":           ("parallel + T loop",
                      "alternating direction implicit; T+sweep loops sequential, "
                      "tridiagonal solves inside each sweep partially serial"),

    # Mixed: significant parallel ops plus reductions/serial constraints.
    "correlation":   ("partial parallel",
                      "mean + stddev reductions parallel; output is symmetric, "
                      "diagonal/off-diagonal phases mostly parallel"),
    "covariance":    ("partial parallel",
                      "mean reduction + centered outer product; mostly parallel "
                      "with reduction phases"),
    "doitgen":       ("partial parallel",
                      "inner contraction parallel; outer r-update sweep "
                      "has loop-carried scratch buffer"),
    "floyd-warshall":("partial parallel",
                      "all-pairs shortest path: (i,j) parallel per k, but k loop "
                      "is strictly sequential (each k uses previous k's distances)"),

    # Strictly serial / poor GPU fit.
    "cholesky":      ("serial",
                      "source recurrence is serial, but the complete algorithm "
                      "is recovered as a cuSOLVER DPOTRF library operation"),
    "lu":            ("serial",
                      "LU factorization — same column-sequential pattern as cholesky"),
    "ludcmp":        ("serial",
                      "LU + forward/back substitution — substitution phase is "
                      "strictly sequential"),
    "gramschmidt":   ("serial",
                      "modified Gram-Schmidt — each column projects against ALL "
                      "previously orthogonalized columns; strictly sequential"),
    "trisolv":       ("serial",
                      "source recurrence is sequential row-by-row, but the complete "
                      "algorithm is recovered as a cuBLAS DTRSV operation"),
    "durbin":        ("serial",
                      "Levinson-Durbin recurrence — O(N²) outer loop with full "
                      "scalar carry (α, β) between iterations; needs persistent "
                      "CUDA kernel with cooperative-groups sync"),
    "nussinov":      ("serial",
                      "RNA folding DP — sequential over diagonals, each cell "
                      "reads from prior diagonals"),
    "seidel-2d":     ("serial",
                      "Gauss-Seidel stencil — IN-PLACE writes within an inner "
                      "iteration, so each cell reads values updated earlier in "
                      "the SAME sweep; not naturally parallel"),
    "deriche":       ("serial",
                      "recursive IIR filter — output sample y[i] depends on "
                      "y[i-1..i-k]; sequential along the filter axis"),
}


# Per-kernel blocker classification: which BLOCKER_TAXONOMY tag applies,
# plus a kernel-specific one-liner. Used to render the "Blocker" column
# in the index and to power the taxonomy panel at the top of each section.
# Kernels not listed default to "none".
POLYBENCH_BLOCKERS: dict[str, tuple[str, str]] = {
    "gemm":          ("none",              ""),
    "syr2k":         ("none",              ""),
    "syrk":          ("none",              ""),
    "gesummv":       ("none",              ""),
    "gemver":        ("none",              ""),
    "symm":          ("matcher-gap",       "lifts, but one residual linalg.generic shape (symm-edge) isn't in library"),
    "trmm":          ("matcher-gap",       "lifts cleanly to cublasDtrmm; one residual triangular-edge body unmatched"),
    "atax":          ("none",              ""),
    "bicg":          ("none",              ""),
    "mvt":           ("none",              ""),
    "2mm":           ("none",              ""),
    "3mm":           ("none",              ""),
    "doitgen":       ("matcher-gap",       "lifts; the per-iter scratch-copy body isn't in the library"),
    "cholesky":      ("none",              ""),
    "gramschmidt":   ("serial-recurrence", "column-by-column modified Gram-Schmidt — column k+1 reads what column k just wrote"),
    "lu":            ("serial-recurrence", "LU factorization — pivot row k modifies rows >k that subsequent iterations consume"),
    "trisolv":       ("none",              ""),
    "ludcmp":        ("serial-recurrence", "LU + triangular solve — both phases have row-by-row carry"),
    "durbin":        ("serial-recurrence", "Levinson-Durbin recurrence — alpha/beta scalars carried across outer k iterations"),
    "heat-3d":       ("t-loop",            "7-point 3D Laplacian update; T-step outer loop is serial, inner 3D body parallel"),
    "jacobi-2d":     ("t-loop",            "5-point 2D smoother; T steps serial, inner 2D parallel"),
    "jacobi-1d":     ("t-loop",            "3-point 1D smoother; T steps serial, inner 1D parallel"),
    "fdtd-2d":       ("t-loop",            "Yee FDTD E/H field update; T steps serial, per-step body parallel"),
    "seidel-2d":     ("serial-recurrence", "Gauss-Seidel — in-place writes within one sweep; current cell reads values updated earlier in SAME sweep"),
    "adi":           ("t-loop",            "ADI (alternating direction implicit) — T-step outer, direction sweeps inside"),
    "floyd-warshall":("none",              ""),
    "deriche":       ("serial-recurrence", "recursive IIR filter — y[i] depends on y[i-1..i-k] along the filter axis"),
    "nussinov":      ("serial-recurrence", "RNA folding DP — diagonal sweep, each cell reads from prior diagonals"),
    "correlation":   ("scratch-carry",     "row-mean + variance accumulation; residual is the cross-pass scratch in cov-style outer loops"),
    "covariance":    ("scratch-carry",     "mean-centred outer product; residual is the cross-pass scratch state"),
}

MACHSUITE_BLOCKERS: dict[str, tuple[str, str]] = {
    "aes":           ("cgeist-frontend",   "byte-oriented AES with 256-entry sbox lookups; cgeist crashes parsing"),
    "backprop":      ("matcher-gap",       "lifts 36 linalg.generic ops; neural-net body shapes (matmul+bias+sigmoid) not in library"),
    "bfs-bulk":      ("cgeist-frontend",   "bulk-synchronous BFS with struct/queue manipulation; cgeist crashes"),
    "bfs-queue":     ("non-affine",        "queue-based BFS; level/horizon-driven iteration not affine"),
    "fft-strided":   ("non-affine",        "bit-reversal addressing: `for (span = N/2; span; span >>= 1)` — not affine"),
    "fft-transpose": ("non-affine",        "FFT butterflies with bit-reversed access patterns; partial body lifts but FFT shape outside model"),
    "gemm-ncubed":   ("none",              ""),
    "gemm-blocked":  ("matcher-gap",       "tiled gemm; collapses to a single linalg.generic but extra tiling loops survive"),
    "kmp":           ("non-affine",        "KMP string matching — backtracking on failure, control-flow heavy"),
    "md-grid":       ("cgeist-frontend",   "molecular dynamics with neighbour-list structs; cgeist crashes"),
    "md-knn":        ("debuf-bug",         "raises cleanly; debufferize hits the gramschmidt-class dominance bug"),
    "nw":            ("serial-recurrence", "Needleman-Wunsch alignment DP; row depends on previous row's cells"),
    "sort-merge":    ("cgeist-frontend",   "recursive merge sort; cgeist's analysis doesn't handle the recursion"),
    "sort-radix":    ("non-affine",        "radix sort with counting buckets; some bucket fills lift but the sort itself is non-affine"),
    "spmv-crs":      ("non-affine",        "sparse matvec CRS — indirect `cols[]` index into the values array"),
    "spmv-ellpack":  ("non-affine",        "same — sparse indirect addressing"),
    "stencil2d":     ("matcher-gap",       "9-tap 3x3 conv2d body; lifts cleanly but matcher has no conv2d-3x3 template"),
    "stencil3d":     ("none",              ""),
    "viterbi":       ("cgeist-frontend",   "Viterbi DP + arg-max; cgeist crashes on the array-of-struct probability table"),
}

NPB_BLOCKERS: dict[str, tuple[str, str]] = {
    "bt-add":        ("matcher-gap",       "4D elementwise add lifts cleanly; matcher's add templates are only 1D/2D today"),
    "ft-evolve":     ("indirect-index",    "ex[t*indexmap[k][j][i]] is a data-dependent index — raise pass refuses"),
    "lu-l2norm":     ("matcher-gap",       "inner sum-of-squares reduction lifts + matches; outer init loop is unmatched"),
    "mg-psinv":      ("scratch-carry",     "27-stencil via per-row r1/r2 scratch buffers; the scaffolded row-privatization pass would unblock"),
    "mg-resid":      ("scratch-carry",     "same shape as psinv"),
    "mg-rprj3":      ("scratch-carry",     "restriction operator with x1/y1 row scratch; same shape"),
    "mg-norm2u3":    ("mixed-reductions",  "combined L2 sum + L∞ max in one loop nest; raise rejects the dual-reduction iter_arg"),
}

# =====================================================================
# Jetson Orin silicon runtime measurements.
# =====================================================================
#
# For kernels that have actually been silicon-validated, one entry per
# (kernel, dataset) combination. The driver (scripts/correctness/
# polygeist_build.sh --target=jetson) cross-compiles two binaries from
# the same source:
#   - "gpu":  Polygeist-lifted kernel routed through cuDNN/cuBLAS via
#             our runtime shim. Time captured from polybench's built-in
#             timer (-DPOLYBENCH_TIME prints seconds to stdout).
#   - "cpu":  Plain aarch64-linux-gnu-gcc -O3 build of the same .c
#             linked with polybench.c; no Polygeist. Runs the textbook
#             C loop on Jetson's aarch64 CPU. Same timing method.
#
# Both shipped to Jetson Orin via the dev-box bounce and run; outputs
# diffed for correctness. Last-decimal FP precision drift at large sizes
# is normal — cuBLAS/cuDNN use tiled reductions with a different
# summation order than the textbook 3-loop, so e.g. `447.11` printed by
# the CPU might come out `447.10` on the GPU. PolyBench's reference
# considers these equivalent.
#
# Schema per entry:
#   { "size":        "MINI" | "LARGE" | "EXTRALARGE" (PolyBench dataset)
#                    or numeric string for non-PolyBench kernels
#     "gpu_s":       cuDNN/cuBLAS kernel time in seconds
#     "cpu_s":       aarch64 textbook-C kernel time in seconds
#     "correct":     "PASS" | "FP-noise" | "DIFF" | "ABORT"
#                    "FP-noise" = same algorithm, last-decimal rounding
#                    differs; functionally equivalent.
#   }
#
# All numbers below are from the *zero-copy* runtime path (cudaHostRegister
# polybench buffers + pass to cuBLAS via cudaHostGetDevicePointer; no
# cudaMalloc + cudaMemcpy bounce within Jetson's unified DRAM). MINI numbers
# dropped ~3× from the older malloc+copy runs; LARGE 25–30% for gemv-style
# kernels (bandwidth-bound), 1.5–2× for gemm-style (compute-bound but
# H↔D copy still meaningful).
#
# "notes" field (optional) is a short blurb shown in the explorer's Notes
# column — used to explain why a specific (kernel, size) entry has
# unexpected slowness or peculiar behaviour. Leave empty when no
# explanation needed (clean compute-bound wins, etc.).
JETSON_RUNTIMES: dict[str, list[dict]] = {
    "gemm": [
        {"size": "MINI",       "gpu_s": 0.029207, "cpu_s": 0.000009, "correct": "PASS",
         "notes": "Setup-bound: cuBLAS handle init + first cudaHostRegister dominate; 1024 flops too small to amortise"},
        {"size": "LARGE",      "gpu_s": 0.078334, "cpu_s": 0.631510, "correct": "FP-noise",
         "notes": ""},
        {"size": "EXTRALARGE", "gpu_s": 0.405161, "cpu_s": 7.138352, "correct": "FP-noise",
         "notes": ""},
    ],
    "2mm": [
        {"size": "MINI",       "gpu_s": 0.029192, "cpu_s": 0.000013, "correct": "PASS",
         "notes": "Setup-bound (same as gemm MINI)"},
        {"size": "LARGE",      "gpu_s": 0.095777, "cpu_s": 4.974022, "correct": "FP-noise",
         "notes": ""},
        {"size": "EXTRALARGE", "gpu_s": 0.466833, "cpu_s": 51.175102, "correct": "FP-noise",
         "notes": ""},
    ],
    "3mm": [
        {"size": "MINI",       "gpu_s": 0.030220, "cpu_s": 0.000020, "correct": "PASS",
         "notes": "Setup-bound (same as gemm MINI)"},
        {"size": "LARGE",      "gpu_s": 0.142634, "cpu_s": 5.883726, "correct": "PASS",
         "notes": ""},
        {"size": "EXTRALARGE", "gpu_s": 0.779139, "cpu_s": 61.008747, "correct": "PASS",
         "notes": ""},
    ],
    # SYRK dataset sizes: MINI=32², LARGE=2000²,
    # EXTRALARGE=4000². Matched as cublasDgemm (A·Aᵀ via OP_T).
    "syrk": [
        {"size": "MINI",       "gpu_s": 0.028913, "cpu_s": 0.000029, "correct": "PASS",
         "notes": "Setup-bound; A=B alias hits register cache early"},
        {"size": "LARGE",      "gpu_s": 0.289359, "cpu_s": 8.684662, "correct": "FP-noise",
         "notes": "cuBLAS dgemm with B=A pointer alias; native cublasDsyrk would be ~2× faster"},
        {"size": "EXTRALARGE", "gpu_s": 1.952076, "cpu_s": 69.050941, "correct": "FP-noise",
         "notes": "Same as LARGE — dgemm-emulated syrk"},
    ],
    # Convolution-2d dataset sizes per the benchmark header:
    # convolution-2d.h: MINI=64², LARGE=4096², EXTRALARGE=8192².
    # Matched as cudnnConvolution2D_9tap_f32. cuDNN is slower than the
    # CPU reference at all sizes because the 3×3 stencil has very low
    # arithmetic intensity (9 muls + 9 loads per output) — bandwidth-
    # bound, cuDNN setup overhead dominates. Numeric outputs match
    # (sorted-distribution identical to %0.2lf precision; differences
    # are rounding artifacts at the third decimal).
    "convolution-2d": [
        {"size": "MINI",       "gpu_s": 0.027487, "cpu_s": 0.000014, "correct": "FP-noise",
         "notes": "cuDNN descriptor + workspace setup ≫ actual 64² stencil; CPU 14 µs is just the math"},
        {"size": "LARGE",      "gpu_s": 0.139948, "cpu_s": 0.045992, "correct": "FP-noise",
         "notes": "3×3 stencil = 9 muls per output: arithmetic intensity ~1, bandwidth-bound; cuDNN can't reuse"},
        {"size": "EXTRALARGE", "gpu_s": 0.305478, "cpu_s": 0.186424, "correct": "FP-noise",
         "notes": "Same story as LARGE; CPU's wider memory subsystem competitive at this AI"},
    ],
    # atax + bicg — gemv-based kernels. The matcher's
    # transpose discriminator (rewriter inspects A's first indexing-map
    # output dim vs the output vector's first dim) now emits
    # @cublasDgemv vs @cublasDgemv_T, and the downstream lowering routes
    # each to the right cuBLAS op flag (CUBLAS_OP_T vs CUBLAS_OP_N).
    # Both kernels are now bit-exact MINI; LARGE uses the same routing
    # and should be equivalent (LARGE dump diff not run).
    # atax/bicg/mvt/gesummv/gemver — all five gemv-based
    # kernels now build + run cleanly after two consecutive fixes:
    #
    # 1. Matcher transpose discriminator: rewriter emits @cublasDgemv vs
    #    @cublasDgemv_T based on whether A's first indexing-map dim
    #    matches the output vector's dim. Downstream picks OP_T or OP_N.
    #
    # 2. -Dstatic=__attribute__((noipa)) in harness CFLAGS: prevents
    #    gcc -O3 from intraprocedurally deducing "kernel_*() preserves
    #    w0" and skipping the AArch64-mandated w0 reload before
    #    print_array. With static functions weakened via objcopy and
    #    replaced at link time, the cached IPA assumptions were wrong.
    #    Tagging the body as noipa keeps gcc honest.
    #
    # atax / bicg / gesummv: bit-exact GPU vs CPU dump (md5 match).
    # mvt / gemver: small numerical drift remains — separate matcher
    # bug where the accumulating init step isn't fissioned correctly
    # (kernel does x1 = A·y_1 with β=0 instead of x1 += A·y_1), so the
    # initial-value contribution from polybench init_array is dropped.
    "atax": [
        {"size": "MINI",  "gpu_s": 0.035718, "cpu_s": 0.000002, "correct": "PASS",
         "notes": "Setup-bound; 32² gemv is trivial"},
        {"size": "LARGE", "gpu_s": 0.243491, "cpu_s": 0.106797, "correct": "PASS",
         "notes": "cuBLAS dgemv(OP_T) strided reads; ~2% of peak DRAM BW; CPU 2× faster"},
    ],
    "bicg": [
        {"size": "MINI",  "gpu_s": 0.035921, "cpu_s": 0.000004, "correct": "PASS",
         "notes": "Setup-bound"},
        {"size": "LARGE", "gpu_s": 0.244687, "cpu_s": 0.293824, "correct": "PASS",
         "notes": "Bandwidth-bound dgemv; tied with CPU"},
    ],
    "gesummv": [
        {"size": "MINI",  "gpu_s": 0.032386, "cpu_s": 0.000004, "correct": "PASS",
         "notes": "Setup-bound"},
        {"size": "LARGE", "gpu_s": 0.242233, "cpu_s": 0.293041, "correct": "PASS",
         "notes": "Two streaming dgemvs through A, B; bandwidth-bound; marginal GPU win"},
    ],
    "mvt": [
        {"size": "MINI",  "gpu_s": 0.036262, "cpu_s": 0.000002, "correct": "DIFF",
         "notes": "Matcher missed accumulating init: kernel overwrites x1/x2 with β=0 instead of += . Numerically off, timing OK"},
    ],
    "gemver": [
        {"size": "MINI",  "gpu_s": 0.033820, "cpu_s": 0.000003, "correct": "DIFF",
         "notes": "Same matcher-fission bug as mvt: initial value dropped"},
        {"size": "LARGE", "gpu_s": 0.390434, "cpu_s": 0.575250, "correct": "DIFF",
         "notes": "Same bug; also 4 separate ops on A (2 gers + 2 gemvs) all bandwidth-bound; could be 5× faster with fused kernel"},
    ],
}

LLAMA_FORWARD_RUNTIMES: dict[str, list[dict]] = {
    "token_embedding": [
        {"size": "toy standalone warm", "raised": "host 0.0319 ms<br>device 0.0243 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Jetson Orin, REPEAT=50, first 5 iterations discarded"},
    ],
    "attention_rmsnorm": [
        {"size": "toy standalone warm", "raised": "host 0.0652 ms<br>device 0.0471 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "RMSNorm composition via runtime shim"},
    ],
    "qkv_projection": [
        {"size": "toy standalone warm", "raised": "host 0.0687 ms<br>device 0.0446 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Six emitted launches for split Q/K/V projection fixture"},
    ],
    "rope_interleaved": [
        {"size": "not run", "raised": "not raised", "reference": "not measured",
         "winner": "n/a", "notes": "Exact interleaved RoPE still leaves loops"},
    ],
    "rope_split": [
        {"size": "toy standalone warm", "raised": "host 0.1486 ms<br>device 0.0969 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Raise-friendly split even/odd RoPE"},
    ],
    "kv_cache_rw": [
        {"size": "toy standalone warm", "raised": "host 0.1244 ms<br>device 0.0908 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "KV write at current position plus cache read fixture"},
    ],
    "attention_scores": [
        {"size": "toy standalone warm", "raised": "host 0.0215 ms<br>device 0.0135 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "QK score reduction over heads/pairs"},
    ],
    "attention_mask_if": [
        {"size": "not run", "raised": "not raised", "reference": "not measured",
         "winner": "n/a", "notes": "Branchy mask variant still leaves if/loop IR"},
    ],
    "attention_mask_select": [
        {"size": "toy standalone warm", "raised": "host 0.0422 ms<br>device 0.0275 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Branchless causal mask"},
    ],
    "attention_softmax": [
        {"size": "toy standalone warm", "raised": "host 0.0552 ms<br>device 0.0384 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Max-shift softmax composition"},
    ],
    "attention_output": [
        {"size": "toy standalone warm", "raised": "host 0.0208 ms<br>device 0.0128 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Weighted sum over V cache"},
    ],
    "output_projection": [
        {"size": "toy standalone warm", "raised": "host 0.0252 ms<br>device 0.0157 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Attention output projection"},
    ],
    "residual_add": [
        {"size": "toy standalone warm", "raised": "host 0.0440 ms<br>device 0.0361 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Elementwise residual add"},
    ],
    "ffn_rmsnorm": [
        {"size": "toy standalone warm", "raised": "host 0.0652 ms<br>device 0.0465 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Same shape as attention RMSNorm"},
    ],
    "gate_up_projection": [
        {"size": "toy standalone warm", "raised": "host 0.0445 ms<br>device 0.0286 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Gate/up FFN projection fixture"},
    ],
    "swiglu": [
        {"size": "toy standalone warm", "raised": "host 0.0376 ms<br>device 0.0248 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Elementwise SiLU(gate) * up"},
    ],
    "down_projection": [
        {"size": "toy standalone warm", "raised": "host 0.0252 ms<br>device 0.0156 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "FFN down projection"},
    ],
    "final_rmsnorm": [
        {"size": "toy standalone warm", "raised": "host 0.0662 ms<br>device 0.0475 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "Final RMSNorm before logits"},
    ],
    "lm_head_projection": [
        {"size": "toy standalone warm", "raised": "host 0.0246 ms<br>device 0.0156 ms",
         "reference": "not measured", "winner": "raised-only",
         "notes": "LM head GEMV to logits"},
    ],
    "extended_forward": [
        {"size": "7B-size FP32 one layer", "raised": "external-library hybrid median 473.546 ms",
         "reference": "Orin CPU 342.515 ms<br>ggml CUDA 15.954 ms",
         "winner": "ggml 29.7x vs raised",
         "notes": "Three processes, 2 warmup + 10 measured each; 13 external launches plus 32 residual Linalg bodies; full 32,000-logit PASS"},
        {"size": "toy one layer warm", "raised": "host 0.719 ms<br>device 0.447 ms",
         "reference": "ggml CUDA host 0.098 ms", "winner": "ggml 7.3x",
         "notes": "MODEL_DIM=64, FFN_DIM=128, VOCAB=256, SEQ_LEN=32; useful for IR/debugging"},
    ],
}

STENCIL_CONV2D_RUNTIMES: dict[str, list[dict]] = {
    "box3x3": [
        {"size": "64x64 warm", "raised": "host 0.426 ms<br>device 0.0059 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.41999996"},
    ],
    "gaussian3x3": [
        {"size": "64x64 warm", "raised": "host 0.418 ms<br>device 0.0059 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.42000079"},
    ],
    "sobel_x3x3": [
        {"size": "64x64 warm", "raised": "host 0.425 ms<br>device 0.0059 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -5.88010693"},
    ],
    "sobel_y3x3": [
        {"size": "64x64 warm", "raised": "host 0.423 ms<br>device 0.0059 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum 4.11986542"},
    ],
    "laplacian4_3x3": [
        {"size": "64x64 warm", "raised": "host 0.166 ms<br>device 0.0417 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum 0.00000403"},
    ],
    "laplacian8_3x3": [
        {"size": "64x64 warm", "raised": "host 0.157 ms<br>device 0.0366 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.00000316"},
    ],
    "sharpen3x3": [
        {"size": "64x64 warm", "raised": "host 0.160 ms<br>device 0.0392 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.42001334"},
    ],
    "emboss3x3": [
        {"size": "64x64 warm", "raised": "host 0.162 ms<br>device 0.0399 ms",
         "reference": "cuDNN 3x3 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -1.74002242"},
    ],
    "box5x5": [
        {"size": "64x64 warm", "raised": "host 0.417 ms<br>device 0.0082 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.02519889"},
    ],
    "gaussian5x5": [
        {"size": "64x64 warm", "raised": "host 0.160 ms<br>device 0.0399 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -0.48238647"},
    ],
    "sobel_x5x5": [
        {"size": "64x64 warm", "raised": "host 0.155 ms<br>device 0.0400 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum 225.14791870"},
    ],
    "sobel_y5x5": [
        {"size": "64x64 warm", "raised": "host 0.156 ms<br>device 0.0369 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum 12.86828041"},
    ],
    "laplacian5x5": [
        {"size": "64x64 warm", "raised": "host 0.170 ms<br>device 0.0416 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -17.16963387"},
    ],
    "sharpen5x5": [
        {"size": "64x64 warm", "raised": "host 0.159 ms<br>device 0.0399 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum -2.78251743"},
    ],
    "emboss5x5": [
        {"size": "64x64 warm", "raised": "host 0.162 ms<br>device 0.0403 ms",
         "reference": "cuDNN 5x5 f32", "winner": "raised-only",
         "notes": "REPEAT=20, first 5 discarded; checksum 18.00988960"},
    ],
    "box7x7": [
        {"size": "64x64 warm", "raised": "host 0.433 ms<br>device 0.0109 ms",
         "reference": "cuDNN ntap f32", "winner": "raised-only",
         "notes": "tensor ntap, K=7, W[49] packed ABI; REPEAT=20, first 5 discarded; checksum 0.03551028"},
    ],
}

# llama2.c blockers — all three lift to linalg.generic cleanly. RMSNorm,
# softmax, and the tensor GEMV form now match/lower through runtime ABI paths;
# the whole tiny-forward fixture currently replaces RMSNorm + GEMV while
# leaving the softmax max/normalize tail as residual tensor code.
LLAMA2C_BLOCKERS: dict[str, tuple[str, str]] = {
    "matmul":   ("none", "Tensor GEMV form emits @cublasSgemv / @cublasSgemv_T and lowers to cuBLAS SGEMV; validated in the tiny forward fixture on Jetson."),
    "rmsnorm":  ("none", "2-step composition matches the ss = sum(x²) reduction + weighted-scale generic. Emits @rmsnorm_f32 for memref or @rmsnorm_f32_tensor after debufferize, lowering to polygeist_rmsnorm_f32."),
    "softmax":  ("none", "3-step composition matches max-reduce + fused exp+sum (multi-yield) + parallel divide. Emits @cudnnSoftmaxForward, lowers to polygeist_cudnn_softmax_forward_f32, and runs on Jetson through cudnnSoftmaxForward."),
}

LLAMA_FORWARD_BLOCKERS: dict[str, tuple[str, str]] = {
    "token_embedding":        ("none", ""),
    "attention_rmsnorm":      ("none", ""),
    "qkv_projection":         ("none", "Raises and matches as split GEMV/copy forms for the standalone fixture."),
    "rope_interleaved":       ("matcher-gap", "Exact interleaved layout still leaves residual loops; split even/odd RoPE is the currently matched form."),
    "rope_split":             ("none", ""),
    "kv_cache_rw":            ("none", ""),
    "attention_scores":       ("none", ""),
    "attention_mask_if":      ("matcher-gap", "Branchy if form still leaves residual control flow; branchless select form raises and matches."),
    "attention_mask_select":  ("none", ""),
    "attention_softmax":      ("none", ""),
    "attention_output":       ("none", ""),
    "output_projection":      ("none", ""),
    "residual_add":           ("none", ""),
    "ffn_rmsnorm":            ("none", ""),
    "gate_up_projection":     ("none", ""),
    "swiglu":                 ("none", ""),
    "down_projection":        ("none", ""),
    "final_rmsnorm":          ("none", ""),
    "lm_head_projection":     ("none", ""),
    "extended_forward":       ("matcher-gap", "Current audited fixture emits 13 external-library launches but leaves 32 Linalg bodies on CPU. Full-logit correctness passes; split RoPE and branchless masking are still source accommodations."),
}

WHISPER_OPS_BLOCKERS: dict[str, tuple[str, str]] = {
    "whisper_vec_dot":      ("none", "Raises to tensor linalg and matches the current dot-product template."),
    "whisper_vec_softmax":  ("matcher-gap", "Raises to tensor linalg in the scalar path. Direct ggml vec.cpp hits SIMD frontend issues, so this fixture captures the canonical math body."),
    "whisper_softmax_full": ("none", "Raises as max-reduce + exp/sum + normalize and matches the out-of-place cuDNN softmax template, including the multiply-by-reciprocal normalize spelling."),
    "whisper_rms_norm":     ("runtime-gap", "Matches the RMSNorm family as unweighted RMSNorm and emits the tensor launch form; runtime/library lowering for the unweighted ABI is still follow-up work."),
    "whisper_gelu":         ("runtime-gap", "Raises to one elementwise tensor linalg.generic with math.tanh and matches the GELU template; ABI lowering/runtime support for the GELU launch is still follow-up work."),
    "whisper_conv1d":       ("matcher-gap", "Raises and matches the per-output inner dot, but still leaves one output-position loop; full 1D conv composition/library routing is future matcher work."),
    "whisper_quantize_q4_0_ref": (
        "no-linalg",
        "Compiles and reaches the raise pipeline, but the result has no linalg.generic and still contains loops/ifs plus fp16 conversion, struct stores, and bit-packing.",
    ),
    "whisper_decode_residue": (
        "raise-fail",
        "cgeist emits MLIR, but raise fails on a mixed LLVM/memref load: llvm.load result must be an LLVM type with size, got memref<?xi8>.",
    ),
    "whisper_inverse_mdct": (
        "no-linalg",
        "The source is an algorithmic IMDCT kernel, but the current selected artifact contains no useful raised function body or linalg.generic.",
    ),
}

STENCIL_CONV2D_BLOCKERS: dict[str, tuple[str, str]] = {
    "box3x3":          ("none", ""),
    "gaussian3x3":     ("none", ""),
    "sobel_x3x3":      ("none", ""),
    "sobel_y3x3":      ("none", ""),
    "laplacian4_3x3":  ("none", ""),
    "laplacian8_3x3":  ("none", ""),
    "sharpen3x3":      ("none", ""),
    "emboss3x3":       ("none", ""),
    "box5x5":          ("none", ""),
    "gaussian5x5":     ("none", ""),
    "sobel_x5x5":      ("none", ""),
    "sobel_y5x5":      ("none", ""),
    "laplacian5x5":    ("none", ""),
    "sharpen5x5":      ("none", ""),
    "emboss5x5":       ("none", ""),
    "box7x7":          ("none", ""),
}

# llm.c blockers — wider coverage than llama2.c includes both forward AND
# backward kernels, plus attention and gelu which surface new blocker classes:
# math.h ext-call bodies (gelu/crossentropy via tanhf/logf), nested
# affine-for+tensor-yield shapes that multi-root debuf can't dominance-resolve
# (layernorm-fwd/bwd), and indirect-index lookup (encoder).
LLMC_BLOCKERS: dict[str, tuple[str, str]] = {
    "encoder-fwd":              ("indirect-index",    "out[b,t,c] = wte[inp[b,t]*C+c] + wpe[t*C+c]; data-dependent index into wte"),
    "encoder-bwd":              ("indirect-index",    "scatter-accumulate by inp[b,t]; raise rejects indirect target index"),
    "layernorm-fwd":            ("debuf-bug",         "raises to 3 linalg.generic ops; BOTH v2 and multi-root debuf hit a dominance bug on the nested affine.for tensor.insert/yield chain"),
    "layernorm-bwd":            ("debuf-bug",         "same dominance failure as layernorm-fwd in both debuf paths"),
    "matmul-fwd-naive":         ("none",              ""),
    "matmul-bwd":               ("matcher-gap",       "raises 2 linalg.generic (dinp + dweight + dbias accumulation); matcher only matches one shape"),
    "attention-fwd":            ("matcher-gap",       "raises 4 linalg.generic (Q·Kᵀ, max-shift, exp+sum, softmax·V); v2 debuf fails on softmax-fused tuple-yield, multi-root succeeds; full attention body not in matcher library"),
    "attention-bwd":            ("matcher-gap",       "raises 1 generic; gradient-through-attention shape not in library"),
    "gelu-fwd":                 ("ext-math-call",     "body calls tanhf — raise can't fold an extern math.h call into a pure-arith linalg.generic body"),
    "gelu-bwd":                 ("ext-math-call",     "body calls tanhf + coshf — same ext-call block"),
    "residual-fwd":             ("matcher-gap",       "single fully-parallel elementwise add; matcher has no axpy/add template that matches this shape"),
    "residual-bwd":             ("matcher-gap",       "two parallel elementwise dinp += dout generics; same axpy gap"),
    "softmax-fwd":              ("matcher-gap",       "per-row softmax with max-shift wrapped in (B, T) outer affine.fors plus an additional masking generic. The base 3-step softmax composition (commit 1235c28) matches llama2's flat softmax but not this nested form. Needs either an outer-loop hoist pass to strip the B/T fors and re-match per row, or a separate 4-step composition that includes the masking step"),
    "crossentropy-fwd":         ("ext-math-call",     "body calls logf with indirect-indexed probs[target[b,t]]; raise can't lift"),
    "crossentropy-softmax-bwd": ("matcher-gap",       "raises 1 linalg.generic — the fused softmax-CE backward formula; shape not in matcher library"),
}


def find_kernel_c(name: str, kset: str = "polybench") -> Path | None:
    """Find <name>.c. Dispatches per kernel-set."""
    if kset == "machsuite":
        info = MACHSUITE_KERNELS.get(name)
        if not info:
            return None
        subdir, _fn = info
        # The kernel .c is the only .c in the subdir that's not local_support
        # or generate (per MachSuite layout convention).
        for p in (MACHSUITE_ROOT / subdir).glob("*.c"):
            if p.name in ("local_support.c", "generate.c"):
                continue
            return p
        return None
    if kset == "npb":
        info = NPB_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = NPB_ROOT / srcname
        return p if p.exists() else None
    if kset == "llama2c":
        info = LLAMA2C_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = LLAMA2C_ROOT / srcname
        return p if p.exists() else None
    if kset == "llama_forward":
        info = LLAMA_FORWARD_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = LLAMA_FORWARD_ROOT / srcname
        return p if p.exists() else None
    if kset == "whisper_ops":
        info = WHISPER_OPS_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = WHISPER_OPS_ROOT / srcname
        return p if p.exists() else None
    if kset == "aten_c":
        info = ATEN_C_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = ATEN_C_ROOT / srcname
        return p if p.exists() else None
    if kset == "stencil_conv2d":
        info = STENCIL_CONV2D_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = STENCIL_CONV2D_ROOT / srcname
        return p if p.exists() else None
    if kset == "llmc":
        info = LLMC_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = LLMC_ROOT / srcname
        return p if p.exists() else None
    if kset == "darknet":
        info = DARKNET_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = DARKNET_ROOT / srcname
        return p if p.exists() else None
    if kset == "extracted_darknet":
        info = EXTRACTED_DARKNET_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = EXTRACTED_DARKNET_ROOT / srcname
        return p if p.exists() else None
    if kset == "fusion_opt":
        info = FUSION_OPT_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = EXTRACTED_DARKNET_ROOT / srcname
        return p if p.exists() else None
    # polybench
    for p in POLYBENCH_TEST_DIR.rglob(f"{name}.c"):
        if "/utilities/" in str(p):
            continue
        if p.name.endswith(".orig.c"):
            continue
        return p
    return None


def discover_kernels(mlir_dir: Path = MLIR_DIR) -> list[str]:
    """Return kernel tags present in `mlir_dir`. A kernel is "present" if
    it has any of <tag>.mlir / <tag>_linalg.mlir / <tag>_debuf.mlir /
    <tag>_debuf_mr.mlir — so kernels that fail one stage still show up
    in the index with a partial set of tabs."""
    tags: set[str] = set()
    for f in mlir_dir.glob("*.mlir"):
        name = f.stem
        for suffix in ("_debuf_mr", "_debuf", "_linalg"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        tags.add(name)
    return sorted(tags)


def build_ce_state(c_src: str, c_kernel_dir: Path, mlir_src: str) -> dict:
    """3-visible-pane CE layout state.

    Visible:
      - C editor (top-left)
      - cgeist_aff compiler reading C editor (bottom-left)
      - Opt Pipeline view bound to polygeist-opt:full (right)

    Hidden (in tab stacks alongside the visible panes):
      - LLVM IR editor with affine MLIR (tab next to C editor)
      - polygeist-opt:full compiler reading MLIR editor (tab next to Opt Pipeline)
    The hidden panes still exist so the Opt Pipeline can bind to popt_full.
    """
    editor_opts = {"compileOnChange": True, "colouriseAsm": True}
    cgeist_compiler_pane = {
        "type": "component",
        "componentName": "compiler",
        "componentState": {
            "id": 1,
            "source": 1,
            "compiler": CGEIST_NAME,
            "lang": "c",
            "editorid": 1,
            "treeid": 0,
            "filters": {},
            "options": f"-I{c_kernel_dir}",
            "libs": [],
        },
    }
    popt_compiler_pane = {
        "type": "component",
        "componentName": "compiler",
        "componentState": {
            "id": 2,
            "source": 2,
            "compiler": POPT_NAME,
            "lang": "llvm",
            "editorid": 2,
            "treeid": 0,
            "filters": {},
            "options": "",
            "libs": [],
        },
    }
    opt_pipeline_pane = {
        "type": "component",
        "componentName": "optPipelineView",
        "componentState": {
            "id": 2,
            "lang": "llvm",
            "compiler": POPT_NAME,
            "compilerName": POPT_DISPLAY,
            "editorid": 2,
            "treeid": 0,
            "selectedGroup": "",
            "selectedIndex": 0,
            "sidebarWidth": 250,
        },
    }
    c_editor = {
        "type": "component",
        "componentName": "codeEditor",
        "componentState": {"id": 1, "source": c_src, "lang": "c", "options": editor_opts},
    }
    mlir_editor = {
        "type": "component",
        "componentName": "codeEditor",
        "componentState": {"id": 2, "source": mlir_src, "lang": "llvm", "options": editor_opts},
    }
    return {
        "version": 4,
        "content": [{
            "type": "row",
            "content": [
                {
                    "type": "column",
                    "width": 50,
                    "content": [
                        # Tab stack: C editor active, LLVM IR editor on a hidden tab.
                        {
                            "type": "stack",
                            "activeItemIndex": 0,
                            "content": [c_editor, mlir_editor],
                        },
                        cgeist_compiler_pane,
                    ],
                },
                # Tab stack: Opt Pipeline active, popt_full compiler on a hidden tab.
                {
                    "type": "stack",
                    "width": 50,
                    "activeItemIndex": 0,
                    "content": [opt_pipeline_pane, popt_compiler_pane],
                },
            ],
        }],
    }


def ce_link_from_paths(c_path: Path | None, mlir_path: Path) -> str | None:
    """Construct a CE deep link from an explicit C/MLIR artifact pair."""
    if not c_path or not mlir_path.exists():
        return None
    c_src = c_path.read_text()
    mlir_src = mlir_path.read_text()
    # Strip the giant dlti spec — saves a lot of URL space and CE will recompute
    # it for the popt_full pane anyway.
    mlir_src = re.sub(
        r'module attributes \{[^\}]*\}',
        'module',
        mlir_src, count=1,
    )
    state = build_ce_state(c_src, c_path.parent, mlir_src)
    payload = json.dumps(state, separators=(',', ':'))
    return CE_BASE + "#" + urllib.parse.quote(payload, safe='')


def ce_link(kernel: str, mlir_dir: Path = MLIR_DIR,
            kset: str = "polybench") -> str | None:
    """Construct the CE deep-link URL for a kernel; None if sources missing."""
    if kset == "whisper_ops" and kernel in WHISPER_OPS_IR_ONLY:
        return None
    return ce_link_from_paths(
        find_kernel_c(kernel, kset=kset),
        mlir_dir / f"{kernel}.mlir",
    )


def render_html(title: str, body_html: str, css: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #ffffff; color: #1f1f1f; }}
  .header {{ background: #f4f4f4; padding: 10px 20px;
             border-bottom: 1px solid #ddd; }}
  .header a {{ color: #0366d6; text-decoration: none; margin-right: 12px; }}
  .header a:hover {{ text-decoration: underline; }}
  h1 {{ font-size: 18px; margin: 0; }}
  h2 {{ font-size: 14px; margin: 16px 20px 4px; color: #444; }}
  .container {{ padding: 8px 20px; }}
  pre {{ margin: 0; font-size: 13px; line-height: 1.4; overflow-x: auto;
         background: #fafafa; padding: 8px; border: 1px solid #eee;
         color: #1f1f1f; font-family: ui-monospace, SFMono-Regular,
         Menlo, Consolas, monospace; }}
  {css}
  table {{ border-collapse: collapse; margin: 16px 20px; }}
  td, th {{ padding: 8px 16px; border-bottom: 1px solid #eee; }}
  th {{ text-align: left; color: #555; font-weight: 600; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.5px; }}
  tr:hover td {{ background: #f8f8f8; }}
  td a.kernel {{ color: #0366d6; text-decoration: none; font-weight: 600;
                 font-size: 15px; }}
  td a.kernel:hover {{ text-decoration: underline; }}
  td a.viewer {{ color: #666; font-size: 12px; }}
  .pass {{ color: #1a7f37; font-weight: 600; }}
  .partial {{ color: #9a6700; font-weight: 600; }}
  .none {{ color: #cf222e; font-weight: 600; }}
  .nope {{ color: #888; }}
  .intro {{ padding: 12px 20px; color: #444; max-width: 900px; }}
  .intro code {{ background: #f1f1f1; padding: 1px 6px; border-radius: 3px;
                 font-size: 13px; }}
</style></head>
<body>{body_html}</body></html>
"""


def syntax_highlight(text: str, lang: str = "llvm") -> tuple[str, str]:
    """Render MLIR as plain text inside a styled <pre>. We deliberately skip
    pygments' LLVM lexer because it doesn't recognise MLIR syntax and marks
    nearly every token with an "error" class — which renders as a red box."""
    text = re.sub(r"#dlti\.dl_spec<[^>]*>", "(dlti spec hidden)", text)
    import html
    return f'<pre class="ir">{html.escape(text)}</pre>', ''


_LOOP_RE = re.compile(r"\b(affine\.for|scf\.for|scf\.while|scf\.parallel|affine\.parallel)\b")


def count_for_loops(text: str) -> int:
    """Count loop-level ops still in the IR. Each match is one loop nest level
    that the raise pipeline did NOT lift to a linalg.generic — a measure of how
    much imperative structure the kernel still carries after the pipeline."""
    return len(_LOOP_RE.findall(text))


def run_rewriter(path: Path) -> tuple[str, list[tuple]]:
    # Keep the static viewer on the production matcher path. Whole-algorithm
    # routes such as histogram, Trisolv, Cholesky, and CSR SpMV live in the
    # structured rewrite stage rather than in single-linalg-op matching.
    command = [PYTHON, str(REWRITER), str(path),
               "--enable-structured-rewrite"]
    try:
        res = subprocess.run(
            command,
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        # One pathological kernel must not prevent unrelated suites (and the
        # already-materialized ATen pages) from being published. Preserve the
        # matcher input as the displayed fallback and report zero launches.
        original = path.read_text()
        return original, [
            ("launches", 0),
            ("residual_lg", len(re.findall(r"\blinalg\.generic\b", original))),
        ]
    if res.returncode != 0:
        raise RuntimeError(
            f"kernel matcher failed for {path} with {PYTHON}:\n{res.stderr}"
        )
    out = res.stdout
    n_launch = len(re.findall(r"kernel\.launch", out))
    n_lg = len(re.findall(r"linalg\.generic", out))
    return out, [("launches", n_launch), ("residual_lg", n_lg)]


def lower_matched_to_abi(matched_text: str) -> tuple[str | None, str | None]:
    """Lower matcher output to the shared polygeist_* pointer ABI.

    The matcher intentionally emits references without copying the complete
    kernel library into every snapshot. Inject the definitions in a temporary
    file, run the production ABI pass, and retain only its dead-stripped
    output. Return a short diagnostic instead of failing the whole site when a
    newly matched symbol does not yet have an ABI lowering.
    """
    if "kernel.launch" not in matched_text:
        return None, None
    if not POLYGEIST_OPT.exists() or not KERNEL_LIBRARY.exists():
        return None, "polygeist-opt or kernel library is unavailable"
    with tempfile.TemporaryDirectory(prefix="polygeist-viewer-abi-") as td:
        work = Path(td)
        matched = work / "matched.mlir"
        with_defns = work / "with_defns.mlir"
        abi = work / "abi.mlir"
        matched.write_text(matched_text)
        injected = subprocess.run(
            [PYTHON, str(INJECT_KERNEL_LIBRARY), str(matched),
             str(KERNEL_LIBRARY), "-o", str(with_defns)],
            capture_output=True, text=True,
        )
        if injected.returncode != 0:
            return None, injected.stderr.strip() or "kernel-library injection failed"
        lowered = subprocess.run(
            [str(POLYGEIST_OPT), "--lower-kernel-launch-to-cublas",
             str(with_defns), "-o", str(abi)],
            capture_output=True, text=True,
        )
        if lowered.returncode != 0:
            detail = lowered.stderr.strip().splitlines()
            error_line = next((line for line in detail if " error:" in line), None)
            return None, error_line or (detail[-1] if detail else "ABI lowering failed")
        return abi.read_text(), None


def runtime_backend_status(abi_text: str | None) -> dict[str, object]:
    """Describe whether emitted ABI calls have CPU/CUDA implementations."""
    if not abi_text:
        return {
            "calls": [], "cpu_supported": False, "cuda_supported": False,
            "cblas_calls": [], "cpu_missing": [], "cuda_missing": [],
        }
    calls = sorted(set(re.findall(r"\bcall\s+@(polygeist_[A-Za-z0-9_]+)", abi_text)))
    # Lifecycle/pipeline hooks are backend plumbing, not selected kernels.
    calls = [c for c in calls if not c.endswith(("_init", "_destroy",
                                                  "_pipeline_begin",
                                                  "_pipeline_end"))]
    cpu_text = CPU_RUNTIME.read_text() if CPU_RUNTIME.exists() else ""
    cuda_text = CUDA_RUNTIME.read_text() if CUDA_RUNTIME.exists() else ""
    cpu_missing = [c for c in calls if f"{c}(" not in cpu_text]
    cuda_missing = [c for c in calls if f"{c}(" not in cuda_text]
    return {
        "calls": calls,
        "cpu_supported": bool(calls) and not cpu_missing,
        "cuda_supported": bool(calls) and not cuda_missing,
        "cblas_calls": [c for c in calls if c in CPU_CBLAS_CALLS],
        "cpu_missing": cpu_missing,
        "cuda_missing": cuda_missing,
    }


_NATIVE_CUDA_CSV = ATEN_C_ROOT / "native_cuda_results" / "torch_aten_silicon.csv"
_MATCH_CANDIDATES_JSON = (
    ATEN_C_ROOT / "native_cuda_results" / "match_candidates.json"
)


def _load_match_candidates():
    """kernel -> {winner, candidates:[...]}: every abi-lowerable library op the
    matcher's enumeration found for a body, not just the greedy winner."""
    if _MATCH_CANDIDATES_JSON.exists():
        try:
            return json.loads(_MATCH_CANDIDATES_JSON.read_text())
        except Exception:
            return {}
    return {}


_MATCH_CANDIDATES = _load_match_candidates()
_RESIDENCY_LEAKS_JSON = (
    ATEN_C_ROOT / "native_cuda_results" / "residency_leaks.json"
)


def _load_residency_leaks():
    """kernel -> {allocs, copies, elidable, genuine, inter_call_*}: buffer
    allocs/copies the lowered code carries (residency leaks in a chain)."""
    if _RESIDENCY_LEAKS_JSON.exists():
        try:
            return json.loads(_RESIDENCY_LEAKS_JSON.read_text())
        except Exception:
            return {}
    return {}


_RESIDENCY_LEAKS = _load_residency_leaks()
_RESIDENT_SILICON_CSV = (
    ATEN_C_ROOT / "native_cuda_results" / "resident_silicon.csv"
)


def _load_resident_silicon():
    """kernel -> {resident_us, warm_us, device_speedup}: genuine device-resident
    timing (operands in cudaMalloc'd DRAM, copies excluded — torch methodology).
    warm_us here is the same-run host-ABI number, so the speedup is exact."""
    out = {}
    if _RESIDENT_SILICON_CSV.exists():
        for r in csv.DictReader(_RESIDENT_SILICON_CSV.open()):
            if r.get("kernel"):
                out[r["kernel"]] = r
    return out


_RESIDENT_SILICON = _load_resident_silicon()
_NATIVE_RESIDENT_CSV = (
    ATEN_C_ROOT / "native_cuda_results" / "native_resident.csv"
)


def _load_native_resident():
    """kernel -> {native_us, shape}: torch native measured at the EXACT resident
    shape (single source of truth), so resident/native is same-shape by design."""
    out = {}
    if _NATIVE_RESIDENT_CSV.exists():
        for r in csv.DictReader(_NATIVE_RESIDENT_CSV.open()):
            if r.get("kernel"):
                out[r["kernel"]] = r
    return out


_NATIVE_RESIDENT = _load_native_resident()
_NATIVE_SUF = [
    "_backward_cpu", "_backward", "_forward_cpu", "_forward", "_out_cpu", "_out",
    "_scalarized", "_transform_cpu", "_transform", "_template_cpu", "_cpu",
    "_impl", "_stub", "_allreduce", "_tensor", "_scalar", "_from_to",
    "_full_64_bits_range", "_legacy", "_nhwc", "_dims", "_serial", "_naive",
    "_generic", "_select", "_columns", "_grad_weight", "_indices", "_launch",
    "_acc", "_zero_acc",
]


def _native_base(k: str) -> str:
    s = k[5:] if k.startswith("aten_") else k
    changed = True
    while changed:
        changed = False
        for suf in _NATIVE_SUF:
            if s.endswith(suf) and len(s) > len(suf) + 2:
                s = s[:-len(suf)]
                changed = True
    return s


def _load_native_cuda():
    us = {}
    if _NATIVE_CUDA_CSV.exists():
        for r in csv.DictReader(_NATIVE_CUDA_CSV.open()):
            k = r.get("kernel", "")
            v = r.get("native_device_us", "")
            if k and v:
                us[k] = v
    base2us = {}
    for k, v in us.items():
        base2us.setdefault(_native_base(k), v)
    return us, base2us


_NATIVE_CUDA_US, _NATIVE_CUDA_BASE = _load_native_cuda()


def native_cuda_for(kernel: str):
    """Return (microseconds, provenance) for a kernel's extracted native CUDA
    number, matching by exact name then by base-op family, else (None, None)."""
    if kernel in _NATIVE_CUDA_US:
        return _NATIVE_CUDA_US[kernel], "measured"
    b = _native_base(kernel)
    if b in _NATIVE_CUDA_BASE:
        return _NATIVE_CUDA_BASE[b], "family"
    # Token-aware fuzzy fallback: the measured op must appear as a whole
    # underscore-delimited token (or a token-boundary prefix/suffix) of the
    # kernel base — NOT an incidental substring. This stops short ops like
    # 'le'/'ge'/'ne'/'eq' from matching inside longer names (e.g. the "le" in
    # "as_complex" wrongly matched torch.le at 225µs).
    b_tokens = b.split("_")
    for mb, v in _NATIVE_CUDA_BASE.items():
        if len(mb) < 3:
            continue
        mb_tokens = mb.split("_")
        if (mb in b_tokens or b in mb_tokens or
                b.startswith(mb + "_") or b.endswith("_" + mb) or
                mb.startswith(b + "_") or mb.endswith("_" + b)):
            return v, "family"
    return None, None


def build_kernel_page(kernel: str, mlir_dir: Path = MLIR_DIR,
                       kset: str = "polybench",
                       file_prefix: str = "") -> dict:
    raised = mlir_dir / f"{kernel}_linalg.mlir"
    debuf = mlir_dir / f"{kernel}_debuf.mlir"
    debuf_mr = mlir_dir / f"{kernel}_debuf_mr.mlir"
    cgeist_mlir = mlir_dir / f"{kernel}.mlir"

    pages: dict[str, str] = {}
    css = ""
    n_for = 0
    n_linalg = 0
    matched_text: str | None = None
    matched_symbols: list[str] = []
    abi_text: str | None = None
    abi_error: str | None = None
    report = [("launches", 0), ("residual_lg", 0)]
    # Whole-algorithm Cholesky/Trisolv recognition currently consumes the
    # memref-Linalg form.  Their tensor/debufferized form remains a distinct
    # pipeline gap, so show the route that was actually built and run.
    prefer_memref_match = (
        kset == "polybench" and kernel in {"cholesky", "trisolv"}
    )

    source = find_kernel_c(kernel, kset=kset)
    if source and source.exists():
        source_html, css = syntax_highlight(source.read_text())
        pages["source"] = source_html

    if cgeist_mlir.exists():
        cgeist_text = cgeist_mlir.read_text()
        rendered, css = syntax_highlight(cgeist_text)
        pages["cgeist"] = rendered
        if not raised.exists() and not debuf.exists() and not debuf_mr.exists():
            n_for = count_for_loops(cgeist_text)
            report = [
                ("launches", 0),
                ("residual_lg", len(re.findall(r"linalg\.generic", cgeist_text))),
            ]
    if raised.exists():
        raised_text = raised.read_text()
        n_linalg = len(re.findall(r"\blinalg\.generic\b", raised_text))
        rendered, css = syntax_highlight(raised_text)
        pages["raised"] = rendered
        if not debuf.exists() and not debuf_mr.exists():
            n_for = count_for_loops(raised_text)
            report = [
                ("launches", 0),
                ("residual_lg", len(re.findall(r"linalg\.generic", raised_text))),
            ]
        if kset == "stencil_conv2d" and not debuf.exists():
            n_for = count_for_loops(raised_text)
            rewritten, report = run_rewriter(raised)
            matched_text = rewritten
            rendered, css = syntax_highlight(rewritten)
            pages["matched"] = rendered
        if prefer_memref_match:
            n_for = count_for_loops(raised_text)
            rewritten, report = run_rewriter(raised)
            matched_text = rewritten
            matched_symbols = sorted(set(
                re.findall(r"kernel\.launch\s+@([A-Za-z0-9_]+)", rewritten)
            ))
            rendered, css = syntax_highlight(rewritten)
            pages["matched"] = rendered
    if debuf.exists():
        debuf_text = debuf.read_text()
        if not prefer_memref_match:
            n_for = count_for_loops(debuf_text)
        rendered, css = syntax_highlight(debuf_text)
        pages["debuf"] = rendered
        # The exhaustive ATen sweep stores the authoritative matcher output
        # beside its diagnostics. Reuse it instead of starting one Egglog
        # process per page (hundreds of avoidable process launches).
        stored_match = mlir_dir / kernel / "matched.mlir"
        if prefer_memref_match:
            rewritten = matched_text or debuf_text
        elif kset == "aten_c" and stored_match.exists():
            rewritten = stored_match.read_text()
            report = [
                ("launches", len(re.findall(r"kernel\.launch\s+@", rewritten))),
                ("residual_lg", len(re.findall(r"\blinalg\.generic\b", rewritten))),
            ]
        else:
            rewritten, report = run_rewriter(debuf)
        # Keep raising coverage tied to the matcher input. A whole-function
        # rewrite may remove every loop, but that must not retroactively claim
        # that RaiseToLinalg raised those loops.
        if not prefer_memref_match:
            matched_text = rewritten
            matched_symbols = sorted(set(
                re.findall(r"kernel\.launch\s+@([A-Za-z0-9_]+)", rewritten)
            ))
            rendered, css = syntax_highlight(rewritten)
            pages["matched"] = rendered
    if debuf_mr.exists():
        debuf_mr_text = debuf_mr.read_text()
        rendered, css = syntax_highlight(debuf_mr_text)
        pages["debuf_mr"] = rendered
        # Fallback: if v2 debuf failed but multi-root succeeded (the
        # common pattern for whole-program-raise suites),
        # run the matcher on the multi-root output so the "matched" tab
        # and the match-status column reflect what's actually achievable.
        if not debuf.exists() and not debuf_mr_text.lstrip().startswith("//"):
            n_for = count_for_loops(debuf_mr_text)
            rewritten, report = run_rewriter(debuf_mr)
            matched_text = rewritten
            matched_symbols = sorted(set(
                re.findall(r"kernel\.launch\s+@([A-Za-z0-9_]+)", rewritten)
            ))
            rendered, css = syntax_highlight(rewritten)
            pages["matched"] = rendered

    # Materialize ABI lowering from the matcher text shown on this page so a
    # stale stored artifact cannot describe a launch that is no longer
    # selected.  Fall back to a stored ABI only when no matched text exists.
    stored_abi_candidates = [
        mlir_dir / kernel / "abi.mlir",
        mlir_dir / f"{kernel}_abi.mlir",
    ]
    stored_abi = next((p for p in stored_abi_candidates if p.exists()), None)
    if matched_text is not None:
        abi_text, abi_error = lower_matched_to_abi(matched_text)
    elif stored_abi:
        abi_text = stored_abi.read_text()
    if abi_text:
        abi_html, css = syntax_highlight(abi_text)
        pages["abi"] = abi_html

    ce_url = ce_link(kernel, mlir_dir=mlir_dir, kset=kset)
    open_link = (f'<a href="{ce_url}" target="_blank" '
                 f'style="margin-left:12px; color:#0366d6;">'
                 f'open in Compiler Explorer →</a>') if ce_url else ''

    matched_n_for = (
        count_for_loops(matched_text) if matched_text is not None else n_for
    )
    n_launches = report[0][1]
    n_resid = report[1][1]
    stage_labels = [
        ("source", "C source"),
        ("cgeist", "cgeist"),
        ("raised", "raised"),
        ("debuf", "debufferized"),
        ("debuf_mr", "debuf multi-root"),
        ("matched", "kernel.launch"),
        ("abi", "shared ABI"),
    ]
    jump_links = " · ".join(
        f'<a href="#{stage}">{label}</a>'
        for stage, label in stage_labels if stage in pages
    )
    summary = (
        f'<div class="summary" style="padding:8px 20px; '
        f'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
        f'<b>{n_launches}</b> kernel.launch op(s) emitted &nbsp;·&nbsp; '
        f'<b>{n_resid}</b> residual linalg.generic &nbsp;·&nbsp; '
        f'<b>{n_for}</b> residual loop(s) before matching &nbsp;·&nbsp; '
        f'<b>{matched_n_for}</b> after matching &nbsp;|&nbsp; '
        f'jump to: {jump_links}'
        f'</div>'
    )
    backend = runtime_backend_status(abi_text)
    if n_launches == 0:
        backend_panel = (
            '<div class="summary" style="padding:10px 20px; '
            'border-bottom:1px solid #eee; background:#fff8e6; font-size:13px;">'
            '<b>Backend branch:</b> no library launch was selected. Residual '
            'Linalg follows the generic CPU loop lowering path.</div>'
        )
    elif abi_error:
        backend_panel = (
            '<div class="summary" style="padding:10px 20px; '
            'border-bottom:1px solid #eee; background:#fff0f0; font-size:13px;">'
            '<b>Shared ABI lowering: unavailable.</b> '
            f'{html.escape(abi_error)}</div>'
        )
    else:
        calls = backend["calls"]
        cblas = backend["cblas_calls"]
        call_names = ", ".join(f'<code>@{html.escape(c)}</code>' for c in calls)
        cpu_state = "available" if backend["cpu_supported"] else "incomplete"
        gpu_state = "available" if backend["cuda_supported"] else "incomplete"
        cblas_note = (
            f'{len(cblas)}/{len(calls)} selected call(s) have an optimized CBLAS route; '
            'the other implemented CPU calls use reference C.'
            if calls else "No runtime calls were emitted."
        )
        backend_panel = (
            '<div class="backend-flow">'
            '<div class="backend-common"><b>Shared target-neutral ABI</b><br>'
            f'{call_names or "—"}</div>'
            '<div class="backend-arrow">↙</div>'
            f'<div class="backend-card"><b>CPU/C backend: {cpu_state}</b><br>'
            '<code>polygeist_cublas_rt_cpu.c</code><br>'
            f'<span>{html.escape(cblas_note)}</span></div>'
            '<div class="backend-arrow">↘</div>'
            f'<div class="backend-card"><b>GPU backend: {gpu_state}</b><br>'
            '<code>polygeist_cublas_rt_cuda.c</code><br>'
            '<span>Vendor CUDA library call plus transfer/residency handling.</span></div>'
            '</div>'
        )
    if kset == "aten_c":
        _nus, _nhow = native_cuda_for(kernel)
        if _nus:
            try:
                _nms = f"{float(_nus) / 1000.0:.4f} ms"
            except ValueError:
                _nms = f"{_nus} us"
            summary += (
                f'<div class="summary" style="padding:8px 20px; '
                f'border-bottom:1px solid #eee; background:#f3faf3; '
                f'font-size:13px;">'
                f'<b>REAL ATen native kernel (torch.&lt;op&gt; on CUDA, Jetson '
                f'Orin sm87):</b> <b>{_nms}</b> '
                f'<span style="color:#666;">({_nhow}; real torch 2.6.0+cu126 '
                f'dispatch, torch.cuda.Event-timed warm best)</span></div>'
            )
    back_href, back_label = "index.html", "index"
    if kset == "polybench":
        back_href, back_label = "polybench.html", "PolyBench"
    elif kset == "aten_c":
        back_href, back_label = "numerical.html", "ATen"
    header = (
        f'<div class="header"><h1><a href="{back_href}">← {back_label}</a> '
        f'&nbsp; {kernel}{open_link}</h1></div>'
        + summary + backend_panel
    )
    body_blocks = []
    for stage, title in [
        ("source",    "original C source"),
        ("cgeist",   "cgeist output (pre-raise MLIR)"),
        ("raised",   "raised (memref linalg, before debuferize)"),
        ("debuf",    "debuferized (tensor linalg, matcher input)"),
        ("debuf_mr", "debuferized — multi-root (--linalg-debufferize=use-multi-root=true)"),
        ("matched",  "kernel.launch (matcher output)"),
        ("abi",      "ABI-lowered IR (func.call to runtime shim)"),
    ]:
        if stage not in pages:
            continue
        body_blocks.append(
            f'<h2 id="{stage}">{title}</h2>'
            f'<div class="container">{pages[stage]}</div>'
        )
    body = header + "\n".join(body_blocks)
    css += (
        ".backend-flow{display:grid;grid-template-columns:minmax(260px,1fr) 30px "
        "minmax(260px,1fr) 30px minmax(260px,1fr);gap:8px;align-items:center;"
        "padding:12px 20px;background:#f4f7fb;border-bottom:1px solid #d8dee8;}"
        ".backend-common,.backend-card{padding:10px 12px;border:1px solid #c8d2e2;"
        "border-radius:6px;background:white;font-size:12px;line-height:1.5;}"
        ".backend-common{background:#eef4ff}.backend-arrow{text-align:center;"
        "font-size:22px;color:#65758b}.backend-card span{color:#555}"
        "@media(max-width:900px){.backend-flow{display:block}.backend-arrow{display:none}"
        ".backend-common,.backend-card{margin:7px 0}}"
    )
    OUTPUT_DIR.joinpath(f"{file_prefix}{kernel}.html").write_text(render_html(kernel, body, css))
    return {
        "launches": report[0][1],
        "linalg_ops": n_linalg,
        "matched_symbols": matched_symbols,
        "abi_lowered": abi_text is not None,
        "abi_error": abi_error,
        "abi_calls": backend["calls"],
        "cpu_backend": backend["cpu_supported"],
        "cuda_backend": backend["cuda_supported"],
        "cpu_cblas_calls": backend["cblas_calls"],
        "residual": report[1][1],
        "residual_for": n_for,
        "matched_residual_for": matched_n_for,
        "ce_url": ce_url,
        "ce_suppressed": kset == "whisper_ops" and kernel in WHISPER_OPS_IR_ONLY,
        "page_filename": f"{file_prefix}{kernel}.html",
    }


def _aten_upstream_info(kernel: str) -> dict[str, object]:
    """Resolve the pinned ATen implementation-family URL and local pointer."""
    upstream_file, token = ATEN_C_PROVENANCE[kernel]
    source = ATEN_UPSTREAM_ROOT / upstream_file
    upstream_line = None
    if token and source.exists():
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            if token in line:
                upstream_line = line_no
                break
    fragment = f"#L{upstream_line}" if upstream_line else ""
    upstream_url = (
        "https://github.com/pytorch/pytorch/blob/"
        f"{ATEN_UPSTREAM_COMMIT}/{upstream_file}{fragment}"
    )
    local_pointer = f"third_party/pytorch/{upstream_file}"
    if upstream_line:
        local_pointer += f":{upstream_line}"
    return {
        "upstream_file": upstream_file,
        "upstream_line": upstream_line,
        "upstream_url": upstream_url,
        "upstream_pointer": local_pointer,
    }


def build_aten_c_source_pages(aten_stats: dict[str, dict]) -> None:
    """Render a C-only page with pinned PyTorch provenance for every fixture."""
    for kernel in ATEN_C_ORDER:
        source = find_kernel_c(kernel, kset="aten_c")
        if source is None or not source.exists():
            continue
        provenance = _aten_upstream_info(kernel)
        c_page_filename = f"aten_c_{kernel}.html"
        highlighted, css = syntax_highlight(source.read_text(), "c")
        lowering_page = aten_stats.get(kernel, {}).get("page_filename", "")
        lowering_link = (
            f'<a href="{html.escape(lowering_page)}">view lowering IR →</a>'
            if lowering_page else "lowering IR unavailable"
        )
        header = (
            '<div class="header"><h1><a href="numerical.html">← ATen</a> '
            f'&nbsp; standalone C: {html.escape(kernel)}</h1></div>'
        )
        provenance_html = (
            '<div class="summary" style="padding:10px 20px; '
            'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
            f'<b>Standalone fixture:</b> <code>{html.escape(str(source.relative_to(REPO_ROOT)))}</code><br>'
            f'<b>Original ATen implementation family:</b> '
            f'<a href="{html.escape(str(provenance["upstream_url"]))}" target="_blank">'
            f'<code>{html.escape(str(provenance["upstream_pointer"]))}</code></a><br>'
            f'<b>PyTorch commit:</b> <code>{ATEN_UPSTREAM_COMMIT}</code><br>'
            '<b>Extraction:</b> numerical algorithm isolated into fixed-shape C; '
            'ATen Tensor/dispatch/template machinery removed.<br>'
            f'{lowering_link}'
            '</div>'
        )
        OUTPUT_DIR.joinpath(c_page_filename).write_text(
            render_html(
                f"ATen standalone C: {kernel}",
                header + provenance_html
                + '<h2>standalone C form lowered by cgeist</h2>'
                + f'<div class="container">{highlighted}</div>',
                css,
            )
        )
        aten_stats.setdefault(kernel, {}).update(
            {
                "c_page_filename": c_page_filename,
                **provenance,
            }
        )


ATEN_PAGE_SIZE = 20

# These measurements were collected from whole-function Python recognizers
# removed on 2026-08-13. Keep the raw CSV as historical evidence, but do not
# attach those executions to the current general matcher output.
ATEN_RETIRED_EARLY_MATCH_KERNELS = {
    "aten_adaptive_avg_pool2d_backward_cpu", "aten_adaptive_avg_pool2d_cpu",
    "aten_adaptive_avg_pool3d_backward_cpu",
    "aten_adaptive_avg_pool3d_cpu", "aten_adaptive_max_pool1d_cpu",
    "aten_adaptive_max_pool2d_backward_cpu", "aten_adaptive_max_pool2d_cpu",
    "aten_adaptive_max_pool3d_backward_cpu", "aten_adaptive_max_pool3d_cpu",
    "aten_adaptive_max_pool3d_legacy_backward_cpu",
    "aten_adaptive_max_pool3d_legacy_cpu",
    "aten_avg_pool2d_cpu",
    "aten_avg_pool3d_cpu",
    "aten_batch_norm_backward_cpu",
    "aten_batch_norm_backward_template_cpu",
    "aten_fp16_gemv_trans_cpu", "aten_joint_scaling_cpu",
    "aten_linalg_powsum_cpu",
    "aten_nansum_cpu", "aten_nested_all_cpu",
    "aten_nested_sum_dim_cpu", "aten_or_reduce_cpu", "aten_powsum_cpu",
    "aten_sinc", "aten_sort_cpu",
    "aten_sparse_norm_cpu", "aten_topk_cpu",
    "aten_upsample_lanczos2d_aa_backward_cpu",
    "aten_upsample_lanczos2d_aa_cpu", "aten_xor_sum_cpu",
}


def _aten_page_filename(sort_by: str, page: int) -> str:
    prefix = {"alphabetical": "numerical",
              "raised": "numerical-raised",
              "resident": "numerical-resident"}.get(sort_by,
                                                    "numerical-correctness")
    return f"{prefix}.html" if page == 1 else f"{prefix}-{page}.html"


def _aten_performance_by_kernel() -> dict[str, dict[str, str]]:
    """Return only measurements that describe the current matcher output.

    Keep this filtering shared by sorting and rendering.  Otherwise a removed
    Thrust route can rank as PASS while its row is rendered as unmeasured.
    """
    performance = {}
    for row in _read_csv(ATEN_SILICON_RESULTS):
        if row.get("kernel", "") in ATEN_RETIRED_EARLY_MATCH_KERNELS:
            continue
        if "thrust" in " ".join(str(value) for value in row.values()).lower():
            continue
        performance[row.get("kernel", "")] = row
    return performance


_RAISED_MATCHED_CACHE: set[str] | None = None


def _raised_matched_kernels() -> set[str]:
    """Kernels whose matcher output contains a library launch (fully raised +
    matched to a library op).  Scans results/*/matched.mlir once."""
    global _RAISED_MATCHED_CACHE
    if _RAISED_MATCHED_CACHE is None:
        found: set[str] = set()
        for mm in (ATEN_C_ROOT / "results").glob("*/matched.mlir"):
            try:
                if "kernel.launch @" in mm.read_text():
                    found.add(mm.parent.name)
            except OSError:
                pass
        _RAISED_MATCHED_CACHE = found
    return _RAISED_MATCHED_CACHE


def _aten_sorted_kernels(sort_by: str) -> list[str]:
    if sort_by == "alphabetical":
        return sorted(ATEN_C_ORDER)
    if sort_by == "raised":
        # Every kernel that raised + matched to a library op.  Ones we already
        # measured on silicon (resident) float to the top, then the rest
        # alphabetically — so the browsable set shows measured work first.
        matched = _raised_matched_kernels()
        return sorted(
            (k for k in ATEN_C_ORDER if k in matched),
            key=lambda k: (0 if _RESIDENT_SILICON.get(k, {}).get("resident_us")
                           else 1, k),
        )
    if sort_by == "resident":
        # Only kernels with resident + native measured at the SAME shape (so
        # resident/native is valid), best ratio first.
        scored = []
        for kernel in ATEN_C_ORDER:
            rs = _RESIDENT_SILICON.get(kernel)
            nr = _NATIVE_RESIDENT.get(kernel)
            if not (rs and rs.get("resident_us") and nr and nr.get("native_us")):
                continue
            if not rs.get("shape") or (sorted(rs["shape"].split("_"))
                                       != sorted(nr.get("shape", "").split("_"))):
                continue
            try:
                scored.append(
                    (float(rs["resident_us"]) / float(nr["native_us"]), kernel))
            except (ValueError, ZeroDivisionError):
                pass
        return [k for _, k in sorted(scored)]
    performance = _aten_performance_by_kernel()
    correctness_rank = {"PASS": 0, "FAIL": 1, "—": 2, "": 2}
    return sorted(
        ATEN_C_ORDER,
        key=lambda kernel: (
            correctness_rank.get(
                performance.get(kernel, {}).get("correctness", "—"), 2
            ),
            kernel,
        ),
    )


def _aten_slowness_diagnosis(kernel: str, baseline: str, ratio: float) -> tuple[str, str, str]:
    """Classify measured ATen gaps by the dominant steady-state cause.

    These labels deliberately describe the current deployment ABI, not the
    semantic matcher.  cudaHostRegister is cached by the runtime, so repeated
    registration is not listed as a warm-run cause.
    """
    if kernel in ("aten_gelu", "aten_gelu_cpu_tanh"):
        return (
            "unfused multi-library decomposition",
            "The raised implementation evaluates GELU as several cuDNN tensor "
            "operations plus cuBLAS scaling, while native uses one fused CUDA "
            "kernel. Device residency removes transfers but cannot remove those "
            "launches, temporaries, and descriptor operations.",
            "Prefer an existing fused GELU frontend/library operation when "
            "available; otherwise this is not a profitable library decomposition.",
        )
    if kernel == "aten_linear_combination_cpu":
        return (
            "low-K library decomposition",
            "Four pointwise terms were represented as a very skinny GEMV. The "
            "library setup and reduction organization cost much more than a fused "
            "elementwise CUDA implementation, even with resident buffers.",
            "Select a fused existing pointwise primitive only when one is available; "
            "otherwise retain this as a semantic match rather than a speed route.",
        )
    if kernel == "aten_rms_norm":
        return (
            "internal RMSNorm staging",
            "The cuDNN backend plan still copies resident inputs into cached internal "
            "buffers and copies its result out. Native uses a three-kernel fused "
            "reduction/scale sequence directly on the public buffers.",
            "Bind the public device pointers directly into the cached cuDNN variant "
            "pack instead of staging through plan-owned allocations.",
        )
    if "gemv" in kernel or kernel == "aten_mv":
        if ratio < 3.0:
            return (
                "direct-buffer GEMV (fixed)",
                "The corrected lowering forwards the original contiguous ABI "
                "buffers to cuBLAS. No matrix/vector tensor materialization "
                "remains; this row is now close to resident cuBLAS.",
                "Fuse the preceding zero into GEMV beta=0 and reduce pipeline "
                "scope/synchronization overhead.",
            )
    if ratio <= 1.25:
        return (
            "near-native library route",
            "The corrected lowering passes cudaMalloc-backed buffers directly. "
            "The remaining difference is ordinary wrapper, descriptor, or launch "
            "overhead rather than tensor materialization.",
            "Cache any remaining descriptors and synchronize at graph boundaries.",
        )
    if kernel == "aten_zeros_cpu":
        return (
            "near-native library route",
            "Device pointers now select cudaMemsetAsync directly; only wrapper and "
            "synchronization overhead remains.",
            "Amortize synchronization in a larger resident graph.",
        )
    if "Memcpy" in baseline or kernel in ("aten_as_complex_cpu",):
        detail = (
            "This path copies through mapped host allocations rather than "
            "between CUDA-resident allocations."
        )
        if kernel == "aten_as_complex_cpu":
            detail += (
                " It also expresses the interleaved split as millions of "
                "four-byte cudaMemcpy2D rows, an intrinsically poor copy shape."
            )
        return (
            "copy residency / geometry",
            detail,
            "Preserve device residency; represent interleaved or strided cases "
            "with a coalesced transform kernel instead of tiny pitched rows.",
        )
    if kernel in ("aten_nested_matmul_broadcast_cpu", "aten_flatten_nd_linear_cpu"):
        return (
            "small batched GEMM + mapped operands",
            "The 256x256 batched products do not amortize the mapped-host "
            "operand and wrapper costs as well as a large dense GEMM.",
            "Use persistent device operands and cache the batched execution plan.",
        )
    if kernel == "aten_outer":
        return (
            "low-intensity GEMM shape",
            "The library call is a GEMM with K=1. That is an outer product with "
            "little reuse, so memory placement dominates despite the GEMM name.",
            "Keep operands/output resident; consider the library's rank-1 update "
            "route when its layout is profitable.",
        )
    if "Convolution" in baseline or "Conv3D" in baseline:
        return (
            "per-call cuDNN setup + mapped operands",
            "The raised wrapper recreates cuDNN descriptors and workspace per "
            "call and uses mapped host operands. Compute-heavy convolutions "
            "amortize this better; smaller or 2D cases expose it.",
            "Cache descriptors, algorithm choice, and workspace, and retain "
            "tensors on the device.",
        )
    if kernel in ("aten_mm", "aten_addmm"):
        return (
            "dense library-call wrapper overhead",
            "Dense GEMM provides useful reuse, but mapped inputs/output and "
            "pipeline synchronization still make the raised end-to-end call "
            "slower than an already-resident cuBLAS operation.",
            "Adopt the device-pointer ABI and synchronize only at graph boundaries.",
        )
    if kernel in ("aten_dot", "aten_blas_dot_naive_cpu", "aten_bf16_dot_cpu", "aten_fp16_dot_cpu"):
        return (
            "mostly amortized reduction",
            "The long reduction and scalar result amortize most wrapper cost; "
            "only a small mapped-memory gap remains.",
            "Device residency should remove most of the remaining difference.",
        )
    if kernel == "aten_softmax":
        return (
            "reduction setup / synchronization",
            "cuDNN does substantial reduction work, so the gap is modest, but "
            "the raised end-to-end call still includes mapped operands, descriptor "
            "setup, and synchronization.",
            "Cache descriptors and keep the tensor resident in a larger GPU graph.",
        )
    return (
        "bandwidth-bound elementwise/reduction",
        "The useful arithmetic per byte is low. Mapped host operands, output "
        "materialization, wrapper setup, and a call-boundary synchronization "
        "dominate the resident fused CUDA/cuDNN operation.",
        "Keep tensors and intermediates resident, fuse adjacent stages, and "
        "synchronize only at the graph boundary.",
    )


def _aten_slowness_page(aten_stats: dict[str, dict]) -> str:
    measurements = []
    for perf in _read_csv(ATEN_DEVICE_RESIDENCY_RESULTS):
        if perf.get("correctness") != "PASS":
            continue
        try:
            ratio = float(perf.get("device_over_resident", ""))
            mapped = float(perf.get("mapped_raised_us", ""))
            device = float(perf.get("device_resident_us", ""))
            resident = float(perf.get("resident_cuda_us", ""))
        except ValueError:
            continue
        measurements.append((ratio, mapped, device, resident, perf))
    measurements.sort(reverse=True, key=lambda item: item[0])

    category_counts: dict[str, int] = {}
    category_styles = {
        "direct-buffer GEMV (fixed)": "cause-amortized",
        "near-native library route": "cause-amortized",
        "unfused multi-library decomposition": "cause-host",
        "low-K library decomposition": "cause-intensity",
        "internal RMSNorm staging": "cause-memory",
        "copy residency / geometry": "cause-copy",
        "small batched GEMM + mapped operands": "cause-intensity",
        "low-intensity GEMM shape": "cause-intensity",
        "per-call cuDNN setup + mapped operands": "cause-setup",
        "dense library-call wrapper overhead": "cause-setup",
        "mostly amortized reduction": "cause-amortized",
        "reduction setup / synchronization": "cause-setup",
        "bandwidth-bound elementwise/reduction": "cause-bandwidth",
    }
    rows = []
    for ratio, mapped, device, resident, perf in measurements:
        kernel = perf.get("kernel", "")
        category, reason, remedy = _aten_slowness_diagnosis(
            kernel, perf.get("baseline", ""), ratio
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        kernel_page = aten_stats.get(kernel, {}).get("page_filename", "")
        kernel_html = (
            f'<a class="kernel" href="{html.escape(kernel_page)}">'
            f'{html.escape(kernel)}</a>' if kernel_page else html.escape(kernel)
        )
        severity = "none" if ratio >= 20 else ("partial" if ratio >= 2 else "pass")
        category_style = category_styles.get(category, "cause-setup")
        rows.append(
            f'<tr><td>{kernel_html}</td>'
            f'<td><code>{html.escape(perf.get("problem", "—"))}</code></td>'
            f'<td>{mapped:,.3f}</td><td>{device:,.3f}</td><td>{resident:,.3f}</td>'
            f'<td class="{severity}"><b>{ratio:,.2f}&times;</b></td>'
            f'<td>{mapped / device:,.2f}&times;</td>'
            f'<td><span class="cause-tag {category_style}">{html.escape(category)}</span>'
            f'<br>{html.escape(reason)}</td>'
            f'<td>{html.escape(remedy)}</td></tr>'
        )

    category_items = "".join(
        f'<li><b>{html.escape(category)}</b>: {count} measured kernel(s)</li>'
        for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    gemvs = [
        item for item in measurements
        if "gemv" in item[4].get("kernel", "").lower()
        or item[4].get("kernel", "") == "aten_mv"
    ]
    gemv_ratio_min = min((item[0] for item in gemvs), default=0.0)
    gemv_ratio_max = max((item[0] for item in gemvs), default=0.0)
    gemv_residency = next(
        (row for row in _read_csv(ATEN_DEVICE_RESIDENCY_RESULTS)
         if row.get("kernel") == "aten_blas_gemv_generic_cpu"), {})
    try:
        mapped_gemv = float(gemv_residency.get("mapped_raised_us", ""))
        device_gemv = float(gemv_residency.get("device_resident_us", ""))
        resident_gemv = float(gemv_residency.get("resident_cuda_us", ""))
        mapped_ratio = float(gemv_residency.get("mapped_over_resident", ""))
        device_ratio = float(gemv_residency.get("device_over_resident", ""))
        gemv_experiment = (
            '<table><thead><tr><th>same raised GEMV path</th><th>warm µs</th>'
            '<th>vs resident</th></tr></thead><tbody>'
            f'<tr><td>mapped-host operands</td><td>{mapped_gemv:,.3f}</td>'
            f'<td>{mapped_ratio:.3f}&times;</td></tr>'
            f'<tr><td>cudaMalloc/device-resident operands</td><td>{device_gemv:,.3f}</td>'
            f'<td>{device_ratio:.3f}&times;</td></tr>'
            f'<tr><td>native resident cuBLAS</td><td>{resident_gemv:,.3f}</td>'
            '<td>1.000&times;</td></tr></tbody></table>'
        )
    except (TypeError, ValueError):
        gemv_experiment = ''
    return (
        '<div class="section-header"><h2 class="section-title">Why are some raised kernels slow?</h2></div>'
        '<div class="intro">'
        '<b>Result: all 40 executable FULL-match ATen kernels pass with true '
        '<code>cudaMalloc</code> operands.</b> This table separates the old mapped-host '
        'ABI from the corrected device-resident lowering and the native resident CUDA '
        'baseline. The median device/native ratio is 1.07&times;; 32/40 are within '
        '1.25&times; and 36/40 are within 2&times;. The four remaining gaps are algorithmic '
        'or internal-library staging (two GELUs, a four-term linear combination, and '
        'RMSNorm), not hidden tensor copies. Red ratios are at least 20&times;, yellow are '
        '2–20&times;, and green are below 2&times;.'
        f'<ul>{category_items}</ul></div>'
        '<div class="section-header" id="gemv-deep-dive"><h2 class="section-title">'
        'GEMV: the old 173× gap is fixed across the family</h2></div>'
        '<div class="intro">'
        f'The rerun GEMV family spans only '
        f'<b>{gemv_ratio_min:.2f}&times;–{gemv_ratio_max:.2f}&times;</b> versus resident '
        'cuBLAS with true device buffers. '
        'For the main f32 case, <code>A</code> is 4096&times;8192 = 33,554,432 floats, '
        'or 128 MiB. Each output uses one matrix row, but the matrix has essentially '
        'no reuse across the call: GEMV performs about two floating-point operations '
        'for every four matrix bytes. It is therefore a memory-bandwidth test wearing '
        'a linear-algebra name.<br><br>'
        'Inspection of the old generated LLVM showed the real dominant cost: before '
        '<code>cublasSgemv</code>, one-shot bufferization allocated and copied the full '
        '128 MiB matrix, then copied vectors/output around the call. A cudaMalloc-pointer '
        'experiment initially crashed because those copies executed on the CPU.<br><br>'
        '<b>Fix:</b> all library lowerings now trace tensor slices back to their original '
        'memrefs, derive direct pointers, and treat destinations as in-place results. The '
        'AArch64 objects have no allocation or CPU copy around the calls. All corrected runs '
        'pass the CPU reference; '
        'uploads happen before timing and the result download happens afterward.'
        '</div>'
        + gemv_experiment +
        '<div class="section-header" id="polybench-gemv-comparison"><h2 class="section-title">'
        'Why PolyBench GESUMMV shows a raised win</h2></div>'
        '<div class="intro">'
        '<b>The native baselines are not equivalent.</b> The warmed PolyBench result '
        'compares two optimized cuBLAS GEMV calls from the raised path against the '
        'handwritten PolyBenchGPU <code>gesummv_kernel</code>. That CUDA kernel assigns '
        'one thread to each output row and executes the complete inner <code>j</code> '
        'dot-product loop serially inside that thread. At N=512 it launches only 512 '
        'threads and does not use a parallel reduction, so cuBLAS can beat it even while '
        'using the mapped-host ABI.<br><br>'
        'The ATen resident baseline is already optimized cuBLAS using '
        '<code>cudaMalloc</code> operands. It therefore removes the algorithm-quality '
        'advantage and exposes the raised ABI penalty directly. The sizes also cross a '
        'different memory regime: one PolyBench N=512 f64 matrix is only 2 MiB (4 MiB '
        'for A+B, repeatedly reused), whereas the ATen 4096x8192 f32 matrix is 128 MiB '
        'and must stream from DRAM. Finally, the PolyBench raised number sums device-event '
        'time inside runtime shims; the ATen raised number is wall time for the complete '
        'raised call. Thus the PolyBench win means <i>cuBLAS beats that naive CUDA '
        'implementation</i>; it does not show that mapped-host GEMV beats resident cuBLAS.'
        '</div>'
        '<table><thead><tr><th>kernel</th><th>large problem</th>'
        '<th>mapped raised (µs)</th><th>device raised (µs)</th>'
        '<th>native CUDA (µs)</th><th>device/native</th><th>mapped/device</th>'
        '<th>dominant reason</th><th>next correction</th></tr></thead><tbody>'
        + "\n".join(rows) + '</tbody></table>'
    )


def _aten_section(aten_stats: dict[str, dict], kernels: list[str],
                  sort_by: str, page: int, page_count: int) -> str:
    performance = _aten_performance_by_kernel()
    cuda_audit = {
        row.get("kernel", ""): row for row in _read_csv(ATEN_CUDA_LIBRARY_AUDIT)
    }
    rows = []
    for kernel in kernels:
        stats = aten_stats.get(kernel, {})
        kernel_page = stats.get("page_filename", "")
        if kernel_page:
            name = f'<a class="kernel" href="{kernel_page}">{kernel}</a>'
        else:
            name = f'<span>{kernel}</span>'
        c_page = stats.get("c_page_filename", "")
        extracted_c = (
            f'<a class="viewer" href="{html.escape(c_page)}">'
            f'<code>{html.escape(ATEN_C_KERNELS[kernel][0])}</code></a>'
            if c_page else "—"
        )
        upstream_url = stats.get("upstream_url", "")
        upstream_pointer = stats.get("upstream_pointer", "")
        upstream = (
            f'<a class="viewer" href="{html.escape(str(upstream_url))}" '
            f'target="_blank"><code>{html.escape(str(upstream_pointer))}</code></a>'
            if upstream_url else "—"
        )
        symbols = stats.get("matched_symbols", [])
        symbol_html = ", ".join(f"<code>@{s}</code>" for s in symbols) or "—"
        # All abi-lowerable library ops the matcher's enumeration found for this
        # body — surfaces alternatives the greedy "first match wins" hides.
        _mc = _MATCH_CANDIDATES.get(kernel, {})
        _cands = _mc.get("candidates", [])
        if _cands:
            _win = _mc.get("winner")
            _parts = []
            for c in _cands:
                _nm = c.get("name", "") if isinstance(c, dict) else c
                _abi = c.get("abi", True) if isinstance(c, dict) else True
                if _nm == _win:
                    _parts.append(f"<code><b>{html.escape(_nm)}</b></code>")
                elif _abi:
                    _parts.append(f'<code style="color:#137333">'
                                  f'{html.escape(_nm)}</code>')
                else:
                    _parts.append(f'<code style="color:#999" title="semantic '
                                  f'match, no library backend">'
                                  f'{html.escape(_nm)}</code>')
            candidates_cell = ", ".join(_parts)
            _nabi = sum(1 for c in _cands
                        if (c.get("abi", True) if isinstance(c, dict) else True))
            if len(_cands) > 1:
                candidates_cell = (
                    f'<span title="{len(_cands)} candidates ({_nabi} '
                    f'abi-lowerable); greedy match takes the first">'
                    f'{candidates_cell}</span>'
                )
        else:
            candidates_cell = "—"
        # Residency leaks: buffer allocs/copies the lowered chain carries.
        # green = clean/elidable, red = genuine copies or inter-call leaks.
        _rl = _RESIDENCY_LEAKS.get(kernel)
        if _rl:
            _a, _c = _rl.get("allocs", 0), _rl.get("copies", 0)
            _g = _rl.get("genuine", 0)
            _inter = _rl.get("inter_call_copies", 0) + _rl.get(
                "inter_call_allocs", 0)
            if _a == 0 and _c == 0:
                residency_cell = '<span style="color:#137333">clean</span>'
            else:
                _col = "#b00020" if (_g > 0 or _inter > 0) else "#8a6d00"
                _lbl = f"{_a}a {_c}c"
                _ttl = (f"{_a} allocs, {_c} copies ({_rl.get('elidable',0)} "
                        f"elidable, {_g} genuine); {_inter} inter-call leak(s)")
                residency_cell = (f'<span style="color:{_col}" '
                                  f'title="{_ttl}">{_lbl}</span>')
        else:
            residency_cell = "—"
        launches = stats.get("launches", 0)
        residual = stats.get("residual", 0)
        loops = stats.get("residual_for", 0)
        matched_loops = stats.get("matched_residual_for", loops)
        linalg_ops = stats.get("linalg_ops", 0)
        if linalg_ops > 0 and loops == 0:
            raise_status_class, raise_status = "pass", "FULL"
        elif linalg_ops > 0:
            raise_status_class, raise_status = "partial", "PARTIAL"
        else:
            raise_status_class, raise_status = "none", "NONE"
        if kernel in ATEN_C_UNSAFE_MATCHES:
            status_class, status = "none", "UNSAFE"
        elif launches and residual == 0 and matched_loops == 0:
            status_class, status = "pass", "FULL"
        elif launches:
            status_class, status = "partial", "PARTIAL"
        else:
            status_class, status = "none", "NONE"
        assessment = ATEN_C_MATCH_ASSESSMENT.get(kernel, "")
        if "thrust" in assessment.lower():
            assessment = ""
        perf = performance.get(kernel, {})
        execution = html.escape(perf.get("executable_status", "—"))
        correctness = html.escape(perf.get("correctness", "—"))
        problem = html.escape(perf.get("problem", "—"))
        raised_us = html.escape(perf.get("raised_us", "—"))
        resident_us = html.escape(perf.get("resident_cuda_us", "—"))
        baseline = html.escape(perf.get("baseline", "—"))
        # Resident (raised) + native are BOTH measured at the resident shape
        # (single source of truth), so the ratio is same-shape by construction.
        # We gate on the shape strings being equal so a mismatch can never
        # print a bogus ratio again.
        _rs = _RESIDENT_SILICON.get(kernel)
        _nr = _NATIVE_RESIDENT.get(kernel)
        _res_shape = (_rs or {}).get("shape", "")
        _nat_shape = (_nr or {}).get("shape", "")
        # order-independent compare (dim key order can differ across processes)
        _shapes_match = bool(_res_shape) and (
            sorted(_res_shape.split("_")) == sorted(_nat_shape.split("_")))
        # native_us used for the column + ratio is the shape-matched one.
        native_us = _nr.get("native_us") if _nr else None
        native_prov = "resident-shape" if native_us else None
        try:
            _res = float(_rs["resident_us"]) if _rs and _rs.get("resident_us") else 0.0
            _nv = float(native_us) if native_us else 0.0
            ratio = (html.escape(f"{_res / _nv:.3f}×")
                     if (_res and _nv and _shapes_match) else "—")
        except (ValueError, TypeError, ZeroDivisionError):
            ratio = "—"
        # Dtype the raised path actually ran (from the mapped library symbol).
        # The ATen native column was measured in torch's default f32, so tag
        # every row and flag the rare raised!=native precision mismatches.
        _dt_text = (perf.get("baseline", "") + " " + perf.get("notes", "")).lower()
        if re.search(r"_f16|half(?!-pixel)|bf16", _dt_text):
            raised_dtype = "f16"
        elif re.search(r"dgemm|dsymm|dgemv|dtrsm|_f64|double", _dt_text):
            raised_dtype = "f64"
        elif re.search(r"_i8\b", _dt_text):
            raised_dtype = "i8"
        elif re.search(r"_i16\b", _dt_text):
            raised_dtype = "i16"
        elif re.search(r"_i32\b", _dt_text):
            raised_dtype = "i32"
        elif re.search(r"sgemm|sdot|saxpby|sscal|ssymm|sgemv|_f32|float|"
                       r"cudnn|cutensor|memcpy|copy", _dt_text):
            raised_dtype = "f32"
        else:
            raised_dtype = ""
        native_dtype = "f32" if native_us else ""
        if raised_dtype and native_dtype and raised_dtype != native_dtype:
            dtype_tag = (
                f'<span title="raised runs {raised_dtype}; ATen native measured '
                f'{native_dtype} — precision mismatch" '
                f'style="color:#b00020;font-size:11px;font-weight:600">'
                f'[{raised_dtype} vs native {native_dtype}]</span>'
            )
        elif raised_dtype:
            _both = raised_dtype if not native_dtype else raised_dtype
            dtype_tag = (
                f'<span title="raised{" and native" if native_dtype else ""} '
                f'{_both}" style="color:#888;font-size:11px">[{_both}]</span>'
            )
        else:
            dtype_tag = ""
        audit = cuda_audit.get(kernel, {})
        library = audit.get("candidate_library", "")
        api = audit.get("candidate_api", "")
        evidence = audit.get("evidence_url", "")
        if library:
            candidate_text = (
                f'<b>{html.escape(library)}</b><br><code>{html.escape(api)}</code>'
            )
            candidate = (
                f'<a class="viewer" href="{html.escape(evidence)}" target="_blank">'
                f'{candidate_text}</a>' if evidence else candidate_text
            )
        else:
            candidate = '<span class="none">no tensor-library API</span>'
        audit_scope = html.escape(
            audit.get("current_match_scope", "—").replace("_", " ")
        )
        implementation_class = audit.get(
            "current_implementation_class", "UNVERIFIED_IMPLEMENTATION")
        implementation_detail = audit.get("current_implementation_detail", "")
        counts_as_library = audit.get("counts_as_library_reuse") == "yes"
        implementation_provenance = (
            f'<b>{html.escape(implementation_class.replace("_", " "))}</b>'
            f'<br><span>{html.escape(implementation_detail)}</span>'
        )
        if status == "FULL" and not counts_as_library:
            status_class, status = "partial", "GPU FALLBACK"
        if any(
            symbol.startswith("cudnnAveragePool_f32_") or
            symbol.startswith("cudnnBatchNormBackward_f32_")
            for symbol in symbols
        ):
            audit_scope = "COMPLETE REWRITE CANDIDATE"
        execution_class = (
            "pass" if execution == "EXECUTED" else
            "partial" if execution.startswith("EXECUTED_") else "none"
        )
        correctness_class = "pass" if correctness == "PASS" else "none"
        if native_us:
            native_cell = (
                f'<td style="color:#137333;font-weight:600" '
                f'title="torch native at the resident shape {_nat_shape} '
                f'(f32, device-resident)">{float(native_us):.1f}</td>'
            )
        else:
            native_cell = '<td class="none">—</td>'
        # Device-resident timing: operands in cudaMalloc'd DRAM, copies
        # excluded (the same way torch measures its own kernels).
        if _rs and _rs.get("resident_us"):
            resident_cell = (
                f'<td style="color:#137333;font-weight:600" '
                f'title="raised kernel, operands device-resident (cudaMalloc, '
                f'copies excluded)">{float(_rs["resident_us"]):.1f}</td>'
            )
        else:
            resident_cell = '<td class="none">—</td>'
        # This is a separate GPU measurement from the resident number.  The
        # mapped ABI can include host/device transfers and runtime allocation,
        # so expose it without presenting it as resident execution.
        try:
            mapped_value = float(perf.get("raised_us", ""))
        except (ValueError, TypeError):
            mapped_value = 0.0
        if correctness == "PASS" and mapped_value > 0.0:
            mapped_cell = (
                f'<td title="correctness-gated mapped raised GPU execution; '
                f'host-pointer ABI transfers and allocations may be included">'
                f'{mapped_value:.1f}</td>'
            )
        else:
            mapped_cell = '<td class="none">—</td>'
        rows.append(
            f'<tr data-op="{html.escape(kernel)}" '
            f'data-native="{1 if native_us else 0}">'
            f"<td>{name} {dtype_tag}</td>"
            f"<td>{upstream}</td><td>{extracted_c}</td>"
            f"<td>{linalg_ops}</td><td>{loops}</td>"
            f'<td class="{raise_status_class}">{raise_status}</td>'
            f"<td>{launches}</td>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{symbol_html}</td>"
            f"<td>{candidates_cell}</td>"
            f"<td>{residency_cell}</td>"
            f'<td class="{correctness_class}">{correctness}</td>'
            f"<td><code>{html.escape(_res_shape.replace('_',' ')) if _res_shape else problem}</code></td>"
            f"{mapped_cell}"
            f"{resident_cell}"
            f"{native_cell}"
            f"<td>{ratio}</td><td>{baseline}</td>"
            f"<td>{assessment}</td></tr>"
        )
    total_linalg = sum(s.get("linalg_ops", 0) for s in aten_stats.values())
    total_launches = sum(s.get("launches", 0) for s in aten_stats.values())
    total_residual_loops = sum(
        s.get("residual_for", 0) for s in aten_stats.values()
    )
    fully_raised = sum(
        s.get("residual_for", 0) == 0 and s.get("linalg_ops", 0) > 0
        for s in aten_stats.values()
    )
    matched_kernels = sum(s.get("launches", 0) > 0 for s in aten_stats.values())
    structurally_complete_matches = sum(
        row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
        for row in cuda_audit.values()
    )
    complete_matches = sum(
        row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE" and
        row.get("counts_as_library_reuse") == "yes"
        for row in cuda_audit.values()
    )
    custom_fallbacks = structurally_complete_matches - complete_matches
    partial_matches = sum(
        row.get("current_match_scope") == "PARTIAL_STAGE_ONLY"
        for row in cuda_audit.values()
    )
    sort_links = (
        f'<b>Sort:</b> <a href="{_aten_page_filename("alphabetical", 1)}">'
        f'{"<b>alphabetical</b>" if sort_by == "alphabetical" else "alphabetical"}</a> &middot; '
        f'<a href="{_aten_page_filename("raised", 1)}">'
        f'{"<b>raised (matched)</b>" if sort_by == "raised" else "raised (matched)"}</a> &middot; '
        f'<a href="{_aten_page_filename("resident", 1)}">'
        f'{"<b>resident/native (measured only)</b>" if sort_by == "resident" else "resident/native (measured only)"}</a>'
    )
    page_links = " &middot; ".join(
        (
            f'<b>{number}</b>' if number == page else
            f'<a href="{_aten_page_filename(sort_by, number)}">{number}</a>'
        )
        for number in range(1, page_count + 1)
    )
    # Global op search: embed every kernel name + its detail-page link so
    # typing filters across ALL pages, not just the current pagination slice.
    search_index = []
    native_total = 0
    for k in ATEN_C_ORDER:
        us, prov = native_cuda_for(k)
        if us:
            native_total += 1
        search_index.append([
            k,
            aten_stats.get(k, {}).get("page_filename", ""),
            (f"{float(us):.1f}" if us else ""),
        ])
    search_json = json.dumps(search_index)
    search_box = (
        '<div class="intro" style="padding-top:10px;padding-bottom:4px">'
        '<b>Search op:</b> '
        '<input id="aten-search" type="text" autocomplete="off" spellcheck="false" '
        'placeholder="type an op name, e.g. gelu, conv2d, softmax…" '
        'style="width:340px;padding:5px 8px;font-size:14px;font-family:monospace;'
        'border:1px solid #bbb;border-radius:4px" '
        'oninput="atenSearch()" onkeydown="atenSearchKey(event)">'
        '<label style="margin-left:12px;font-size:13px">'
        '<input id="aten-native-only" type="checkbox" onchange="atenSearch()"> '
        'only ops with a <span style="color:#137333">native number</span></label>'
        '<span id="aten-search-count" style="margin-left:10px;color:#666"></span>'
        '<div id="aten-search-offpage" style="margin-top:6px;font-family:monospace;'
        'font-size:12px;color:#666;line-height:1.8"></div>'
        f'<div style="margin-top:4px;color:#666;font-size:12px">'
        f'<span style="color:#137333">&#9679;</span> = has a real ATen native '
        f'runtime ({native_total} of {len(ATEN_C_ORDER)} ops). Filters the table '
        f'below; matches on other pages are linked here.</div></div>'
        '<script>'
        f'var ATEN_OPS={search_json};'
        'function atenSearch(){'
        'var q=document.getElementById("aten-search").value.trim().toLowerCase();'
        'var nativeOnly=document.getElementById("aten-native-only").checked;'
        'var c=document.getElementById("aten-search-count");'
        'var off=document.getElementById("aten-search-offpage");'
        'var trs=document.querySelectorAll("tr[data-op]");'
        'var here={},shown=0;'
        'trs.forEach(function(tr){'
        'var op=tr.getAttribute("data-op").toLowerCase();here[op]=1;'
        'var ok=op.indexOf(q)>=0&&(!nativeOnly||tr.getAttribute("data-native")=="1");'
        'tr.style.display=((q||nativeOnly)&&!ok)?"none":"";'
        'if(ok)shown++;});'
        'if(!q&&!nativeOnly){c.textContent="";off.innerHTML="";return;}'
        'var all=ATEN_OPS.filter(function(o){'
        'return o[0].toLowerCase().indexOf(q)>=0&&(!nativeOnly||o[2]);});'
        'c.textContent=shown+" shown on this page, "+all.length+" total match"+(all.length==1?"":"es");'
        'var elsewhere=all.filter(function(o){return !here[o[0].toLowerCase()];});'
        'if(elsewhere.length){'
        'off.innerHTML="On other pages: "+elsewhere.slice(0,60).map(function(o){'
        'var badge=o[2]?"<span style=\\"color:#137333\\">&#9679;</span> ":"";'
        'var link=o[1]?"<a href=\\""+o[1]+"\\">"+o[0]+"</a>":o[0];'
        'return badge+link;}).join(" &middot; ")'
        '+(elsewhere.length>60?" &middot; +"+(elsewhere.length-60)+" more":"");'
        '}else{off.innerHTML="";}'
        '}'
        'function atenSearchKey(e){if(e.key=="Enter"){atenSearch();}}'
        '</script>'
    )
    controls = (
        search_box +
        '<div class="intro" style="padding-top:10px;padding-bottom:10px">'
        f'{sort_links}<span style="margin-left:24px"><b>Page:</b> '
        f'{page_links}</span><span style="margin-left:24px">Showing '
        f'{(page - 1) * ATEN_PAGE_SIZE + 1}–'
        f'{(page - 1) * ATEN_PAGE_SIZE + len(kernels)} of '
        f'{len(_aten_sorted_kernels(sort_by))}</span></div>'
    )
    return (
        '<a name="aten-c"></a>'
        '<div class="section-header"><h2 class="section-title">'
        'ATen extracted C numerical kernels</h2></div>'
        '<div class="intro">'
        f'<b>{fully_raised}/{len(aten_stats)} fully raised:</b> {total_linalg} '
        f'<code>linalg.generic</code> operations and {total_residual_loops} '
        f'residual loops. '
        f'The current matcher emitted {total_launches} launches across '
        f'{matched_kernels}/{len(aten_stats)} kernels, but exhaustive residual-IR '
        f'checking finds {complete_matches} complete genuine library/runtime '
        f'rewrites, {custom_fallbacks} complete generated GPU fallbacks, and '
        f'{partial_matches} partial stage matches. '
        'Raising FULL/PARTIAL/NONE means Linalg with no residual loops, Linalg '
        'with residual loops, or no raised Linalg, respectively. Match '
        'FULL/PARTIAL/NONE describes semantic matcher coverage; GPU FALLBACK '
        'means the complete rewrite executes compiler-authored GPU code and is '
        'not counted as CUDA-library reuse. '
        'The cuTENSOR permutation lowering preserves affine view strides and '
        'rank-reduced singleton dimensions. '
        'The newly available cuTensorNet tensor-product definition produced '
        'no additional ATen match: none of these kernels has its rank-6 '
        'separable 3D tensor-product signature. Only genuine vendor-library '
        'and CUDA-runtime definitions are counted as matches. These are '
        'standalone C extractions of ATen mathematics, not the unmodified '
        'PyTorch C++ translation units (whose direct 224-file sweep produced '
        '0 Linalg operations). Large-problem silicon results use a Jetson '
        'Orin in MAXN mode. Raised time is the current host-pointer ABI; the '
        'resident baseline keeps operands on the GPU and times only the '
        'cuBLAS/cuDNN or fused CUDA operation. Both columns are warm medians '
        'of process runs 2–4 and are shown only after correctness passes. '
        'Generic cuDNN pointwise-graph rows use 3–4 independently warmed '
        'processes (three untimed warmups and the mean of ten calls per '
        'process); their median is likewise published only when the runtime '
        'confirmed that a cuDNN graph executed and the reference comparison '
        'passed. A failed boundary-state check or a large-shape graph/build '
        'gap is displayed explicitly with its timing withheld.'
        ' <a href="performance.html"><b>Why are some kernels slow?</b></a> '
        'groups the measured gaps by cause and starts with a GEMV deep dive.'
        '</div>'
        + controls
        +
        '<table><thead><tr><th>kernel</th><th>original ATen CPU implementation</th>'
        '<th>standalone C form</th><th>Linalg ops</th>'
        '<th>residual loops</th><th>raising status</th>'
        '<th>launches</th><th>match status</th>'
        '<th>matched implementation</th>'
        '<th>candidate matches</th>'
        '<th>residency leaks<br><span style="font-weight:normal;'
        'text-transform:none;font-size:10px">allocs/copies</span></th>'
        '<th>correctness</th>'
        '<th>large problem</th>'
        '<th>mapped raised '
        '(<span style="text-transform:none">µs</span>)<br>'
        '<span style="font-weight:normal;text-transform:none;font-size:10px">'
        'host-pointer ABI</span></th>'
        '<th>resident '
        '(<span style="text-transform:none">µs</span>)</th>'
        '<th>ATen native '
        '(<span style="text-transform:none">µs</span>)</th>'
        '<th>resident / native</th>'
        '<th>resident baseline</th><th>assessment</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_polybench_results_page() -> None:
    """Render the single correctness-gated four-runtime PolyBench ledger."""
    manifest_path = SECTION42_RESULTS_DIR / "manifest.csv"
    if not manifest_path.exists():
        return
    rows = _read_csv(manifest_path)
    cpu_records = _read_csv(SECTION42_RESULTS_DIR / "performance_cpu.csv")
    gpu_records = _read_csv(SECTION42_RESULTS_DIR / "performance_gpu.csv")

    records = {
        (record.get("kernel", ""), record.get("configuration", "")): record
        for record in cpu_records + gpu_records
    }

    def cpu_time(kernel: str, configurations: tuple[str, ...]) -> str:
        for configuration in configurations:
            record = records.get((kernel, configuration), {})
            if record.get("correctness_status") == "pass" and record.get("time_ms"):
                return record["time_ms"]
        return ""

    def gpu_times(kernel: str, prefixes: tuple[str, ...]) -> tuple[str, str]:
        device = ""
        end_to_end = ""
        for (record_kernel, configuration), record in records.items():
            if record_kernel != kernel or not configuration.startswith(prefixes):
                continue
            if record.get("correctness_status") != "pass":
                continue
            device = record.get("device_time_ms", "") or device
            end_to_end = record.get("end_to_end_time_ms", "") or end_to_end
            if configuration.endswith("_device") and record.get("time_ms"):
                device = record["time_ms"]
            if configuration.endswith("_end_to_end") and record.get("time_ms"):
                end_to_end = record["time_ms"]
        return device, end_to_end

    def fmt(value: str) -> str:
        if not value:
            return "&mdash;"
        try:
            return f"{float(value):.3f} ms"
        except ValueError:
            return html.escape(value)

    def cpu_cell(status: str, value: str, detail: str) -> str:
        status = status.lower() or "unavailable"
        timing = f'<b>{fmt(value)}</b>' if value else "&mdash;"
        return (f'<span class="result-status {html.escape(status)}">'
                f'{html.escape(status)}</span><br>{timing}'
                f'<br><span class="scope">{html.escape(detail)}</span>')

    def gpu_cell(status: str, device: str, end_to_end: str, detail: str) -> str:
        status = status.lower() or "unavailable"
        timing = "&mdash;"
        if device or end_to_end:
            timing = (f'<b>device {fmt(device)}</b><br>'
                      f'<b>E2E {fmt(end_to_end)}</b>')
        return (f'<span class="result-status {html.escape(status)}">'
                f'{html.escape(status)}</span><br>{timing}'
                f'<br><span class="scope">{html.escape(detail)}</span>')

    def retained(path: Path, label: str) -> str:
        if not path.exists():
            return ""
        rel = path.relative_to(SECTION42_RESULTS_DIR)
        return (f'<a href="polybench_section42_artifacts/'
                f'{html.escape(str(rel))}">{label}</a>')

    rendered_rows = []
    incomplete = []
    for row in rows:
        kernel = row["kernel"]
        overall = row.get("overall_status", "unavailable").lower()
        stages = [row.get(field, "") for field in (
            "raise_status", "matcher_status", "residual_cpu_status",
            "cpu_library_status", "raised_gpu_status", "polybenchgpu_status")]
        has_runtime_result = any(row.get(field, "").lower() == "pass" for field in (
            "cpu_library_status", "raised_gpu_status", "polybenchgpu_status"))
        buckets = []
        if any(value in ("fail", "partial", "blocked") for value in stages):
            buckets.append("failed")
        if has_runtime_result:
            buckets.append("passed")
        else:
            buckets.append("unavailable")
        if row.get("modified_source", "false").lower() == "true":
            buckets.append("modified")
        bucket = " ".join(buckets)
        if overall != "pass":
            incomplete.append(
                f'<li><b>{html.escape(kernel)}</b>: '
                f'{html.escape(row.get("failure_reason", "not completed"))}</li>')
        native_cpu = cpu_time(kernel, ("native_clang18_noinline", "native_clang18"))
        raised_cpu = cpu_time(kernel, ("openblas_cblas_1t",))
        native_gpu_device, native_gpu_e2e = gpu_times(
            kernel, ("polybenchgpu", "native_gpu"))
        raised_gpu_device, raised_gpu_e2e = gpu_times(kernel, ("raised_gpu",))
        cpu_speedup = "&mdash;"
        gpu_speedup = "&mdash;"
        try:
            if native_cpu and raised_cpu:
                cpu_speedup = f"{float(native_cpu) / float(raised_cpu):.2f}&times;"
            if native_gpu_e2e and raised_gpu_e2e:
                gpu_speedup = f"{float(native_gpu_e2e) / float(raised_gpu_e2e):.2f}&times; E2E"
            elif native_gpu_device and raised_gpu_device:
                gpu_speedup = f"{float(native_gpu_device) / float(raised_gpu_device):.2f}&times; device"
        except (ValueError, ZeroDivisionError):
            pass
        log_links = [retained(SECTION42_RESULTS_DIR / row["log_dir"] / name, label)
                     for name, label in (("large_residual.log", "residual"),
                                         ("cpu_library_correctness.log", "CPU-lib"),
                                         ("cpu_library_timing_raw.log", "CPU time"),
                                         ("polybenchgpu_correctness.log", "native GPU"),
                                         ("polybenchgpu_timing_raw.log", "native GPU time"),
                                         ("raised_gpu_correctness.log", "raised GPU"),
                                         ("raised_gpu_timing_raw.log", "raised GPU time"))]
        ir_links = [retained(SECTION42_RESULTS_DIR / row["ir_dir"] / name, label)
                    for name, label in (("orig.mlir", "affine"),
                                        ("raised_debufferized.mlir", "raised"),
                                        ("matched.mlir", "matched"),
                                        ("residual_llvm.mlir", "LLVM"))]
        pipeline_fields = (
            ("raise", row.get("raise_status", "")),
            ("match", row.get("matcher_status", "")),
            ("CPU-lib", row.get("cpu_library_status", "")),
            ("GPU", row.get("raised_gpu_status", "")),
            ("PBGPU", row.get("polybenchgpu_status", "")),
        )
        pipeline = "<br>".join(
            f"{label}: {html.escape(value)}" for label, value in pipeline_fields)
        rendered_rows.append(
            f'<tr data-filter="{bucket}"><td><a class="kernel" href="{html.escape(kernel)}.html">'
            f'{html.escape(kernel)}</a></td>'
            f'<td>{cpu_cell(row.get("native_cpu_status", ""), native_cpu, "Clang -O3")}</td>'
            f'<td>{cpu_cell(row.get("cpu_library_status", ""), raised_cpu, "external CPU library")}</td>'
            f'<td>{gpu_cell(row.get("polybenchgpu_status", ""), native_gpu_device, native_gpu_e2e, "PolyBenchGPU CUDA")}</td>'
            f'<td>{gpu_cell(row.get("raised_gpu_status", ""), raised_gpu_device, raised_gpu_e2e, "external CUDA library")}</td>'
            f'<td>CPU {cpu_speedup}<br>GPU {gpu_speedup}</td><td>{pipeline}</td>'
            f'<td>{html.escape(row.get("failure_reason", ""))}</td>'
            f'<td>{" &middot; ".join(x for x in log_links if x) or "&mdash;"}</td>'
            f'<td>{" &middot; ".join(x for x in ir_links if x) or "&mdash;"}</td></tr>')

    body = (
        '<div class="header"><h1><a href="index.html">Polygeist IR explorer</a></h1>'
        '<div><a href="index.html">Overview</a> &middot; '
        '<a href="polybench.html">PolyBench results</a></div></div>'
        '<div class="intro"><b>PolyBench four-runtime correctness-gated results.</b> '
        'All rows use checked-in PolyBench/C initialization, LARGE dimensions, and FP64. '
        '<b>Native CPU</b> is the original C kernel at Clang -O3; <b>raised CPU</b> is '
        'the raised/matched path through a real optimized CPU library; <b>native GPU</b> '
        'is equivalent handwritten PolyBenchGPU CUDA; and <b>raised GPU</b> is the '
        'raised/matched path through real CUDA libraries. CPU times are pinned '
        'single-core medians with one OpenBLAS thread. GPU cells report CUDA-event '
        'device time and PolyBench kernel-call end-to-end time separately. A timing is '
        'shown only after that exact configuration passes correctness. Native GPU '
        'rows marked modified source retain the external PolyBenchGPU computational '
        'kernels while normalizing FP64 data, LARGE dimensions, canonical inputs, ABI, '
        'and timing. Same-named upstream programs with different algorithms remain '
        'unavailable.</div>'
        '<div class="s42-filters">Show: <button onclick="s42Filter(\'all\')">all</button> '
        '<button onclick="s42Filter(\'passed\')">passed</button> '
        '<button onclick="s42Filter(\'failed\')">failed</button> '
        '<button onclick="s42Filter(\'unavailable\')">unavailable</button> '
        '<button onclick="s42Filter(\'modified\')">modified source</button></div>'
        '<div class="table-wrap"><table class="s42"><thead><tr><th>kernel</th>'
        '<th>native CPU runtime</th><th>raised CPU runtime</th>'
        '<th>native GPU runtime</th><th>raised GPU runtime</th>'
        '<th>speedup</th><th>pipeline status</th><th>result / blocker</th>'
        '<th>logs</th><th>IR</th></tr></thead><tbody>' + "".join(rendered_rows) +
        '</tbody></table></div><div class="section-header"><h2 class="section-title">'
        'Incomplete or blocked experiments</h2></div><div class="intro"><ul>' +
        ("".join(incomplete) or '<li>None.</li>') + '</ul></div>'
        '<script>function s42Filter(v){document.querySelectorAll("table.s42 tbody tr")'
        '.forEach(r=>r.style.display=(v==="all"||r.dataset.filter.split(" ").includes(v))?"":"none");}</script>')
    css = (
        '.s42-filters{padding:12px 20px}.s42-filters button{margin-right:6px}'
        '.table-wrap{overflow:auto;padding:0 20px}.s42{border-collapse:collapse;font-size:12px}'
        '.s42 th,.s42 td{border:1px solid #d8dee8;padding:6px;vertical-align:top}'
        '.s42 th{background:#eef2f7;position:sticky;top:0}.s42 td{white-space:nowrap}'
        '.s42 td:nth-child(8){white-space:normal;min-width:260px}'
        '.result-status{display:inline-block;border-radius:10px;padding:2px 7px;'
        'font-size:10px;font-weight:bold;text-transform:uppercase;background:#eee}'
        '.result-status.pass{background:#dff5e5;color:#176b35}'
        '.result-status.fail,.result-status.partial{background:#ffe2e2;color:#8a1c1c}'
        '.result-status.blocked{background:#fff0c2;color:#745600}'
        '.scope{font-size:10px;color:#666}'
        '.section-header{background:#eaeefa;padding:8px 20px;margin-top:20px}'
        '.section-title{margin:0;font-size:16px}')
    OUTPUT_DIR.joinpath("polybench.html").write_text(
        render_html("Polygeist: PolyBench four-runtime results", body, css))
    artifact_link = OUTPUT_DIR / "polybench_section42_artifacts"
    if artifact_link.is_symlink():
        if artifact_link.resolve() != SECTION42_RESULTS_DIR.resolve():
            # The viewer output is shared across local worktrees.  Refresh this
            # narrowly named generated link when results are rebuilt elsewhere.
            artifact_link.unlink()
            artifact_link.symlink_to(SECTION42_RESULTS_DIR, target_is_directory=True)
    elif not artifact_link.exists():
        artifact_link.symlink_to(SECTION42_RESULTS_DIR, target_is_directory=True)


def _modified_kernels_page() -> tuple[str, int]:
    """Describe source forms introduced because upstream code was not directly usable.

    This is deliberately an evidence/provenance page, not another success-count
    page.  A source extraction is not automatically a frontend bug, and a
    normalized source is not counted as an unmodified-source compiler result.
    """
    rows: list[dict[str, str]] = []

    def add(suite: str, kernel: str, original: str, modified: str,
            category: str, direct_result: str, change: str, diagnosis: str,
            evidence: str, validation: str) -> None:
        rows.append({
            "suite": suite, "kernel": kernel, "original": original,
            "modified": modified, "category": category,
            "direct_result": direct_result, "change": change,
            "diagnosis": diagnosis, "evidence": evidence,
            "validation": validation,
        })

    # The original translation units inline initialization and the stencil
    # into main.  Constant folding then removes the input operand that the
    # matcher needs.  This is a pipeline/input-isolation interaction, not a
    # claim that cgeist mistranslated the source.
    for kernel in ("conv2d", "conv3d"):
        add(
            "PolyBenchGPU", kernel,
            f"third_party/polybenchGpu/OpenMP/stencils/convolution-{kernel[-2:]}/"
            f"convolution-{kernel[-2:]}.c",
            f"third_party/polybenchGpu-extracted/{kernel}.c",
            "preprocessing / inlining",
            "Whole-TU inlining exposes init_array; constant folding removes the "
            "stencil input operand, so semantic convolution matching is bypassed.",
            "Isolated the kernel in its own translation unit, made arrays explicit "
            "parameters, and fixed compile-time extents.",
            "Not a frontend correctness bug; this is an inlining and benchmark-"
            "harness interaction that changes the IR denominator.",
            f"third_party/polybenchGpu-extracted/{kernel}.c (file header)",
            "Extracted source raises cleanly; equivalence/performance is tracked on "
            "the PolyBenchGPU page.",
        )

    npb_details = {
        "bt-add": (
            "BT/bt.c:add", "file-local static 4D arrays and bounds loaded from the "
            "runtime grid_points array", "Passed arrays explicitly and converted "
            "grid bounds to scalar parameters so loops are affine.",
            "raising representation / extraction"),
        "ft-evolve": (
            "FT/ft.c:evolve", "dcomplex structs plus static/global benchmark state",
            "Flattened complex values to a trailing real/imag dimension and passed "
            "dimensions and arrays explicitly.", "frontend representation / extraction"),
        "lu-l2norm": (
            "LU/lu.c:l2norm", "kernel embedded in a monolithic benchmark with global "
            "state and runtime bounds", "Isolated the function while preserving NPB "
            "padding and passed bounds/data explicitly.", "framework extraction"),
        "mg-psinv": (
            "MG/mg.c:psinv", "double*** storage and file-level benchmark state do not "
            "form a static ranked memref suitable for the raising pipeline",
            "Re-expressed triple-pointer data as fixed-size rank-3 arrays and exposed "
            "bounds and coefficients as parameters.", "memory representation / raising"),
        "mg-resid": (
            "MG/mg.c:resid", "double*** storage, static state, and scratch-row staging",
            "Used fixed-size rank-3 arrays, explicit parameters, and preserved the "
            "scratch-row stencil algorithm.", "memory representation / raising"),
        "mg-norm2u3": (
            "MG/mg.c:norm2u3", "monolithic/global context; max/fabs branch is not a "
            "pure affine reduction", "Isolated the function and used local scalar "
            "helpers; retained both L2 and L-infinity semantics.",
            "extraction plus remaining raising gap"),
        "mg-rprj3": (
            "MG/mg.c:rprj3", "double*** fine/coarse grids, global sizes, and runtime "
            "boundary-dependent step factors", "Converted grids to fixed ranked arrays "
            "and passed fine/coarse bounds and step factors explicitly.",
            "memory representation / raising"),
    }
    for kernel, (original, direct, change, category) in npb_details.items():
        source = NPB_KERNELS[kernel][0]
        add(
            "NPB 3.0", kernel, f"third_party/NPB3.0-omp-C/{original}",
            f"third_party/NPB-polybenchified/{source}", category,
            direct, change,
            "Primarily extraction and memory-shape normalization; do not report as an "
            "unmodified NPB source result.",
            f"third_party/NPB-polybenchified/{source} (file header)",
            "The extracted source is tracked independently from the original suite.",
        )

    # Every faithful MFEM form reaches the frontend and raising pass, but all
    # retain imperative loops.  The paired normalized forms are the honest
    # place to record structural changes made for complete Linalg recovery.
    mfem_all_rows = _read_csv(MFEM_C_ROOT / "results/summary.csv")
    mfem_rows = [
        row for row in mfem_all_rows
        if row.get("variant") == "normalized"
    ]
    for row in mfem_rows:
        original_id = re.sub(r"_(scratch_sliced|stage_sliced|scalarized)$", "", row["id"])
        original_row = next(
            (candidate for candidate in mfem_all_rows
             if candidate.get("variant") == "original"
             and candidate.get("operation") == row.get("operation")
             and candidate.get("dimension") == row.get("dimension")),
            {},
        )
        if row["id"].endswith("scalarized"):
            change = (
                "Scalarized pointwise elasticity outputs and separated dependent "
                "stages so each result has an explicit producer/consumer boundary."
            )
        elif "scratch_sliced" in row["id"]:
            change = (
                "Sliced temporary scratch by element/component so independent outer "
                "iterations no longer appear to reuse the same storage."
            )
        else:
            change = (
                "Split sum-factorized work into explicit contraction stages and sliced "
                "scratch/component storage to remove false reuse dependencies."
            )
        add(
            "MFEM", row["id"],
            f'issues/mfem_c_kernels/{original_row.get("source", "original/")}',
            f'issues/mfem_c_kernels/{row["source"]}',
            "raising normalization",
            f'The faithful source reached cgeist/raise but retained '
            f'{original_row.get("residual_loops", "one or more")} imperative loop(s).',
            change,
            "Raising limitation: scratch reuse and multi-stage dependencies are not "
            "automatically normalized. One isolated remove-iter-args dominance defect "
            "also requires stage separation; see the MFEM status evidence.",
            "issues/mfem_c_kernels/STATUS.md and results/summary.csv",
            "Normalized form is fully raised with zero residual loops and validated "
            "against the faithful original (maximum reported error 7.2e-15).",
        )

    llama_modifications = [
        (
            "rope_split", "kernel_llama_rope (interleaved layout)",
            "Split even/odd Q and K values into separate tensors.",
            "Exact interleaved indexing remains a raising gap; the split form is a "
            "raise-friendly semantic variant."
        ),
        (
            "attention_mask_select", "kernel_llama_attention_mask (branchy if/else)",
            "Replaced the conditional store with an equivalent branchless select.",
            "The branchy mask retains if/loop IR; select-form raises. This is a control-"
            "flow raising limitation."
        ),
    ]
    for kernel, original, change, diagnosis in llama_modifications:
        add(
            "Llama forward", kernel,
            f"third_party/cnn-extracted/llama_forward_ops.c:{original}",
            f"third_party/cnn-extracted/llama_forward_ops.c:{LLAMA_FORWARD_KERNELS[kernel][1]}",
            "raising issue", "The exact source form was not fully raised or run.",
            change, diagnosis,
            "third_party/cnn-extracted/llama2_extended_forward_bench.c (file header) "
            "and the AI viewer rows", "The alternative form raises and has a tracked "
            "Jetson fixture run; it is not an unmodified-source result.",
        )

    darknet_changes = {
        "conv2d_batched": (
            "Specialized framework tensors and runtime dimensions into fixed NCHW "
            "arrays/macros and isolated the convolution body.",
            "Framework extraction; compile-time bounds also serve the affine raiser."),
        "darknet_im2col_gemm": (
            "Placed im2col and GEMM in one standalone translation unit so inlining "
            "exposes the producer/consumer pair.",
            "Framework/external-call isolation for semantic composition."),
        "maxpool_batched": (
            "Used an equivalent ternary/select update instead of a conditional store.",
            "Control-flow raising limitation: the conditional store remains imperative."),
        "batchnorm_batched": (
            "Isolated the inference arithmetic from layer objects and framework dispatch.",
            "Framework extraction, not a demonstrated frontend bug."),
        "shortcut_batched": (
            "Isolated residual-add arithmetic with fixed tensor extents.",
            "Framework extraction, not a demonstrated frontend bug."),
        "conv_bn_relu_batched": (
            "Constructed a chained fixed-shape C fixture for convolution, batch norm, "
            "and ReLU composition.",
            "Algorithmic fixture for composition; not a textual upstream kernel."),
    }
    for kernel, (change, diagnosis) in darknet_changes.items():
        source = EXTRACTED_DARKNET_KERNELS[kernel][0]
        add(
            "Darknet/CNN", kernel, "third_party/darknet (framework implementation)",
            f"third_party/cnn-extracted/{source}",
            "framework extraction" if "raising" not in diagnosis else "raising issue",
            "The full-source sweep either presents framework/dispatch code or does not "
            "expose this complete arithmetic body as one raiseable static kernel.",
            change, diagnosis, f"third_party/cnn-extracted/{source} and viewer notes",
            "Standalone fixture is tracked separately; it must not be counted as direct "
            "translation of the full Darknet source file.",
        )

    # Whisper rows are semantic C fixtures.  Two point back to large upstream
    # ggml/stb sources; the others isolate the corresponding operation family.
    for kernel in WHISPER_OPS_ORDER:
        source, function = WHISPER_OPS_KERNELS[kernel]
        add(
            "Whisper/ggml", kernel,
            "third_party/whisper.cpp/ggml framework or embedded codec source",
            f"third_party/cnn-extracted/{source}:{function}",
            "framework extraction",
            "The upstream implementation is embedded in tensor metadata, SIMD/dispatch, "
            "helper calls, or a very large translation unit rather than a standalone "
            "static loop kernel suitable for this pipeline.",
            "Isolated the numerical body with fixed ranks, extents, dtype, and ordinary "
            "C arrays; retained helper calls when they are part of the tested limitation.",
            "Not classified as a frontend bug without a kernel-specific direct-source "
            "failure. This is a semantic fixture/extraction boundary.",
            f"{source} and notes/whisper_linalg_raise_results.md",
            "Results describe the extracted fixture only, not unmodified whisper.cpp.",
        )

    # ATen is too large for one main-table row per specialization.  Preserve
    # every name and upstream pointer in an expandable, searchable inventory.
    aten_items = []
    for kernel in ATEN_C_ORDER:
        upstream = ATEN_C_PROVENANCE.get(kernel)
        pointer = (
            f"{upstream[0]}:{upstream[1]}" if upstream and upstream[1]
            else (upstream[0] if upstream else "upstream pointer unavailable")
        )
        page = f'{kernel}.html' if kernel in ATEN_C_KERNELS else "numerical.html"
        aten_items.append(
            f'<li><a href="{html.escape(page)}"><code>{html.escape(kernel)}</code></a> '
            f'&larr; <code>{html.escape(pointer)}</code></li>'
        )
    add(
        "PyTorch ATen", f"{len(ATEN_C_ORDER)} standalone C specializations",
        f"third_party/pytorch at {ATEN_UPSTREAM_COMMIT}",
        "issues/aten_c_kernels/aten_*.c",
        "framework/frontend boundary",
        "The direct 224-file ATen C/C++ translation-unit sweep produced zero Linalg "
        "operations: numerical loops are hidden behind TensorIterator, templates, "
        "dispatch/registration, vectorization, dynamic shapes, and Tensor APIs.",
        "Removed framework orchestration and specialized rank, extent, dtype, and modes "
        "while preserving the selected scalar or loop algorithm.",
        "Not one blanket frontend bug. It measures an unsupported framework abstraction "
        "boundary; each C file is an extracted specialization, not upstream ATen text.",
        "issues/aten_c_kernels/extraction_inventory.csv and generated*_provenance.csv",
        "Each fixture has provenance; raising/matching/correctness are reported on the "
        "ATen pages. Expand the complete inventory below.",
    )

    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    category_html = "".join(
        f'<li><b>{html.escape(category)}</b>: {count} table row(s)</li>'
        for category, count in sorted(categories.items())
    )
    table_rows = []
    for row in rows:
        searchable = " ".join(row.values()).lower()
        table_rows.append(
            f'<tr data-search="{html.escape(searchable, quote=True)}">'
            f'<td><b>{html.escape(row["suite"])}</b></td>'
            f'<td><code>{html.escape(row["kernel"])}</code></td>'
            f'<td><code>{html.escape(row["original"])}</code></td>'
            f'<td><code>{html.escape(row["modified"])}</code></td>'
            f'<td><span class="mod-tag">{html.escape(row["category"])}</span><br>'
            f'{html.escape(row["diagnosis"])}</td>'
            f'<td>{html.escape(row["direct_result"])}</td>'
            f'<td>{html.escape(row["change"])}</td>'
            f'<td><code>{html.escape(row["evidence"])}</code><br>'
            f'<small>{html.escape(row["validation"])}</small></td></tr>'
        )
    body = (
        '<div class="section-header"><h2 class="section-title">Modified and extracted '
        'kernel sources</h2></div>'
        '<div class="intro"><b>Purpose:</b> identify every benchmark source family for '
        'which the published result uses an extracted, specialized, or structurally '
        'normalized C form instead of compiling the upstream source directly. '
        '<b>These rows are not all compiler bugs.</b> The diagnosis distinguishes '
        'framework extraction, preprocessing/inlining effects, unsupported memory or '
        'control-flow representations, raising limitations, and a confirmed compiler '
        'defect. Modified forms must be reported separately from unmodified-source '
        'coverage.<br><br><b>Confirmed compiler defect:</b> MFEM exposed an invalid '
        '<code>remove-iter-args</code> dominance relation when a reduction result is '
        'combined with a load defined after the reduction. Stage separation avoids it. '
        'The general MFEM scratch/staging problem is a raising-normalization limitation, '
        'not solely that defect.'
        f'<ul>{category_html}</ul>'
        '<label><b>Filter rows:</b> <input id="modified-filter" type="search" '
        'placeholder="suite, kernel, category, reason…" oninput="filterModified()" '
        'style="width:min(520px,80%);padding:6px"></label></div>'
        '<table class="audit-table modified-table"><thead><tr>'
        '<th>suite</th><th>kernel / cohort</th><th>upstream form</th>'
        '<th>form actually tested</th><th>classification</th>'
        '<th>why direct form was not used</th><th>source change</th>'
        '<th>evidence and validation</th></tr></thead><tbody>'
        + "\n".join(table_rows) + '</tbody></table>'
        '<details class="intro"><summary><b>Complete ATen extracted-kernel inventory '
        f'({len(aten_items)} kernels)</b></summary><p>Each entry is a fixed-shape '
        'standalone C specialization linked to its viewer page and pinned upstream '
        'implementation-family pointer. The shared reason is framework isolation; '
        'kernel-specific raising outcomes remain on the ATen pages.</p><ul class="columns">'
        + "".join(aten_items) + '</ul></details>'
        '<script>function filterModified(){var q=document.getElementById('
        '"modified-filter").value.toLowerCase();document.querySelectorAll('
        '".modified-table tbody tr").forEach(function(r){r.style.display='
        'r.dataset.search.indexOf(q)>=0?"":"none";});}</script>'
    )
    individual_count = len(rows) - 1 + len(ATEN_C_ORDER)
    return body, individual_count


def _extract_c_function(text: str, function: str) -> str:
    """Extract one named C function, including a directly preceding comment."""
    match = re.search(rf"\b{re.escape(function)}\s*\(", text)
    if not match:
        return text
    opening = text.find("{", match.end())
    if opening < 0:
        return text
    depth = 0
    end = opening
    while end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    start = text.rfind("\n", 0, match.start()) + 1
    # Include contiguous // comments immediately above the declaration.
    while start > 0:
        previous_end = start - 1
        previous_start = text.rfind("\n", 0, previous_end) + 1
        if not text[previous_start:previous_end].lstrip().startswith("//"):
            break
        start = previous_start
    return text[start:end].strip() + "\n"


def _mfem_upstream_line(upstream_file: str, upstream_symbol: str) -> int | None:
    path = MFEM_UPSTREAM_ROOT / upstream_file
    if not path.exists():
        return None
    # Manifest spellings such as ElasticityAddMultPA_<2> denote a template
    # specialization; the source definition is ElasticityAddMultPA_.
    symbol = re.sub(r"<[^>]+>$", "", upstream_symbol)
    definition = re.compile(rf"\b{re.escape(symbol)}\s*\(")
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if definition.search(line):
            return line_number
    return None


def build_mfem_pages() -> list[dict]:
    """Render stored MFEM frontend/raise/matcher artifacts.

    MFEM uses a manifest-driven artifact layout rather than the conventional
    <kernel>[_linalg|_debuf].mlir layout used by the other explorer suites.
    Stored rewritten IR is authoritative for executable launches.
    """
    manifest = _read_csv(MFEM_C_ROOT / "manifest.csv")
    raise_rows = {
        (row["id"], row["variant"]): row
        for row in _read_csv(MFEM_RESULTS_DIR / "summary.csv")
    }
    match_rows = {
        row["id"]: row
        for row in _read_csv(MFEM_MATCH_RESULTS_DIR / "summary.csv")
    }
    composition_rows = {
        row["id"]: row
        for row in _read_csv(MFEM_MATCH_RESULTS_DIR / "composition_summary.csv")
    }
    silicon_rows = {
        row["id"]: row
        for row in _read_csv(
            MFEM_SILICON_RESULTS_DIR / "native_vs_raised_large_ne.csv"
        )
    }
    stats = []
    for row in manifest:
        ident = row["id"]
        variant = row["variant"]
        artifact_stem = f"{ident}__{variant}"
        frontend = MFEM_RESULTS_DIR / f"{artifact_stem}.frontend.mlir"
        raised = MFEM_RESULTS_DIR / f"{artifact_stem}.raised.mlir"
        match_dir = MFEM_MATCH_RESULTS_DIR / ident
        debufferized = match_dir / "debufferized.mlir"
        matched = match_dir / "matched.mlir"
        composed = match_dir / "composed.mlir"
        abi = match_dir / "abi.mlir"
        source = MFEM_C_ROOT / row["source"]
        raise_row = raise_rows.get((ident, variant), {})
        match_row = match_rows.get(ident, {}) if variant == "normalized" else {}
        composition_row = (
            composition_rows.get(ident, {}) if variant == "normalized" else {}
        )
        silicon_row = silicon_rows.get(ident, {}) if variant == "normalized" else {}

        blocks = []
        css = ""
        source_text = ""
        if source.exists():
            source_text = _extract_c_function(source.read_text(), row["function"])
            highlighted, css = syntax_highlight(source_text, "c")
            source_label = html.escape(row["source"])
            function_label = html.escape(row["function"])
            blocks.append(
                '<h2 id="extracted-c">extracted C lowered by cgeist</h2>'
                '<div class="summary" style="padding:8px 20px; '
                'border:1px solid #eee; background:#fafafa; font-size:13px;">'
                f'<b>Corpus source:</b> <code>{source_label}</code> &nbsp;·&nbsp; '
                f'<b>function:</b> <code>{function_label}</code></div>'
                f'<div class="container">{highlighted}</div>'
            )

        upstream_file = row["upstream_file"]
        upstream_symbol = row["upstream_symbol"]
        upstream_line = _mfem_upstream_line(upstream_file, upstream_symbol)
        line_fragment = f"#L{upstream_line}" if upstream_line else ""
        upstream_url = (
            "https://github.com/mfem/mfem/blob/"
            f"{MFEM_UPSTREAM_COMMIT}/{upstream_file}{line_fragment}"
        )
        local_pointer = f"third_party/mfem/{upstream_file}"
        if upstream_line:
            local_pointer += f":{upstream_line}"
        c_page_filename = f"mfem_c_{ident}.html"
        if source_text:
            c_highlighted, c_css = syntax_highlight(source_text, "c")
            c_header = (
                '<div class="header"><h1><a href="mfem.html">← MFEM</a> '
                f'&nbsp; extracted C: {html.escape(ident)} '
                f'<small>({html.escape(variant)})</small></h1></div>'
            )
            c_provenance = (
                '<div class="summary" style="padding:10px 20px; '
                'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
                f'<b>Corpus source:</b> <code>{html.escape(row["source"])}</code><br>'
                f'<b>Function lowered:</b> <code>{html.escape(row["function"])}</code><br>'
                f'<b>Upstream symbol:</b> <code>{html.escape(upstream_symbol)}</code><br>'
                f'<b>Pinned source:</b> <a href="{html.escape(upstream_url)}" '
                f'target="_blank"><code>{html.escape(local_pointer)}</code></a><br>'
                f'<b>MFEM commit:</b> <code>{MFEM_UPSTREAM_COMMIT}</code> &nbsp;·&nbsp; '
                f'<a href="mfem_{html.escape(ident)}.html">view lowering IR →</a>'
                '</div>'
            )
            OUTPUT_DIR.joinpath(c_page_filename).write_text(
                render_html(
                    f"MFEM extracted C: {ident}",
                    c_header + c_provenance
                    + '<h2>exact extracted function lowered by cgeist</h2>'
                    + f'<div class="container">{c_highlighted}</div>',
                    c_css,
                )
            )
        blocks.append(
            '<h2 id="provenance">MFEM extraction provenance</h2>'
            '<div class="summary" style="padding:10px 20px; '
            'border:1px solid #eee; background:#fafafa; font-size:13px;">'
            f'<b>Upstream symbol:</b> <code>{html.escape(upstream_symbol)}</code><br>'
            f'<b>Pinned source:</b> <a href="{html.escape(upstream_url)}" '
            f'target="_blank"><code>{html.escape(local_pointer)}</code></a><br>'
            f'<b>MFEM commit:</b> <code>{MFEM_UPSTREAM_COMMIT}</code>'
            '</div>'
        )

        stage_paths = [
            ("frontend", "cgeist output (pre-raise MLIR)", frontend),
            ("raised", "raised Linalg IR", raised),
            ("debufferized", "debufferized tensor Linalg (matcher input)",
             debufferized),
            ("matched", "executable matcher output (kernel.launch)",
             matched),
            ("composed", "post-match composed cuTensorNet network IR",
             composed),
            ("abi", "ABI-lowered IR (func.call to runtime shim)", abi),
        ]
        for anchor, title, path in stage_paths:
            if not path.exists():
                continue
            highlighted, css = syntax_highlight(path.read_text())
            blocks.append(
                f'<h2 id="{anchor}">{title}</h2>'
                f'<div class="container">{highlighted}</div>'
            )

        matcher_launches = int(match_row.get("kernel_launches", "0") or 0)
        launches = int(
            composition_row.get("composed_launches", matcher_launches) or 0
        )
        symbols = [
            value for value in composition_row.get(
                "composed_symbols", match_row.get("launch_symbols", "")
            ).split(",")
            if value
        ]
        network_launches = int(
            composition_row.get("network_launches", "0") or 0
        )
        linalg_ops = int(raise_row.get("linalg_ops", "0") or 0)
        residual_loops = int(raise_row.get("residual_loops", "0") or 0)
        fully_raised = raise_row.get("fully_raised") == "true"
        ce_url = ce_link_from_paths(source if source.exists() else None, frontend)
        open_link = (
            f'<a href="{ce_url}" target="_blank" '
            'style="margin-left:12px; color:#0366d6;">'
            'open in Compiler Explorer →</a>'
        ) if ce_url else ""
        c_link = (
            f'<a href="{c_page_filename}" style="margin-left:12px; '
            'color:#0366d6;">view extracted C →</a>'
        ) if source_text else ""
        title = html.escape(ident)
        summary = (
            '<div class="summary" style="padding:8px 20px; '
            'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
            f'<b>{linalg_ops}</b> Linalg op(s) &nbsp;·&nbsp; '
            f'<b>{residual_loops}</b> residual loop(s) &nbsp;·&nbsp; '
            f'<b>{matcher_launches}→{launches}</b> library launches after '
            f'composition &nbsp;·&nbsp; <b>{network_launches}</b> composed '
            'cuTensorNet network(s)'
            '</div>'
        )
        header = (
            '<div class="header"><h1><a href="mfem.html">← MFEM</a> '
            f'&nbsp; {title} <small>({html.escape(variant)})</small>'
            f'{c_link}{open_link}</h1></div>'
        )
        page_filename = f"mfem_{ident}.html"
        OUTPUT_DIR.joinpath(page_filename).write_text(
            render_html(f"MFEM: {ident}", header + summary + "\n".join(blocks), css)
        )
        stats.append({
            **row,
            "page_filename": page_filename,
            "c_page_filename": c_page_filename,
            "upstream_line": upstream_line,
            "upstream_url": upstream_url,
            "upstream_pointer": local_pointer,
            "linalg_ops": linalg_ops,
            "residual_loops": residual_loops,
            "fully_raised": fully_raised,
            "launches": launches,
            "matcher_launches": matcher_launches,
            "network_launches": network_launches,
            "matched_symbols": symbols,
            "silicon": silicon_row,
        })
    return stats


def build_mfem_application_pages() -> list[dict]:
    """Render MFEM example hot-operator ports and measured status."""
    stats = []
    composition_rows = {
        row["id"]: row
        for row in _read_csv(MFEM_MATCH_RESULTS_DIR / "composition_summary.csv")
    }
    for row in _read_csv(MFEM_APPLICATIONS_DIR / "summary.csv"):
        ident = row["id"]
        harness = MFEM_APPLICATIONS_DIR / row["harness"]
        normalized = (MFEM_APPLICATIONS_DIR / row["normalized"]).resolve()
        blocks = []
        css = ""
        for anchor, title, path in (
            ("harness", "application hot-operator harness", harness),
            ("normalized", "stage-sliced implementation", normalized),
        ):
            if not path.exists():
                continue
            highlighted, css = syntax_highlight(path.read_text())
            blocks.append(
                f'<h2 id="{anchor}">{title}</h2>'
                f'<div class="container">{highlighted}</div>'
            )

        status = html.escape(row["raised_status"])
        composition = composition_rows.get(row["kernel_id"], {})
        before = int(
            composition.get("matcher_launches", row["library_launches"]) or 0
        )
        after = int(composition.get("composed_launches", before) or 0)
        networks = int(composition.get("network_launches", "0") or 0)
        kernel_page = f'mfem_{html.escape(row["kernel_id"])}.html'
        summary = (
            '<div class="summary" style="padding:10px 20px; '
            'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
            f'<b>{html.escape(row["application"])}</b> &nbsp;·&nbsp; '
            f'{html.escape(row["operator"])} &nbsp;·&nbsp; '
            f'{html.escape(row["dimension"])}D &nbsp;·&nbsp; '
            f'<b>{html.escape(row["speedup"])}x</b> stage-sliced C speedup '
            f'({html.escape(row["reference_us"])} us → '
            f'{html.escape(row["sliced_us"])} us) &nbsp;·&nbsp; '
            f'max error <b>{html.escape(row["max_error"])}</b><br>'
            f'<b>Library-backed status:</b> {status}; '
            f'{before}→{after} structural launch(es) after composition; '
            f'{networks} composed network(s). '
            f'<b>Blocker:</b> {html.escape(row["blocker"])}. '
            f'<a href="{kernel_page}">Open the kernel IR and matches →</a>'
            '</div>'
        )
        header = (
            '<div class="header"><h1><a href="mfem.html">← MFEM</a> '
            f'&nbsp; {html.escape(ident)}</h1></div>'
        )
        page_filename = f"mfem_app_{ident}.html"
        OUTPUT_DIR.joinpath(page_filename).write_text(
            render_html(
                f"MFEM application: {ident}",
                header + summary + "\n".join(blocks),
                css,
            )
        )
        stats.append({
            **row, "page_filename": page_filename,
            "composition_launches": f"{before}→{after}",
            "network_launches": str(networks),
        })
    return stats


def build_mfem_application_extraction_pages() -> list[dict]:
    """Render raised hot paths extracted from larger MFEM applications."""
    stats = []
    summary = _read_csv(MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / "summary.csv")
    composition_rows = {
        row["id"]: row for row in _read_csv(
            MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / "composition_summary.csv"
        )
    }
    comparison_rows = {
        row["function"]: row
        for row in _read_csv(
            MFEM_SILICON_RESULTS_DIR
            / "application_native_vs_raised_large_ne.csv"
        )
    }
    for row in summary:
        function = row["function"]
        source = MFEM_APPLICATION_EXTRACTIONS_DIR / row["source"]
        support_source = None
        if row.get("support_source"):
            support_source = MFEM_APPLICATION_EXTRACTIONS_DIR / row["support_source"]
        frontend = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.frontend.mlir"
        raised = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.raised.mlir"
        debufferized = (
            MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.debufferized.mlir"
        )
        matched = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.matched.mlir"
        composed = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.composed.mlir"
        abi = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.abi.mlir"
        log = MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / f"{function}.log"
        blocks = []
        css = ""

        for anchor, title, path, language in (
            ("source", "extracted C application hot path", source, "c"),
            ("support-source", "extracted supporting operator kernels",
             support_source, "c"),
            ("frontend", "cgeist output (pre-raise MLIR)", frontend, None),
            ("raised", "raised Linalg IR", raised, None),
            ("debufferized", "debufferized tensor Linalg", debufferized, None),
            ("matched", "matcher-rewritten candidate launches", matched, None),
            ("composed", "post-match composed cuTensorNet network IR",
             composed, None),
            ("abi", "ABI-lowered IR (func.call to runtime shim)", abi, None),
            ("matcher-report", "raising and matcher report", log, None),
        ):
            if path is None or not path.exists():
                continue
            highlighted, css = syntax_highlight(path.read_text(), language)
            blocks.append(
                f'<h2 id="{anchor}">{title}</h2>'
                f'<div class="container">{highlighted}</div>'
            )

        upstream_file = row["upstream_file"]
        upstream_lines = row["upstream_lines"]
        first_line = re.match(r"\d+", upstream_lines)
        fragment = f"#L{first_line.group(0)}" if first_line else ""
        upstream_url = (
            "https://github.com/mfem/mfem/blob/"
            f"{MFEM_UPSTREAM_COMMIT}/{upstream_file}{fragment}"
        )
        local_pointer = f"third_party/mfem/{upstream_file}:{upstream_lines}"
        page_filename = f"mfem_benchmark_{function}.html"
        c_page_filename = f"mfem_benchmark_c_{function}.html"
        if source.exists():
            c_highlighted, c_css = syntax_highlight(source.read_text(), "c")
            support_html = ""
            if support_source is not None and support_source.exists():
                support_highlighted, _ = syntax_highlight(
                    support_source.read_text(), "c"
                )
                support_html = (
                    '<h2>supporting extracted operator kernels: '
                    f'<code>{html.escape(row["support_source"])}</code></h2>'
                    f'<div class="container">{support_highlighted}</div>'
                )
            c_header = (
                '<div class="header"><h1><a href="mfem.html">← MFEM</a> '
                f'&nbsp; extracted C: {html.escape(function)}</h1></div>'
            )
            c_provenance = (
                '<div class="summary" style="padding:10px 20px; '
                'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
                f'<b>Application:</b> {html.escape(row["application"])}<br>'
                f'<b>Corpus source:</b> <code>{html.escape(row["source"])}</code><br>'
                f'<b>Function lowered:</b> <code>{html.escape(function)}</code><br>'
                f'<b>Upstream:</b> <a href="{html.escape(upstream_url)}" '
                f'target="_blank"><code>{html.escape(local_pointer)}</code></a><br>'
                f'<a href="{html.escape(page_filename)}">view lowering IR →</a>'
                '</div>'
            )
            OUTPUT_DIR.joinpath(c_page_filename).write_text(
                render_html(
                    f"MFEM application extracted C: {function}",
                    c_header + c_provenance
                    + '<h2>exact application hot-path C lowered by cgeist</h2>'
                    + f'<div class="container">{c_highlighted}</div>'
                    + support_html,
                    c_css,
                )
            )
        missing = row.get("missing_operator", "") or "none"
        families = row.get("operator_families", "") or "—"
        loops = int(row.get("residual_loops", "0") or 0)
        composition = composition_rows.get(function, {})
        selected_ir = composed if composition.get("composition_ok") == "true" else matched
        matched_symbols = []
        if selected_ir.exists():
            matched_symbols = sorted(set(re.findall(
                r"kernel\.launch\s+@([A-Za-z0-9_.$-]+)",
                selected_ir.read_text(),
            )))
        matcher_launches = int(
            composition.get("matcher_launches", row.get("launches", "0")) or 0
        )
        composed_launches = int(
            composition.get("composed_launches", matcher_launches) or 0
        )
        network_launches = int(composition.get("network_launches", "0") or 0)
        matched_implementations = ", ".join(
            f"<code>@{html.escape(symbol)}</code>"
            for symbol in matched_symbols
        ) or "—"
        comparison = comparison_rows.get(function)
        if (
            comparison
            and network_launches
            and function != "mfem_app_abs_l1_mass_3d"
            and comparison.get("comparison_quality") != "COMPOSED_CORRECTNESS_FAIL"
        ):
            comparison = dict(comparison)
            comparison["comparison_quality"] = "PRE_COMPOSITION_BASELINE"
            comparison["comparison_scope"] = (
                "runtime predates cuTensorNet network composition; structural "
                "before/after launch counts are current, performance rerun pending"
            )
        silicon = MFEM_APPLICATION_JETSON_RUNS.get(function)
        if comparison:
            raised_us = comparison.get("raised_runtime_us", "")
            native_us = comparison.get("mfem_native_runtime_us", "")
            ratio = comparison.get("raised_over_native", "")
            runtime_parts = []
            if raised_us:
                runtime_parts.append(f"raised {float(raised_us) / 1000.0:.6f} ms")
            if native_us:
                runtime_parts.append(
                    f"MFEM native {float(native_us) / 1000.0:.6f} ms"
                )
            if ratio:
                runtime_parts.append(f"raised/native {ratio}x")
            correctness = comparison.get("correctness", "NOT RUN")
            outcome = (
                "CORRECTNESS PASS" if correctness == "PASS"
                else "CORRECTNESS FAIL"
            )
            silicon = {
                "outcome": outcome,
                "correctness": comparison.get("comparison_scope", ""),
                "runtime": "; ".join(runtime_parts) or "timing withheld",
                "calls": (
                    f'{composed_launches} post-composition launches; '
                    f'native components: {comparison.get("native_components", "—")}'
                ),
                "params": (
                    f'NE={comparison.get("ne", "—")}; '
                    f'D1D={comparison.get("d1d", "—")}; '
                    f'Q1D={comparison.get("q1d", "—")}; '
                    f'raised iterations={comparison.get("raised_iterations", "—")}; '
                    f'native iterations={comparison.get("native_iterations", "—") or "n/a"}; '
                    f'{comparison.get("measurement_statistic", "")}; '
                    f'comparison={comparison.get("comparison_quality", "—")}'
                ),
                "hardware": comparison.get("hardware", ""),
                "test_label": "large-problem raised/native comparison",
            }
        if silicon:
            silicon_class = (
                "pass" if "PASS" in silicon["outcome"] else "partial"
            )
            test_label = silicon.get("test_label", "Jetson silicon test")
            hardware = silicon.get(
                "hardware",
                "attached Jetson tegra-ubuntu, MAXN, CUDA 12.6, cuTensorNet",
            )
            silicon_html = (
                f'<br><b>{html.escape(test_label)}:</b> '
                f'<span class="{silicon_class}">{html.escape(silicon["outcome"])}</span>; '
                f'{html.escape(silicon["correctness"])}<br>'
                f'<b>Runtime:</b> {html.escape(silicon["runtime"])}<br>'
                f'<b>Per-launch diagnostics:</b> {html.escape(silicon["calls"])}<br>'
                f'<b>Test parameters:</b> {html.escape(silicon["params"])}<br>'
                f'<b>Hardware:</b> {html.escape(hardware)}'
            )
        else:
            silicon_html = (
                '<br><b>Jetson silicon test:</b> <span class="partial">NOT RUN</span>; '
                'no runtime measurement'
            )
        coverage_class = "pass" if missing == "none" else "partial"
        status_class = "pass" if loops == 0 else "partial"
        ce_url = ce_link_from_paths(source if source.exists() else None, frontend)
        ce_link = (
            f'<a href="{ce_url}" target="_blank" style="margin-left:12px; '
            'color:#0366d6;">open in Compiler Explorer →</a>'
        ) if ce_url else ""
        summary_html = (
            '<div class="summary" style="padding:10px 20px; '
            'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
            f'<b>Application:</b> {html.escape(row["application"])} &nbsp;·&nbsp; '
            f'<b>coverage:</b> <span class="{coverage_class}">'
            f'{html.escape(row["coverage"])}</span> &nbsp;·&nbsp; '
            f'<b>{html.escape(row["linalg_ops"])}</b> Linalg op(s) &nbsp;·&nbsp; '
            f'<b class="{status_class}">{html.escape(row["residual_loops"])}</b> '
            f'residual loop(s) &nbsp;·&nbsp; '
            f'<b>{html.escape(row["matched_groups"])}</b> semantic match group(s) '
            f'&nbsp;·&nbsp; <b>{matcher_launches}→{composed_launches}</b> launches '
            f'after composition &nbsp;·&nbsp; <b>{network_launches}</b> composed '
            'cuTensorNet network(s)<br>'
            f'<b>Matched implementations:</b> {matched_implementations}<br>'
            f'<b>Extracted operator families:</b> {html.escape(families)}<br>'
            f'<b>Missing operator families:</b> {html.escape(missing)}<br>'
            f'<b>Extracted C:</b> <a href="{html.escape(c_page_filename)}">'
            f'<code>{html.escape(row["source"])}</code></a> &nbsp;·&nbsp; '
            f'<b>function:</b> <code>{html.escape(function)}</code><br>'
            f'<b>Upstream:</b> <a href="{html.escape(upstream_url)}" target="_blank">'
            f'<code>{html.escape(local_pointer)}</code></a> &nbsp;·&nbsp; '
            f'<b>MFEM commit:</b> <code>{MFEM_UPSTREAM_COMMIT}</code>'
            f'{silicon_html}'
            '</div>'
        )
        header = (
            '<div class="header"><h1><a href="mfem.html">← MFEM</a> '
            f'&nbsp; {html.escape(function)}{ce_link}</h1></div>'
        )
        OUTPUT_DIR.joinpath(page_filename).write_text(
            render_html(
                f"MFEM benchmark: {function}",
                header + summary_html + "\n".join(blocks),
                css,
            )
        )
        stats.append({
            **row,
            "page_filename": page_filename,
            "c_page_filename": c_page_filename,
            "upstream_url": upstream_url,
            "upstream_pointer": local_pointer,
            "linalg_ops_int": int(row.get("linalg_ops", "0") or 0),
            "residual_loops_int": loops,
            "matches_int": int(row.get("matched_groups", "0") or 0),
            "launches_int": composed_launches,
            "matcher_launches_int": matcher_launches,
            "network_launches_int": network_launches,
            "matched_symbols": matched_symbols,
            "silicon": silicon,
            "comparison": comparison,
        })
    return stats


def _mfem_application_extraction_section(stats: list[dict]) -> str:
    rows = []
    for row in stats:
        missing = row.get("missing_operator", "") or "none"
        coverage_class = "pass" if missing == "none" else "partial"
        raised_class = "pass" if row["residual_loops_int"] == 0 else "partial"
        name = (
            f'<a class="kernel" href="{html.escape(row["page_filename"])}">'
            f'{html.escape(row["function"])}</a>'
        )
        extracted_c = (
            f'<a href="{html.escape(row["c_page_filename"])}"><code>'
            f'{html.escape(row["source"])}:{html.escape(row["function"])}</code></a>'
        )
        upstream = (
            f'<a href="{html.escape(row["upstream_url"])}" target="_blank">'
            f'<code>{html.escape(row["upstream_pointer"])}</code></a>'
        )
        matched_implementations = ", ".join(
            f"<code>@{html.escape(symbol)}</code>"
            for symbol in row["matched_symbols"]
        ) or "—"
        comparison = row.get("comparison")
        if comparison:
            correctness = comparison.get("correctness", "NOT RUN")
            silicon_class = "pass" if correctness == "PASS" else "fail"
            silicon_outcome = (
                f'<span class="{silicon_class}">{html.escape(correctness)}</span>'
            )
            raised_us = comparison.get("raised_runtime_us", "")
            native_us = comparison.get("mfem_native_runtime_us", "")
            ratio = comparison.get("raised_over_native", "")
            raised_runtime = (
                f'{float(raised_us) / 1000.0:.6f} ms' if raised_us else "—"
            )
            native_runtime = (
                f'{float(native_us) / 1000.0:.6f} ms' if native_us else "—"
            )
            ratio_text = f'{html.escape(ratio)}x' if ratio else "—"
            quality = comparison.get("comparison_quality", "—")
            quality_class = (
                "pass" if quality == "EXACT_OPERATOR" else "partial"
            )
            comparison_scope = (
                f'<span class="{quality_class}">{html.escape(quality)}</span><br>'
                f'<small>{html.escape(comparison.get("comparison_scope", ""))}</small>'
            )
            silicon_params = (
                f'NE={html.escape(comparison.get("ne", "—"))}; '
                f'D1D={html.escape(comparison.get("d1d", "—"))}; '
                f'Q1D={html.escape(comparison.get("q1d", "—"))}; '
                f'raised iterations={html.escape(comparison.get("raised_iterations", "—"))}; '
                f'native iterations={html.escape(comparison.get("native_iterations", "") or "n/a")}'
            )
        else:
            silicon_outcome = '<span class="partial">NOT RUN</span>'
            raised_runtime = "—"
            native_runtime = "—"
            ratio_text = "—"
            comparison_scope = "—"
            silicon_params = "—"
        rows.append(
            f'<tr><td>{name}</td><td>{extracted_c}</td>'
            f'<td>{html.escape(row["application"])}</td>'
            f'<td class="{coverage_class}">{html.escape(row["coverage"])}</td>'
            f'<td>{upstream}</td><td>{row["linalg_ops_int"]}</td>'
            f'<td class="{raised_class}">{row["residual_loops_int"]}</td>'
            f'<td>{row["matches_int"]}</td>'
            f'<td>{row["matcher_launches_int"]}→{row["launches_int"]}<br>'
            f'<small>{row["network_launches_int"]} network(s)</small></td>'
            f'<td>{matched_implementations}</td>'
            f'<td>{silicon_outcome}</td><td>{raised_runtime}</td>'
            f'<td>{native_runtime}</td><td>{ratio_text}</td>'
            f'<td>{comparison_scope}</td><td>{silicon_params}</td></tr>'
        )
    total_linalg = sum(row["linalg_ops_int"] for row in stats)
    total_matches = sum(row["matches_int"] for row in stats)
    loop_free = sum(row["residual_loops_int"] == 0 for row in stats)
    applications = len({row["application"] for row in stats})
    return (
        '<div class="section-header"><h2 class="section-title">'
        'Larger MFEM application C extractions</h2></div>'
        '<div class="intro">'
        f'<b>{len(stats)} concrete operator paths</b> from <b>{applications} larger '
        f'applications</b>. All passed cgeist and raising; {loop_free}/{len(stats)} '
        f'are loop-free, producing {total_linalg} Linalg operations and '
        f'{total_matches} semantic library matches. These are numerical hot paths, '
        'not translations of MPI, mesh, or solver-control code. All rows were rebuilt '
        'at <b>NE=1024, D1D=4, Q1D=5</b> and run on the Jetson Orin in MAXN mode. '
        'Ten paths pass correctness; the minimal-surface path fails at this larger '
        'size and is intentionally not timed. Raised values are medians of five '
        'process-level means, with five timed application calls per process. '
        '<b>EXACT_OPERATOR</b> is a directly paired native MFEM '
        'CUDA operator. <b>COMPONENT_SUM</b> sums separately measured resident MFEM '
        'CUDA PA kernels and is a conservative component baseline, not a fused '
        'whole-application timing. PARTIAL_COMPONENT_SUM omits the ex9 PCG algebra. '
        'UNAVAILABLE means this MFEM revision exposes no equivalent native CUDA '
        'microbenchmark path.'
        '</div>'
        '<table><thead><tr><th>extracted entry</th><th>extracted C</th>'
        '<th>application</th>'
        '<th>coverage</th><th>upstream MFEM call site</th><th>Linalg ops</th>'
        '<th>residual loops</th><th>matches</th>'
        '<th>launches before→after composition</th>'
        '<th>matched implementation</th><th>correctness</th>'
        '<th>raised warm</th><th>MFEM native CUDA</th><th>raised/native</th>'
        '<th>comparison scope</th><th>test parameters</th></tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def _mfem_application_section(app_stats: list[dict]) -> str:
    rows = []
    for stats in app_stats:
        status = stats["raised_status"]
        status_class = "pass" if status == "VALIDATED" else "partial"
        name = (
            f'<a class="kernel" href="{html.escape(stats["page_filename"])}">'
            f'{html.escape(stats["id"])}</a>'
        )
        rows.append(
            f'<tr><td>{name}</td>'
            f'<td>{html.escape(stats["application"])}</td>'
            f'<td>{html.escape(stats["operator"])}</td>'
            f'<td>{html.escape(stats["dimension"])}D</td>'
            f'<td>{html.escape(stats["max_error"])}</td>'
            f'<td>{html.escape(stats["reference_us"])}</td>'
            f'<td>{html.escape(stats["sliced_us"])}</td>'
            f'<td>{html.escape(stats["speedup"])}x</td>'
            f'<td>{html.escape(stats["composition_launches"])}<br>'
            f'<small>{html.escape(stats["network_launches"])} network(s)</small></td>'
            f'<td class="{status_class}">{html.escape(status)}</td>'
            f'<td>{html.escape(stats["blocker"])}</td></tr>'
        )
    return (
        '<div class="section-header"><h2 class="section-title">'
        'MFEM application hot-operator ports</h2></div>'
        '<div class="intro">'
        f'<b>{len(app_stats)} application operators</b> from MFEM examples '
        '1, 3, 4, and 9. CPU timings compare faithful extracted C with the '
        'equivalent stage-sliced C over 10,000 warmed two-element applies. '
        '<b>These CPU speedups are normalization results, not GPU/library '
        'speedups.</b> Library-backed status is shown separately and silicon '
        'execution remains gated on end-to-end correctness.'
        '</div>'
        '<table><thead><tr><th>port</th><th>application</th><th>operator</th>'
        '<th>dim</th><th>max error</th><th>faithful C (us)</th>'
        '<th>stage-sliced C (us)</th><th>CPU speedup</th>'
        '<th>launches before→after composition</th><th>library-backed status</th>'
        '<th>blocker</th></tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def _mfem_section(mfem_stats: list[dict]) -> str:
    rows = []
    for stats in mfem_stats:
        ident = html.escape(stats["id"])
        page = html.escape(stats["page_filename"])
        name = f'<a class="kernel" href="{page}">{ident}</a>'
        c_page = html.escape(stats["c_page_filename"])
        extracted_c = (
            f'<a href="{c_page}"><code>'
            f'{html.escape(stats["source"])}:{html.escape(stats["function"])}</code></a>'
        )
        upstream = (
            f'<a href="{html.escape(stats["upstream_url"])}" target="_blank">'
            f'<code>{html.escape(stats["upstream_pointer"])}</code></a>'
        )
        variant = stats["variant"]
        launches = stats["launches"]
        matcher_launches = stats["matcher_launches"]
        network_launches = stats["network_launches"]
        if variant == "original":
            status_class, status = "partial", "RESIDUAL"
        elif launches:
            status_class, status = "pass", "EXECUTABLE"
        else:
            status_class, status = "pass", "RAISED"
        symbols = ", ".join(
            f"<code>@{html.escape(symbol)}</code>"
            for symbol in stats["matched_symbols"]
        ) or "—"
        silicon = stats.get("silicon", {})
        if silicon:
            correctness = html.escape(silicon["correctness"])
            correctness_class = "pass" if correctness == "PASS" else "none"
            raised_value = silicon.get("raised_runtime_us", "")
            native_value = silicon.get("mfem_native_runtime_us", "")
            ratio_value = silicon.get("raised_over_native", "")
            raised_runtime = (
                _fmt_seconds(float(raised_value) / 1.0e6) if raised_value else "—"
            )
            if correctness != "PASS" and raised_value:
                raised_runtime += " (invalid result)"
            elif network_launches and stats["id"] != "mass_apply_3d_stage_sliced":
                raised_runtime += " (pre-composition)"
            native_runtime = (
                _fmt_seconds(float(native_value) / 1.0e6) if native_value else "—"
            )
            if correctness != "PASS":
                ratio_cell = "withheld (incorrect result)"
            elif ratio_value:
                ratio = float(ratio_value)
                ratio_cell = f'MFEM <b>{ratio:.1f}&times;</b> faster'
            else:
                ratio_cell = "—"
        else:
            correctness = raised_runtime = native_runtime = ratio_cell = "—"
            correctness_class = ""
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{extracted_c}</td><td>{upstream}</td>"
            f"<td>{html.escape(stats['family'])}</td>"
            f"<td>{html.escape(stats['dimension'])}D</td>"
            f"<td>{html.escape(variant)}</td>"
            f"<td>{stats['linalg_ops']}</td>"
            f"<td>{stats['residual_loops']}</td>"
            f"<td>{matcher_launches}→{launches}<br>"
            f"<small>{network_launches} network(s)</small></td>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{symbols}</td>"
            f'<td class="{correctness_class}">{correctness}</td>'
            f"<td>{raised_runtime}</td>"
            f"<td>{native_runtime}</td>"
            f"<td>{ratio_cell}</td></tr>"
        )

    originals = [row for row in mfem_stats if row["variant"] == "original"]
    normalized = [row for row in mfem_stats if row["variant"] == "normalized"]
    total_linalg = sum(row["linalg_ops"] for row in mfem_stats)
    fully_raised = sum(row["fully_raised"] for row in mfem_stats)
    matched_kernels = sum(row["launches"] > 0 for row in normalized)
    total_matches = sum(row["launches"] for row in normalized)
    total_matcher_launches = sum(row["matcher_launches"] for row in normalized)
    total_networks = sum(row["network_launches"] for row in normalized)
    return (
        '<a name="mfem"></a>'
        '<div class="section-header"><h2 class="section-title">'
        'MFEM finite-element kernels</h2></div>'
        '<div class="intro">'
        f'<b>{len(mfem_stats)} tracked kernels:</b> {len(originals)} faithful '
        f'originals and {len(normalized)} structurally normalized equivalents. '
        f'The raising pipeline produced {total_linalg} Linalg operations; '
        f'{fully_raised}/{len(mfem_stats)} kernels are loop-free, including '
        f'all {len(normalized)}/{len(normalized)} normalized variants. '
        f'The matcher emitted {total_matcher_launches} ABI-lowerable library '
        f'launches; network composition leaves {total_matches} launches, '
        f'including {total_networks} multi-contraction cuTensorNet networks, '
        f'across {matched_kernels}/{len(normalized)} normalized kernels. '
        'Each row links to the '
        'stored frontend, raised, debufferized, matcher-rewritten, and composed '
        'IR plus '
        'a Compiler Explorer deep link.'
        '<br><b>Silicon comparison:</b> all 18 matcher-covered normalized '
        'kernels (ten PA operators plus eight DFEM interpolation/integration '
        'maps) use f64, NE=1024, D1D=4, Q1D=5, and 20 timed calls on '
        'Jetson Orin sm_87 in MAXN mode with CUDA 12.6. The raised value is '
        'the median of five process trials; each trial warms three calls and '
        'reports the best of 20 individually synchronized calls. '
        '“Raised” is the current zero-copy mapped host-pointer ABI; prepared '
        'cuTensorNet plans, workspaces, and scratch are reused, while any '
        'residual host loops remain included. “MFEM CUDA” uses the same warmup, '
        'best-of-20, and synchronization policy on resident device buffers. '
        'This intentionally records the '
        'performance gap in the current end-to-end lowering and is not a '
        'kernel-only claim.'
        '</div>'
        '<table><thead><tr><th>kernel</th><th>extracted C</th>'
        '<th>upstream MFEM source</th><th>family</th><th>dim</th>'
        '<th>variant</th><th>Linalg ops</th><th>residual loops</th>'
        '<th>launches before→after composition</th><th>status</th>'
        '<th>matched implementation</th>'
        '<th>silicon correctness</th><th>raised current ABI</th>'
        '<th>MFEM native CUDA</th><th>runtime difference</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


# Map blocker tag to a CSS class so the table cell can be colour-coded.
# "FIXABLE" categories (scratch-carry, indirect-index, mixed-reductions,
# matcher-gap, debuf-bug) -> partial (yellow). Fundamental blockers
# (serial-recurrence, t-loop, non-affine, cgeist-frontend) -> none (red).
# "none" -> pass (green).
_BLOCKER_CSS = {
    "none":              "pass",
    "matcher-gap":       "partial",
    "runtime-gap":       "partial",
    "scratch-carry":     "partial",
    "indirect-index":    "partial",
    "mixed-reductions":  "partial",
    "debuf-bug":         "partial",
    "t-loop":            "none",
    "serial-recurrence": "none",
    "non-affine":        "none",
    "cgeist-frontend":   "none",
    "raise-fail":        "none",
    "raise-crash":       "none",
    "no-linalg":         "none",
    "ext-math-call":     "partial",
    # Pipeline is correct; the gap is downstream (library / frontend). Mark
    # as "partial" — matcher / lowering still validate end-to-end.
    "cudnn-dtype-gap":   "partial",
    "cgeist-dtype-gap":  "partial",
    "partial-pipeline":  "partial",
}


def _fmt_seconds(s: float) -> str:
    """Format a seconds value for display in the runtime cells:
    sub-millisecond → µs, sub-second → ms, otherwise s."""
    if s < 0.001:
        return f"{s*1e6:.1f} µs"
    if s < 1.0:
        return f"{s*1000:.2f} ms"
    return f"{s:.2f} s"


def _runtime_cells_for(kernel: str, runtimes: dict[str, list[dict]] | None) -> list[str]:
    """One <td> block per runtime entry.
    Empty list if no runtime comparison exists for this kernel; the caller
    emits empty placeholders for all five runtime cells. PolyBench entries use
    raised_ms/pbgpu_ms and get an automatic speed comparison. Other sections
    can pass preformatted raised/reference/winner strings.
    """
    entries = (runtimes or {}).get(kernel, [])
    cells_per_row = []
    for e in entries:
        size = e["size"]
        if "raised_ms" in e and "pbgpu_ms" in e:
            raised_s = e["raised_ms"] / 1000.0
            pbgpu_s = e["pbgpu_ms"] / 1000.0
            raised_cell = _fmt_seconds(raised_s)
            reference_cell = _fmt_seconds(pbgpu_s)
            raised_speedup = pbgpu_s / raised_s if raised_s > 0 else 0.0
            if raised_speedup >= 1.10:
                su_cls = "pass"
                winner = f'raised {raised_speedup:.2f}&times;'
            elif raised_speedup >= 0.90:
                su_cls = "partial"
                if raised_speedup >= 1.0:
                    winner = f'raised {raised_speedup:.2f}&times;'
                else:
                    winner = f'PBGPU {1.0 / raised_speedup:.2f}&times;'
            else:
                su_cls = "none"
                winner = f'PBGPU {1.0 / raised_speedup:.2f}&times;'
        else:
            raised_cell = e.get("raised", "—")
            reference_cell = e.get("reference", "—")
            winner = e.get("winner", "—")
            su_cls = e.get("winner_class")
            if not su_cls:
                if winner.startswith("raised"):
                    su_cls = "pass"
                elif winner in ("n/a", "—", "raised-only"):
                    su_cls = "partial"
                else:
                    su_cls = "none"
        note = e.get("notes", "") or ""
        note_html = (f'<td style="font-size:11px; color:#555; max-width:340px">'
                     f'{note}</td>' if note else
                     '<td style="font-size:11px"></td>')
        cells_per_row.append(
            f'<td style="font-size:12px"><b>{size}</b></td>'
            f'<td style="font-size:12px; text-align:right">{raised_cell}</td>'
            f'<td style="font-size:12px; text-align:right">{reference_cell}</td>'
            f'<td class="{su_cls}" style="font-size:12px; text-align:right">'
            f'{winner}</td>'
            + note_html
        )
    return cells_per_row


def _render_section_rows(kernel_stats: dict[str, dict],
                          notes: dict[str, tuple[str, str]],
                          blockers: dict[str, tuple[str, str]],
                          runtimes: dict[str, list[dict]] | None = None,
                          display_names: dict[str, str] | None = None,
                          order: list[str] | None = None) -> str:
    rows = []
    if order:
        ordered = [k for k in order if k in kernel_stats]
        ordered += sorted(k for k in kernel_stats if k not in set(order))
    else:
        ordered = sorted(kernel_stats)
    for k in ordered:
        s = kernel_stats[k]
        page_file = s.get("page_filename", f"{k}.html")
        l = s["launches"]; r = s["residual"]; f = s["residual_for"]
        matched_f = s.get("matched_residual_for", f)
        if l > 0 and r == 0 and matched_f == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        for_cls = "none" if f > 0 else "pass"

        label = (display_names or {}).get(k, k)
        if page_file:
            kernel_link = f'<a class="kernel" href="{page_file}">{label}</a>'
        elif s.get("ce_suppressed"):
            kernel_link = f'<span class="nope">{label} (IR only)</span>'
        else:
            kernel_link = f'<span class="nope">{label} (no source)</span>'

        note_tag, note_blurb = notes.get(k, ("", ""))
        tag_cls = {
            "highly parallel":   "pass",
            "parallel + T loop": "partial",
            "partial parallel":  "partial",
            "serial":            "none",
        }.get(note_tag, "")
        note_cell = (
            f'<td class="{tag_cls}" style="white-space:nowrap"><b>{note_tag}</b></td>'
            f'<td style="font-size:12px; color:#555">{note_blurb}</td>'
            if note_tag else '<td></td><td></td>'
        )

        block_tag, block_blurb = blockers.get(k, ("none", ""))
        block_label = BLOCKER_TAXONOMY.get(block_tag, ("", ""))[0]
        block_cls = _BLOCKER_CSS.get(block_tag, "")
        if block_tag == "none":
            block_cell = (
                '<td class="pass" style="white-space:nowrap; font-size:12px">—</td>'
                '<td style="font-size:12px; color:#555"></td>'
            )
        else:
            block_cell = (
                f'<td class="{block_cls}" style="white-space:nowrap; font-size:12px">'
                f'<a href="index.html#taxonomy" style="color:inherit; text-decoration:none">'
                f'<b>{block_label}</b></a></td>'
                f'<td style="font-size:12px; color:#555">{block_blurb}</td>'
            )

        kernel_cell = f'<td>{kernel_link}</td>'
        match_cells = (
            f'<td>{l}</td><td>{r}</td><td class="{for_cls}">{f}</td>'
            f'<td class="{cls}">{status}</td>'
        )
        if l == 0:
            backend_cells = (
                '<td style="color:#999">—</td><td style="color:#999">generic loops</td>'
                '<td style="color:#999">—</td>'
            )
        elif not s.get("abi_lowered"):
            backend_cells = (
                '<td class="none">GAP</td><td class="none">blocked</td>'
                '<td class="none">blocked</td>'
            )
        else:
            abi_cls = "pass"
            cpu_cls = "pass" if s.get("cpu_backend") else "partial"
            cuda_cls = "pass" if s.get("cuda_backend") else "partial"
            abi_label = "READY"
            abi_calls = s.get("abi_calls", [])
            n_cblas = len(s.get("cpu_cblas_calls", []))
            if s.get("cpu_backend"):
                cpu_label = "CBLAS" if abi_calls and n_cblas == len(abi_calls) else "reference C"
            else:
                cpu_label = "incomplete"
            cuda_label = "CUDA libs" if s.get("cuda_backend") else "incomplete"
            backend_cells = (
                f'<td class="{abi_cls}">{abi_label}</td>'
                f'<td class="{cpu_cls}">{cpu_label}</td>'
                f'<td class="{cuda_cls}">{cuda_label}</td>'
            )

        # Jetson-runtime cells: one <tr> per warmed comparison entry when data
        # exists; otherwise one <tr> with five empty runtime cells.
        runtime_rows = _runtime_cells_for(k, runtimes)
        if not runtime_rows:
            runtime_rows = ['<td style="font-size:12px; color:#bbb">—</td>'
                            '<td style="font-size:12px; color:#bbb">—</td>'
                            '<td style="font-size:12px; color:#bbb">—</td>'
                            '<td style="font-size:12px; color:#bbb">—</td>'
                            '<td style="font-size:12px; color:#bbb">—</td>']

        # Multi-row layout: the kernel-shared cells (name, match-status,
        # parallelism, blocker) use rowspan to span all the runtime rows
        # for this kernel. The first runtime row joins them; the rest are
        # standalone <tr>s with only the four runtime cells.
        n_rows = len(runtime_rows)
        rowspan_attr = f' rowspan="{n_rows}"' if n_rows > 1 else ''

        # Re-apply rowspan to each <td> in kernel_cell / match_cells /
        # note_cell / block_cell. We need to inject rowspan into each
        # opening <td>. Simplest: substitute via string ops.
        def _with_rowspan(html: str) -> str:
            # Only adds rowspan to <td> tags (not </td>); used when n_rows>1.
            if n_rows <= 1:
                return html
            # Replace each `<td` (with or without attrs) with `<td rowspan="N"`.
            # Idempotent enough for our generated strings.
            return re.sub(r'<td(\s|>)', f'<td rowspan="{n_rows}"\\1', html)

        first_kernel  = _with_rowspan(kernel_cell)
        first_match   = _with_rowspan(match_cells)
        first_backend = _with_rowspan(backend_cells)
        first_note    = _with_rowspan(note_cell)
        first_block   = _with_rowspan(block_cell)

        rows.append(
            f'<tr>{first_kernel}{first_match}{first_backend}{first_note}{first_block}'
            f'{runtime_rows[0]}</tr>'
        )
        for rr in runtime_rows[1:]:
            rows.append(f'<tr>{rr}</tr>')
    return "\n".join(rows)


def _build_section(title: str, anchor: str, blurb: str,
                    kernel_stats: dict[str, dict],
                    notes: dict[str, tuple[str, str]],
                    blockers: dict[str, tuple[str, str]],
                    extra_html: str = "",
                    runtimes: dict[str, list[dict]] | None = None,
                    display_names: dict[str, str] | None = None,
                    order: list[str] | None = None,
                    runtime_headers: tuple[str, str, str, str, str] = (
                        "Jetson<br>case",
                        "Raised pipeline<br>(rt-gpu)",
                        "PolyBenchGPU<br>CUDA",
                        "winner<br>speed",
                        "notes",
                    )) -> str:
    """Render one benchmark-suite section: a section header, blurb, then table."""
    rows_html = _render_section_rows(
        kernel_stats, notes, blockers,
        runtimes=runtimes,
        display_names=display_names,
        order=order,
    )
    case_h, raised_h, reference_h, winner_h, notes_h = runtime_headers
    return (
        f'<a name="{anchor}"></a>'
        f'<div class="section-header"><h2 class="section-title">{title}</h2></div>'
        f'<div class="intro">{blurb}</div>'
        + extra_html +
        '<table><thead><tr>'
        '<th>kernel</th><th>kernel.launches</th>'
        '<th>residual linalg.generic</th>'
        '<th>residual for-loops</th>'
        '<th>match status</th>'
        '<th>shared ABI</th><th>CPU/C backend</th><th>GPU backend</th>'
        '<th>parallelism</th>'
        '<th>parallelism notes</th>'
        '<th>blocker</th>'
        '<th>blocker notes</th>'
        f'<th>{case_h}</th>'
        f'<th>{raised_h}</th>'
        f'<th>{reference_h}</th>'
        f'<th>{winner_h}</th>'
        f'<th>{notes_h}</th>'
        '</tr></thead><tbody>'
        + rows_html +
        '</tbody></table>'
    )


def _llama2c_runtime_summary() -> str:
    """Render the Llama numbers as a visible section-local table.

    The shared runtime columns compare PolyBench rows against PolyBenchGPU, so
    Llama gets its own table with the appropriate comparison target.
    """
    return (
        '<div class="intro" style="padding-top:0">'
        '<b>Latest Jetson Llama runtime numbers</b>'
        '</div>'
        '<table style="margin-top:4px"><thead><tr>'
        '<th>fixture</th>'
        '<th>coverage</th>'
        '<th>raised device time</th>'
        '<th>comparison</th>'
        '<th>host-visible time</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        '<tr>'
        '<td><b>N=1024, H=4096 forward tensor path</b></td>'
        '<td>RMSNorm + zero-fill + SGEMV + softmax</td>'
        '<td>RMSNorm ~0.09-0.10 ms<br>'
        'SGEMV ~0.53-0.55 ms<br>'
        'softmax ~0.028-0.030 ms</td>'
        '<td>validated against native C output</td>'
        '<td>not the headline metric</td>'
        '<td>warm timings after first-use setup; RMSNorm uses cuDNN backend '
        'graph at this size</td>'
        '</tr>'
        '<tr>'
        '<td><b>N=2048, H=32000 logits suffix</b></td>'
        '<td>RMSNorm + scale + output projection GEMV</td>'
        '<td>raised device-only median 1.614 ms</td>'
        '<td>ggml/llama.cpp CUDA median 1.494 ms</td>'
        '<td>raised median 1.652 ms after RMSNorm plan caching</td>'
        '<td>remaining gap is mostly SGEMV/output projection plus separate '
        'shim overhead</td>'
        '</tr>'
        '<tr>'
        '<td><b>standalone Llama op sweep</b></td>'
        '<td>17 raised standalone ops, MODEL_DIM=64, FFN_DIM=128, '
        'SEQ_LEN=32, VOCAB=256</td>'
        '<td>one-layer sum 0.575 ms device median<br>'
        'embedding + one layer + final RMSNorm + lm_head 0.662 ms</td>'
        '<td>runtime-shim warm timings, first 5 of 50 iterations discarded</td>'
        '<td>one-layer sum 0.832 ms host median<br>'
        'embedding + one layer + final RMSNorm + lm_head 0.955 ms</td>'
        '<td>covers split RoPE and branchless mask; interleaved RoPE and '
        'branchy mask still remain non-raised variants</td>'
        '</tr>'
        '</tbody></table>'
    )


def _llama_forward_runtime_summary() -> str:
    return (
        '<div class="intro" style="padding-top:0">'
        '<b>Exact one-token Llama fixture comparison</b>'
        '</div>'
        '<table style="margin-top:4px"><thead><tr>'
        '<th>implementation</th>'
        '<th>median of process medians</th>'
        '<th>correctness</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        '<tr>'
        '<td><b>native Orin CPU, strict -O3</b></td>'
        '<td>342.515 ms</td><td>numerical reference</td>'
        '<td>three processes; 2 warmup + 10 measured iterations each</td>'
        '</tr>'
        '<tr>'
        '<td><b>native Orin CPU, -ffast-math</b></td>'
        '<td>144.152 ms</td><td>informational; changes FP32 result</td>'
        '<td>reported separately from the strict reference</td>'
        '</tr><tr>'
        '<td><b>Polygeist external-library hybrid</b></td>'
        '<td>473.546 ms</td>'
        '<td>PASS: max abs 8.201e-4; atol=1e-3, rtol=1e-4</td>'
        '<td>13 CUDA/cuBLAS/cuTENSOR/cuDNN launches; 32 residual Linalg '
        'bodies execute on CPU. Project-authored mask/add/SwiGLU helpers are excluded.</td>'
        '</tr><tr>'
        '<td><b>ggml CUDA expert implementation</b></td>'
        '<td>15.954 ms</td>'
        '<td>PASS: max abs 4.5185e-3; atol=1e-2, rtol=1e-4</td>'
        '<td>ggml revision f24588a; identical FP32 fixture math</td>'
        '</tr>'
        '</tbody></table>'
        '<div class="intro"><b>Scope:</b> one token at position 1024, one '
        '7B-size layer (4096/11008/32000/2048, 32 heads). This is an extracted '
        'FP32 fixture—not full 32-layer inference or quantized GGUF execution. '
        'It uses split even/odd RoPE and branchless masking because the exact '
        'interleaved and branchy forms remain raising gaps. Authoritative data: '
        '<code>issues/llama_section42/performance.csv</code>.</div>'
    )


def _build_taxonomy_panel() -> str:
    """A top-of-page explainer for the per-kernel `blocker` column.
    Categories link from each row's blocker cell to the right entry here."""
    rows = []
    for tag, (label, longer) in BLOCKER_TAXONOMY.items():
        cls = _BLOCKER_CSS.get(tag, "")
        rows.append(
            f'<tr><td class="{cls}" style="white-space:nowrap; font-size:13px">'
            f'<b>{label}</b></td>'
            f'<td style="font-size:13px; color:#444">{longer}</td></tr>'
        )
    return (
        '<a name="taxonomy"></a>'
        '<div class="section-header" style="background:#f0e6ff; border-color:#c9b8e0">'
        '  <h2 class="section-title">Algorithm-blocker taxonomy</h2>'
        '</div>'
        '<div class="intro">'
        '  Each kernel below carries a <em>blocker</em> tag describing what '
        '  prevents it from lifting fully (or matching to a kernel.launch). '
        '  Green tags are wins (no blocker); yellow tags are <b>fixable</b> '
        '  gaps in our raise / matcher / debufferize passes; red tags are '
        '  <b>fundamental</b> — the algorithm has cross-iteration data '
        '  dependencies that no transformation can remove. Categories:'
        '</div>'
        '<table><thead><tr>'
        '<th>category</th><th>meaning</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
    )


# Polybench-style single-file CNN-block kernels extracted from darknet
# for the matcher+cuDNN-shim end-to-end work. Each kernel is its own
# `.c` in third_party/cnn-extracted/, with MINI/LARGE dataset macros
# and (for the multi-step ones) a chained body that exercises the
# matcher's longest-first composition library. See the section blurb
# for which library entry each kernel matches.
EXTRACTED_DARKNET_KERNELS: dict[str, tuple[str, str]] = {
    "conv2d_batched":      ("conv2d_batched.c",      "kernel_conv2d_batched"),
    "darknet_im2col_gemm": ("darknet_im2col_gemm.c", "kernel_darknet_im2col_gemm"),
    "maxpool_batched":     ("maxpool_batched.c",     "kernel_maxpool_batched"),
    "batchnorm_batched":   ("batchnorm_batched.c",   "kernel_batchnorm_batched"),
    "shortcut_batched":    ("shortcut_batched.c",    "kernel_shortcut_batched"),
    "conv_bn_relu_batched":("conv_bn_relu_batched.c","kernel_conv_bn_relu_batched"),
}

# Fusion-optimization kernels — algebraic rewrites that exploit specific
# patterns to route to faster cuBLAS / cublasLt / cuDNN entry points.
# Same .c source layout (third_party/cnn-extracted/) and bake pipeline
# as extracted_darknet, but a separate section in the IR explorer so
# the headline speedups are easy to spot.
FUSION_OPT_KERNELS: dict[str, tuple[str, str]] = {
    "conv_bias_relu_add_batched": ("conv_bias_relu_add_batched.c", "kernel_conv_bias_relu_add_batched"),
    "gemm_bias_relu":              ("gemm_bias_relu.c",              "kernel_gemm_bias_relu"),
    "ata_gemm":                    ("ata_gemm.c",                    "kernel_ata_gemm"),
    "conv1x1_batched":             ("conv1x1_batched.c",             "kernel_conv1x1_batched"),
}


EXTRACTED_DARKNET_RUNTIMES: dict[str, list[dict]] = {
    # Jetson Orin silicon runs (2026-05-25). All FP32 NCHW. The MINI
    # shapes are overhead-bound (cuDNN descriptor + workspace setup
    # dominates a sub-ms kernel). LARGE conv2d is where cuDNN's
    # tensor-core kernels shine — 23.8× over the CPU 3-loop reference.
    # batchnorm/shortcut LARGE remain bandwidth-bound and lose to the
    # CPU at single-call granularity; that's the well-known story for
    # standalone elementwise ops without device-residency hoisting.
    "conv2d_batched": [
        {"size": "MINI",  "shape": "B=4 IC=OC=8 H=W=32 K=3",
         "gpu_s": 0.084316, "cpu_s": 0.001871, "correct": "FP-noise",
         "notes": "Setup-bound: cuDNN descriptor + workspace + algo selection "
                  "≫ 28K-elem output; the 1.87 ms CPU 3-loop is just the math"},
        {"size": "LARGE", "shape": "B=32 IC=OC=64 H=W=56 K=3",
         "gpu_s": 0.137029, "cpu_s": 3.260427, "correct": "FP-noise",
         "notes": "ResNet conv2_x shape, tensor cores light up; 23.8× GPU win"},
    ],
    "maxpool_batched": [
        {"size": "MINI",  "shape": "B=4 C=8 H=W=32 K=S=2",
         "gpu_s": 0.012863, "cpu_s": 0.000057, "correct": "PASS",
         "notes": "Setup-bound; 8K output elems is trivial"},
        {"size": "LARGE", "shape": "B=32 C=64 H=W=112 K=3 S=2",
         "gpu_s": 0.023644, "cpu_s": 0.030398, "correct": "PASS",
         "notes": "ResNet stem maxpool; bandwidth-bound, cuDNN marginal win"},
    ],
    "batchnorm_batched": [
        {"size": "MINI",  "shape": "B=4 C=8 H=W=32",
         "gpu_s": 0.005291, "cpu_s": 0.000059, "correct": "FP-noise",
         "notes": "Setup-bound; 32K elems too small for cuDNN's BN to win"},
        {"size": "LARGE", "shape": "B=32 C=64 H=W=56",
         "gpu_s": 0.011313, "cpu_s": 0.004263, "correct": "FP-noise",
         "notes": "Bandwidth-bound elementwise; cuDNN BN setup overhead "
                  "doesn't amortize on a single call. Would need device-"
                  "residency to win"},
    ],
    "shortcut_batched": [
        {"size": "MINI",  "shape": "B=4 C=8 H=W=32",
         "gpu_s": 0.045177, "cpu_s": 0.000008, "correct": "PASS",
         "notes": "Setup-bound; cudnnAddTensor on 32K elems is pure overhead"},
        {"size": "LARGE", "shape": "B=32 C=64 H=W=56",
         "gpu_s": 0.049720, "cpu_s": 0.004171, "correct": "PASS",
         "notes": "Bandwidth-bound 2-buffer add; 6.4M float ops finish in "
                  "4ms on CPU. cuDNN AddTensor adds descriptor setup cost"},
    ],
    # Fused conv + bn + relu — the canonical ResNet inner pattern. The
    # matcher folds all four loop nests (init + conv + bn-inplace +
    # relu-inplace) into one launch. The runtime shim uses the standard
    # BN-folding trick (pre-multiply filter by scale*inv_std, adjust
    # bias) and issues a single cudnnConvolutionBiasActivationForward
    # call. Result: same wall-clock as conv2d_batched alone, but doing
    # all three ops — bn and relu effectively ride free on conv's
    # compute-bound win.
    "conv_bn_relu_batched": [
        {"size": "MINI",  "shape": "B=4 IC=OC=8 H=W=32 K=3",
         "gpu_s": 0.186320, "cpu_s": 0.002020, "correct": "PASS",
         "notes": "Setup-bound (the larger MINI gap vs conv2d alone is "
                  "the first-call init of cudnnConvolutionBiasActivation"
                  "Forward + a host BN-fold pass)"},
        {"size": "LARGE", "shape": "B=32 IC=OC=64 H=W=56 K=3",
         "gpu_s": 0.137820, "cpu_s": 3.243928, "correct": "FP-noise",
         "notes": "Same 23.5× as conv2d_batched alone, but doing 3 ops. "
                  "Fusion absorbs the bandwidth-bound bn+relu cost — they "
                  "become free in the conv's memory pass. Best argument "
                  "for cuDNN's fused-op API"},
    ],
}


# Silicon numbers for the four fusion-optimization kernels (Jetson Orin,
# 2026-05-25). All FP32. The "vs naive" column says what we'd be doing
# without the rewrite — e.g. running the standalone op chain through
# separate cuDNN launches, or routing K=1 conv through cuDNN's generic
# path, or computing AᵀA as a full gemm.
FUSION_OPT_RUNTIMES: dict[str, list[dict]] = {
    "conv_bias_relu_add_batched": [
        {"size": "MINI",  "shape": "B=4 IC=OC=8 H=W=32 K=3",
         "gpu_s": 0.121859, "cpu_s": 0.001943, "correct": "PASS",
         "notes": "Setup-bound (single-call init of cudnnConvolutionBias"
                  "ActivationForward); fused bias+add+relu shows here only "
                  "via the descriptor count, not via actual work"},
        {"size": "LARGE", "shape": "B=32 IC=OC=64 H=W=56 K=3",
         "gpu_s": 0.139847, "cpu_s": 3.253224, "correct": "FP-noise",
         "notes": "Same ~23.3× as conv2d_batched alone (137 ms) — bias + "
                  "residual-add + relu absorbed FREE into the conv's memory "
                  "pass. Closes the standalone shortcut-add GPU LOSS"},
    ],
    "gemm_bias_relu": [
        {"size": "MINI",  "shape": "M=N=K=64",
         "gpu_s": 0.075925, "cpu_s": 0.000201, "correct": "PASS",
         "notes": "Setup-bound (first-call init of cublasLtMatmul) "
                  "+ host BN-folding overhead"},
        {"size": "LARGE", "shape": "M=N=K=2048",
         "gpu_s": 0.056678, "cpu_s": 51.083039, "correct": "FP-noise",
         "notes": "cublasLt EPILOGUE_RELU_BIAS fires tensor cores; 901× "
                  "vs CPU 3-loop (which on 2048³ is brutally cache-unfriendly)"},
    ],
    "ata_gemm": [
        {"size": "MINI",  "shape": "M=K=64",
         "gpu_s": 0.003577, "cpu_s": 0.000203, "correct": "PASS",
         "notes": "Setup-bound; syrk's half-flops can't shine at this size"},
        {"size": "LARGE", "shape": "M=K=2048",
         "gpu_s": 0.019123, "cpu_s": 64.939412, "correct": "PASS",
         "notes": "cublasSsyrk does HALF the flops of an equivalent gemm "
                  "(only upper triangle of symmetric output). 3393× vs CPU."},
    ],
    "conv1x1_batched": [
        {"size": "MINI",  "shape": "B=4 IC=OC=16 H=W=32",
         "gpu_s": 0.045098, "cpu_s": 0.000796, "correct": "PASS",
         "notes": "Setup-bound; per-batch gemms are small"},
        {"size": "LARGE", "shape": "B=32 IC=OC=256 H=W=56",
         "gpu_s": 0.068130, "cpu_s": 7.132080, "correct": "PASS",
         "notes": "cublasSgemmStridedBatched on B=32 independent (256,3136)="
                  "(256,256)·(256,3136) gemms. 105× vs CPU 3-loop. Way "
                  "faster than cuDNN's generic K=1 conv path"},
    ],
}


# ------------------------------------------------------------------
# PVA backend — kernels lowered through --lower-kernel-launch-to-pva
# to NVIDIA PVA Solutions' libpva_operator on the Jetson Orin
# Programmable Vision Accelerator. PVA-only datapoints; no CPU compare.
# ------------------------------------------------------------------

PVA_KERNELS: list[dict] = [
    {
        "id": "conv2d_i8",
        "op": "OpConv2d",
        "vendor_call": "pvaConv2dCreate / pvaConv2dSubmit",
        "shim": "polygeist_pva_conv2d_3x3_i8",
        "matched": True,
        "build_dir": "/tmp/conv2d_jetson_i8_256",
        "timings": [("256×256", "33.3 ms"),
                    ("1024×1024", "33.7 ms"),
                    ("10240×10240", "216.3 ms")],
        "note": "Single-channel 3×3 9-tap signed conv from "
                "the extracted conv2d_i8 dtype source. Full matcher pipeline "
                "(cgeist → linalg → @cudnnConvolution2D_9tap_i8 → "
                "--lower-kernel-launch-to-pva).",
    },
    {
        "id": "conv2d_i16",
        "op": "OpConv2d",
        "vendor_call": "pvaConv2dCreate / pvaConv2dSubmit",
        "shim": "polygeist_pva_conv2d_3x3_i16",
        "matched": True,
        "build_dir": "/tmp/conv2d_jetson_i16_256",
        "timings": [("256×256", "33.5 ms"),
                    ("1024×1024", "34.8 ms"),
                    ("10240×10240", "372.9 ms")],
        "note": "Same shape as i8, 2-byte elements. PVA hardware applies "
                "Q16.16 fixed-point semantics to kernel coefficients.",
    },
    {
        "id": "boxfilter_i8",
        "op": "OpBoxFilter",
        "vendor_call": "pvaBoxFilterCreate / pvaBoxFilterSubmit",
        "shim": "polygeist_pva_boxfilter_3x3_i8",
        "matched": False,
        "build_dir": "/tmp/pva_boxfilter_i8_256",
        "timings": [("256×256", "40.4 ms")],
        "note": "Uniform 1/K² 3×3 mean filter — no coefficient tensor. "
                "Validated via hand-authored MLIR (matcher template for "
                "uniform-weight conv is not yet written).",
    },
    {
        "id": "gaussian_i8",
        "op": "OpGaussianFilter",
        "vendor_call": "pvaGaussianFilterCreate / pvaGaussianFilterSubmit",
        "shim": "polygeist_pva_gaussian_3x3_i8",
        "matched": False,
        "build_dir": "/tmp/pva_gaussian_i8_256",
        "timings": [("256×256", "32.6 ms")],
        "note": "σ=1, K=3 hardcoded in shim. PVA computes the discrete "
                "Gaussian kernel internally; matches canonical "
                "[1,2,1;2,4,2;1,2,1]/16. Hand-authored MLIR.",
    },
    {
        "id": "bilateral_i8",
        "op": "OpBilateralFilter",
        "vendor_call": "pvaBilateralFilterCreate / pvaBilateralFilterSubmit",
        "shim": "polygeist_pva_bilateral_3x3_i8",
        "matched": False,
        "build_dir": "/tmp/pva_bilateral_i8_256",
        "timings": [("256×256", "57.5 ms")],
        "note": "PVA Bilateral only accepts U8; shim reinterprets i8 bytes "
                "bitwise as U8 via make_pva_image_tensor_dtype. "
                "sigmaRange=25, sigmaSpace=10 hardcoded.",
    },
    {
        "id": "histeq_i8",
        "op": "OpHistogramEqualization",
        "vendor_call": "pvaHistogramEqualizationCreate / pvaHistogramEqualizationSubmit",
        "shim": "polygeist_pva_histeq_i8",
        "matched": False,
        "build_dir": "/tmp/pva_histeq_i8_256",
        "timings": [("256×256", "38.8 ms")],
        "note": "Pointwise 256-bin LUT (no spatial kernel). PVA computes "
                "the histogram + CDF + LUT internally. Hand-authored MLIR.",
    },
]


def _pva_section() -> str:
    """Polygeist → PVA Solutions kernels. Each row is a kernel we successfully
    lowered through --lower-kernel-launch-to-pva and ran on the Jetson Orin
    PVA accelerator. Timings are wall-clock from pva*Submit (full setup +
    submit + sync round-trip, single-shot). No CPU comparison here — PVA-only
    datapoints; the CPU stubs exist for separate per-op correctness validation."""
    rows = []
    for spec in PVA_KERNELS:
        first = True
        rowspan = len(spec["timings"]) or 1
        match_lbl = "matcher" if spec["matched"] else "hand-authored"
        match_cls = "pass" if spec["matched"] else "partial"
        for size, ms in (spec["timings"] or [("—", "—")]):
            if first:
                kernel_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top">'
                    f'<b>{spec["id"]}</b>'
                    f'<div style="font-size:11px; color:#666; margin-top:4px">'
                    f'frontend: <span class="{match_cls}"><b>{match_lbl}</b></span>'
                    f'</div></td>'
                )
                op_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top; '
                    f'font-size:12px">'
                    f'<b>{spec["op"]}</b><br>'
                    f'<span style="font-family:monospace; font-size:11px; '
                    f'color:#666">{spec["vendor_call"]}</span></td>'
                )
                shim_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top; '
                    f'font-family:monospace; font-size:11px">'
                    f'{spec["shim"]}</td>'
                )
                note_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top; '
                    f'font-size:11px; color:#555; max-width:340px">'
                    f'{spec["note"]}</td>'
                )
            else:
                kernel_cell = op_cell = shim_cell = note_cell = ""
            first = False
            rows.append(
                "<tr>"
                + kernel_cell + op_cell + shim_cell
                + f'<td style="font-size:12px"><b>{size}</b></td>'
                + f'<td style="font-size:12px; text-align:right; '
                  f'font-family:monospace"><b>{ms}</b></td>'
                + note_cell
                + "</tr>"
            )
    table = (
        '<table><thead><tr>'
        '<th>kernel</th><th>PVA op</th><th>runtime shim</th>'
        '<th>dataset</th><th>PVA wall-clock</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
    )
    return (
        '<div class="section-header" id="pva" '
        'style="background:#e4f3e4; border-color:#7faf8a">'
        '  <h2 class="section-title">PVA backend '
        '  (Polygeist → libpva_operator on Jetson Orin\'s Programmable '
        '   Vision Accelerator)</h2>'
        '</div>'
        '<div class="intro">'
        '  Kernels lowered through the new <code>--lower-kernel-launch-to-pva</code> '
        '  pass (see <code>lib/polygeist/Passes/LowerKernelLaunchToPVA.cpp</code>). '
        '  Each row is a kernel that successfully reaches PVA silicon via a '
        '  <code>func.call @polygeist_pva_*</code> emitted by the lowering pass and '
        '  resolved at link-time against the PVA shim in '
        '  <code>runtime/polygeist_pva_rt.c</code>, which wraps the corresponding '
        '  <code>pva*Create</code> / <code>pva*Submit</code> entrypoint in '
        '  <code>libpva_operator.so</code>.'
        '  <br><br>'
        '  Two kernels come through the full <em>matcher</em> pipeline today '
        '  (Conv2d i8 and i16, lifted from extracted dtype-specific conv2d sources). '
        '  The remaining four were validated via <em>hand-authored</em> kernel.launch '
        '  MLIR — the lowering + shim + silicon work, but matcher templates that '
        '  recognise their C-level patterns (uniform-weight conv, Gaussian-weighted '
        '  conv, bilateral, histogram-eq) have not been written yet.'
        '  <br><br>'
        '  <b>Per-call timing floor</b>: ~30&ndash;35 ms at any image size up to '
        '  ~1024², dominated by PVA allocator + <code>CupvaMemGetHostPointer</code> '
        '  + operator create/submit + cuPVA scheduling + stream sync. Compute is '
        '  sub-ms at these sizes. At 10240² (105M pixels) the per-call setup '
        '  amortises and PVA compute dominates.'
        '  <br><br>'
        '  No CPU comparison shown here; for bit-exact CPU/PVA diff validation '
        '  see the <code>scripts/correctness/pva_*_jetson.sh</code> test scaffolds '
        '  and the matching CPU stubs in '
        '  <code>runtime/polygeist_cublas_rt_cpu.c</code>.'
        '</div>'
        + table
        + '<div style="margin-top:14px; padding:10px 14px; '
          'background:#e4f3e4; border-left:4px solid #7faf8a;">'
          '  <b>What is <em>new</em> infrastructure</b> for this section:'
          '  <ul style="margin:6px 0 0 24px; padding:0; font-size:13px">'
          '  <li>New pass <code>LowerKernelLaunchToPVA</code> '
          '      (<code>lib/polygeist/Passes/LowerKernelLaunchToPVA.cpp</code>)</li>'
          '  <li>Shared 9-tap conv lowering helper extracted from the cuBLAS '
          '      pass into <code>KernelLaunchLoweringUtils.{h,cpp}</code>; '
          '      both passes call it. Added a parallel '
          '      <code>lowerImageFilter2Operand</code> helper for the 2-memref '
          '      filter shape (Box/Gaussian/Bilateral/HistogramEq).</li>'
          '  <li>PVA runtime shim <code>runtime/polygeist_pva_rt.c</code> with '
          '      a generic <code>make_pva_image_tensor_dtype</code> backbone, '
          '      <code>CupvaMemGetHostPointer</code>-mediated host I/O, '
          '      and one <code>pva&lt;Op&gt;Create</code> + '
          '      <code>pva&lt;Op&gt;Submit</code> wrapper per op.</li>'
          '  <li>Matching CPU reference stubs in '
          '      <code>runtime/polygeist_cublas_rt_cpu.c</code>, hand-modelled '
          '      to mirror PVA hardware semantics (centred anchor, REPLICATE '
          '      border, Q-shift, unsigned-kernel reinterpretation) so the '
          '      <code>conv2d_jetson</code> &harr; <code>conv2d_jetson_cpustub</code> '
          '      diff is bit-exact.</li>'
          '  <li>Cross-compile script <code>conv2d_cudnn_jetson_dtype.sh</code> '
          '      extended with an <code>i8</code> dtype branch + PVA-library '
          '      link line (<code>libpva_operator</code>, <code>libcvcuda</code>, '
          '      <code>libnvcv_types</code>, <code>libcupva_host</code>, plus '
          '      <code>libnvscibuf</code> / <code>libnvscisync</code> as '
          '      direct DT_NEEDEDs via <code>-Wl,--no-as-needed</code>).</li>'
          '  </ul>'
          '</div>'
    )


def _fusion_opt_section(fopt_stats: dict[str, dict]) -> str:
    """4 algebraic / fusion-optimization kernels: conv+bias+relu+add,
    gemm+bias+relu (cublasLt), AᵀA→cublasSsyrk via operand alias,
    1×1 conv → cublasSgemmStridedBatched. Each picks a faster cuBLAS /
    cublasLt / cuDNN entry point than the matcher's default routing."""
    rows = []
    for k, entries in FUSION_OPT_RUNTIMES.items():
        first = True
        rowspan = len(entries)
        stats = fopt_stats.get(k, {})
        if stats.get("ce_url"):
            kernel_link = (
                f'<a class="kernel" href="{stats["ce_url"]}" target="_blank">'
                f'{k}</a>'
            )
        else:
            kernel_link = f'<span class="nope">{k}</span>'
        ir_link = (
            f'<a class="viewer" href="{stats["page_filename"]}" '
            f'style="margin-left:10px">[IR preview]</a>'
            if stats.get("page_filename") else ""
        )
        l = stats.get("launches", 0)
        r = stats.get("residual", 0)
        fcount = stats.get("residual_for", 0)
        match_status = ("FULL" if l > 0 and r == 0 and fcount == 0 else
                        "PARTIAL" if l > 0 else "NONE")
        match_cls = ("pass" if match_status == "FULL" else
                     "partial" if match_status == "PARTIAL" else "none")
        for e in entries:
            size, shape = e["size"], e["shape"]
            gpu, cpu = e["gpu_s"], e["cpu_s"]
            speedup = cpu / gpu if gpu > 0 else 0.0
            su_cls = ("pass" if speedup >= 2.0
                      else "partial" if speedup >= 0.8
                      else "none")
            cmark = {"PASS": "&check;", "FP-noise": "&asymp;",
                     "DIFF": "&cross;"}.get(e["correct"], "?")
            note = e.get("notes", "")
            if first:
                kernel_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top">'
                    f'{kernel_link}{ir_link}'
                    f'<div style="font-size:11px; color:#666; margin-top:4px">'
                    f'  matcher: <span class="{match_cls}">'
                    f'<b>{match_status}</b></span> ({l} launch, {r} res lg, '
                    f'{fcount} loops)</div></td>'
                )
            else:
                kernel_cell = ""
            first = False
            rows.append(
                "<tr>"
                + kernel_cell
                + f'<td style="font-size:12px"><b>{size}</b></td>'
                + f'<td style="font-size:11px; font-family:monospace">{shape}</td>'
                + f'<td style="font-size:12px; text-align:right">{_fmt_seconds(gpu)}</td>'
                + f'<td style="font-size:12px; text-align:right">{_fmt_seconds(cpu)}</td>'
                + f'<td class="{su_cls}" style="font-size:12px; text-align:right">'
                + f'{speedup:.0f}&times; {cmark}</td>'
                + f'<td style="font-size:11px; color:#555; max-width:340px">{note}</td>'
                + "</tr>")
    table = (
        '<table><thead><tr>'
        '<th>kernel</th><th>dataset</th><th>shape</th>'
        '<th>GPU</th><th>CPU (3-loop)</th>'
        '<th>GPU speedup</th><th>notes</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
    )
    return (
        '<div class="section-header" id="fusion-opt" '
        'style="background:#ffeacd; border-color:#d6b078">'
        '  <h2 class="section-title">Fusion optimization '
        '  (algebraic rewrites for fast cuBLAS / cublasLt / cuDNN paths)</h2>'
        '</div>'
        '<div class="intro">'
        '  Four follow-on entries to the extracted-darknet matcher work. '
        '  Each is an <em>algebraic</em> rewrite — same math as the naive '
        '  multi-op chain, but routed to a single fused cuDNN / cublasLt / '
        '  cuBLAS call that fires faster paths. The wins range from '
        '  <b>23×</b> (conv chain) to <b>3393×</b> (AᵀA → syrk) over the '
        '  CPU 3-loop reference.'
        '  <br><br>'
        '  <b>Matched launch symbols</b> introduced by these compositions:'
        '  <ul style="margin:6px 0 10px 24px; padding:0; font-size:12px">'
        '  <li><code>@cudnnConvBiasReluAddFwdFused</code> — 5-step: init + conv + '
        '      bias + residual-add + relu. Routes to '
        '      <code>cudnnConvolutionBiasActivationForward</code> with the Z '
        '      addend (α₂=1) for the skip connection.</li>'
        '  <li><code>@cublasLtMatmulBiasReluFused</code> — 4-step: init + gemm + '
        '      bias + relu. Routes to <code>cublasLtMatmul</code> with '
        '      <code>CUBLASLT_EPILOGUE_RELU_BIAS</code>. Needs '
        '      <code>libcublasLt</code> at link.</li>'
        '  <li><code>@cublasDsyrk_alias</code> — operand-alias discriminator on '
        '      the gemm-shape composition. Detected when both gemm inputs '
        '      resolve (after walking through <code>polygeist.submap</code>) '
        '      to the same underlying tensor. Routes to '
        '      <code>cublasSsyrk_v2</code> — half the flops, half the bandwidth.</li>'
        '  <li><code>@cublasGemmFor1x1Conv</code> — distinguishes a 4-par+1-red '
        '      contraction (K=1 conv after trivial-loop elimination) from the '
        '      4-par+3-red K×K conv. Routes to <code>cublasSgemmStridedBatched</code> '
        '      because cuDNN&apos;s K=1 path is generic / slow.</li>'
        '  </ul>'
        '  Pre-pass in the lowering elides redundant <code>memset_zero_2D</code> '
        '  launches that precede a <code>syrk_alias</code> (since syrk uses β=0). '
        '  <code>resolveSubmapBase</code> now walks through both '
        '  <code>polygeist.submap</code> and <code>polygeist.submapInverse</code>, '
        '  chaining up to 16 hops — needed to handle the nested chains the '
        '  pre-init memset leaves behind.'
        '</div>'
        + table
        # Headline call-out.
        + '<div style="margin-top:14px; padding:10px 14px; '
          'background:#fff3e0; border-left:4px solid #d6824a;">'
          '  <b>Speedup headlines (LARGE on Jetson Orin):</b>'
          '  <ul style="margin:6px 0 0 24px; padding:0; font-size:13px">'
          '  <li>conv + bias + relu + residual-add — <b>23×</b> (closes '
          '      the standalone shortcut-add GPU loss; bandwidth-bound bn '
          '      effectively rides free on the conv)</li>'
          '  <li>gemm + bias + relu — <b>901×</b> (cublasLt epilogue + '
          '      tensor cores on 2048³ FP32; CPU 3-loop is cache-hostile)</li>'
          '  <li>AᵀA → cublasSsyrk — <b>3393×</b> (half the flops + clean '
          '      tensor-core dispatch + cache-hostile CPU pattern)</li>'
          '  <li>1×1 conv → cublasSgemmStridedBatched — <b>105×</b> '
          '      (bypasses cuDNN&apos;s generic K=1 path; gets tensor cores '
          '      via the per-batch gemm)</li>'
          '  </ul>'
          '</div>'
    )


def _extracted_darknet_section(ex_darknet_stats: dict[str, dict]) -> str:
    """5 batched CNN-block primitives extracted from darknet, raised
    through the full Polygeist pipeline, matched to cuDNN library
    symbols, ABI-lowered, cross-compiled, run on the Jetson Orin
    silicon. Each kernel gets a Compiler Explorer deep-link (clickable
    name) + an IR-preview page (the [IR preview] link)."""
    rows = []
    for k, entries in EXTRACTED_DARKNET_RUNTIMES.items():
        first = True
        rowspan = len(entries)
        stats = ex_darknet_stats.get(k, {})
        # Kernel-name cell on the first row carries the CE deep-link +
        # an [IR preview] page link, mirroring the polybench / darknet
        # row layout. CE URL & per-kernel page are produced by
        # build_kernel_page → returns ce_url + page_filename.
        if stats.get("ce_url"):
            kernel_link = (
                f'<a class="kernel" href="{stats["ce_url"]}" target="_blank">'
                f'{k}</a>'
            )
        else:
            kernel_link = f'<span class="nope">{k}</span>'
        ir_link = (
            f'<a class="viewer" href="{stats["page_filename"]}" '
            f'style="margin-left:10px">[IR preview]</a>'
            if stats.get("page_filename") else ""
        )
        # Per-kernel match stats — same shape the other sections use.
        l = stats.get("launches", 0)
        r = stats.get("residual", 0)
        fcount = stats.get("residual_for", 0)
        match_status = ("FULL" if l > 0 and r == 0 and fcount == 0 else
                        "PARTIAL" if l > 0 else "NONE")
        match_cls = ("pass" if match_status == "FULL" else
                     "partial" if match_status == "PARTIAL" else "none")
        for e in entries:
            size, shape = e["size"], e["shape"]
            gpu, cpu = e["gpu_s"], e["cpu_s"]
            speedup = cpu / gpu if gpu > 0 else 0.0
            su_cls = ("pass" if speedup >= 2.0
                      else "partial" if speedup >= 0.8
                      else "none")
            cmark = {"PASS": "&check;", "FP-noise": "&asymp;",
                     "DIFF": "&cross;"}.get(e["correct"], "?")
            note = e.get("notes", "")
            if first:
                kernel_cell = (
                    f'<td rowspan="{rowspan}" style="vertical-align:top">'
                    f'{kernel_link}{ir_link}'
                    f'<div style="font-size:11px; color:#666; margin-top:4px">'
                    f'  matcher: <span class="{match_cls}">'
                    f'<b>{match_status}</b></span> ({l} launch,'
                    f' {r} residual lg, {fcount} loops)'
                    f'</div></td>'
                )
            else:
                kernel_cell = ""
            first = False
            rows.append(
                "<tr>"
                + kernel_cell
                + f'<td style="font-size:12px"><b>{size}</b></td>'
                + f'<td style="font-size:11px; font-family:monospace">{shape}</td>'
                + f'<td style="font-size:12px; text-align:right">{_fmt_seconds(gpu)}</td>'
                + f'<td style="font-size:12px; text-align:right">{_fmt_seconds(cpu)}</td>'
                + f'<td class="{su_cls}" style="font-size:12px; text-align:right">'
                + f'{speedup:.2f}&times; {cmark}</td>'
                + f'<td style="font-size:11px; color:#555; max-width:340px">{note}</td>'
                + "</tr>")
    table = (
        '<table><thead><tr>'
        '<th>kernel</th>'
        '<th>dataset</th>'
        '<th>shape</th>'
        '<th>GPU (cuDNN)</th>'
        '<th>CPU (3-loop)</th>'
        '<th>GPU speedup</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows) +
        '</tbody></table>'
        # Fusion punchline — make the "ride free" insight crisp.
        '<div style="margin-top:14px; padding:10px 14px; '
        'background:#eafff0; border-left:4px solid #4a8;">'
        '  <b>Fusion punchline.</b> Sum the three standalone LARGE '
        '  GPU launches as if you ran them back-to-back '
        '  (conv2d_batched 137.0&nbsp;ms + batchnorm_batched 11.3&nbsp;ms + '
        '  one cudnnAddTensor-shaped ReLU &asymp; 50&nbsp;ms &approx; '
        '  <b>~198&nbsp;ms</b>) vs the fused '
        '  <code>conv_bn_relu_batched</code> LARGE at '
        '  <b>137.8&nbsp;ms</b>. Same conv work, but with bn + relu '
        '  absorbed into the conv&apos;s compute-bound memory pass &mdash; '
        '  the bandwidth-bound ops effectively cost zero. On the CPU '
        '  side the two are within 0.5% of each other (3260 vs 3244&nbsp;ms) '
        '  because the CPU never paid per-call setup in the first place; '
        '  the GPU&apos;s gain comes entirely from collapsing 3 cuDNN '
        '  descriptor / algo-select / sync rounds into 1.'
        '</div>'
        # Numeric agreement (FP-noise) callout.
        '<div style="margin-top:8px; padding:10px 14px; '
        'background:#f4f6fa; border-left:4px solid #88a;">'
        '  <b>FP-noise comparison.</b> Tensor-core kernels reorder the '
        '  accumulation; CPU 3-loop accumulates in natural order. '
        '  Dumps printed at <code>%0.4f</code>:'
        '  <ul style="margin:6px 0 0 24px; padding:0; font-size:12px">'
        '  <li><code>conv2d_batched LARGE</code>: 0% bit-exact, max|d| = '
        '      7.9e-3, mean|d| = 6.8e-3, max relative = 6.5e-5. Every '
        '      output drifts by ~7 ULPs at print precision because 576 '
        '      muladds per output (IC=64 &times; K&sup2;=9) make the '
        '      accumulation-order drift visible.</li>'
        '  <li><code>conv_bn_relu_batched LARGE</code>: '
        '      <b>75% bit-exact</b>, max|d| = 3.4e-3, mean|d| = 1.4e-4. '
        '      Better than conv alone &mdash; BN&apos;s per-channel '
        '      normalization scales drifts down, ReLU zeros 73% of '
        '      outputs (zero is exactly representable). Of the remaining '
        '      27% live outputs only 3.7% exceed |d| > 1e-3.</li>'
        '  <li><code>maxpool_batched</code>, <code>shortcut_batched</code>: '
        '      100% bit-exact at all sizes. Max + plain add are '
        '      order-independent.</li>'
        '  <li><code>batchnorm_batched LARGE</code>: 99.9% bit-exact, '
        '      max|d| = 1e-4 (one print-precision ULP) on 0.1% of elems.</li>'
        '  </ul>'
        '</div>'
    )
    return (
        '<div class="section-header" id="extracted-darknet">'
        '  <h2 class="section-title">extracted darknet '
        '  (matcher + cuDNN runtime, Jetson Orin silicon)</h2>'
        '</div>'
        '<div class="intro">'
        '  Four batched CNN-block primitives extracted as polybench-style '
        '  single-file <code>.c</code> kernels in '
        '  <code>third_party/cnn-extracted/</code>: <b>conv2d_batched</b>, '
        '  <b>maxpool_batched</b>, <b>batchnorm_batched</b>, '
        '  <b>shortcut_batched</b>. Together they cover every primitive '
        '  in a ResNet residual block except ReLU.'
        '  <br><br>'
        '  Each kernel goes through the full Polygeist pipeline: cgeist '
        '  &rarr; <code>--raise-affine-to-linalg-pipeline</code> &rarr; '
        '  <code>--linalg-debufferize</code> &rarr; '
        '  <code>kernel_match_rewrite.py</code> &rarr; '
        '  <code>--lower-kernel-launch-to-cublas</code> (resolves '
        '  <code>polygeist.submap</code> operands back to their base 4D '
        '  tensors, emits <code>func.call</code> to the runtime shim) '
        '  &rarr; aarch64 cross-compile against <code>libcudnn.so.9</code> '
        '  &rarr; ship to Jetson Orin &rarr; run. Numbers below are wall-'
        '  clock for a single shim call including <code>cudaHostRegister</code> '
        '  mapping + the cuDNN forward call + a final stream sync.'
        '  <br><br>'
        '  <b>Matched launch symbols</b> (one per row in the table, '
        '  ordered longest-composition first in <code>composition_library()</code>):'
        '  <ul style="margin:6px 0 10px 24px; padding:0; font-size:12px">'
        '  <li><code>@cudnnConvBnReluFwdFused</code> — 4-step: init zero + '
        '      conv contraction (4 par + 3 red) + bn in-place (4 par, 4 ins) + '
        '      relu in-place. Lowers to one '
        '      <code>cudnnConvolutionBiasActivationForward</code> with '
        '      <code>CUDNN_ACTIVATION_RELU</code> after host-side BN-folding '
        '      (<code>F&apos;[oc] = F[oc] * scale[oc] * inv_std[oc]</code>, '
        '      <code>b&apos;[oc] = bias[oc] - scale[oc] * mean[oc] * inv_std[oc]</code>).</li>'
        '  <li><code>@cudnnConvolutionFwd_batched</code> — 2-step: init zero + 7-iter '
        '      contraction. Lowers to <code>cudnnConvolutionForward</code>.</li>'
        '  <li><code>@cudnnMaxPoolFwd_batched</code> — 2-step: init -INF + max-reduce. '
        '      Lowers to <code>cudnnPoolingForward</code>.</li>'
        '  <li><code>@cudnnBatchNormalizationForwardInference</code> — 1-step elementwise '
        '      (5 ins, 4 par, 0 red). Lowers to '
        '      <code>cudnnBatchNormalizationForwardInference</code> with variance '
        '      derived from inv_std + eps.</li>'
        '  <li><code>@cudnnAddTensor_batched</code> — 1-step <code>Out + In(0)</code>. '
        '      Lowers to <code>cudnnAddTensor</code> with &alpha;=&beta;=1.</li>'
        '  </ul>'
        '  <br><br>'
        '  The headline win is <b>23.8&times; for conv2d_batched LARGE</b> — '
        '  cuDNN&apos;s tensor-core kernels shred a 32&times;64&times;56&sup2; '
        '  ResNet conv where the CPU 3-loop reference takes 3.3 s. The '
        '  bandwidth-bound elementwise kernels (batchnorm, shortcut) lose '
        '  to the CPU at single-call granularity — the cuDNN setup overhead '
        '  doesn&apos;t amortize without device-residency hoisting (the '
        '  documented Phase-2 follow-up in '
        '  <code>project-phase2-cublas-abi-lowering</code>).'
        '  <br><br>'
        '  The last row, <b>conv_bn_relu_batched</b>, is the operator-'
        '  fusion follow-up: a kernel that chains conv + bn-inference + '
        '  relu (canonical ResNet inner pattern) and a matcher 4-step '
        '  composition <code>cudnnConvBnReluFwdFused</code> that folds '
        '  all four loop nests (init + conv + bn-inplace + relu-inplace) '
        '  into one launch. The runtime shim applies the standard '
        '  &quot;BN-folding&quot; trick — pre-multiplying the filter by '
        '  <code>scale * inv_std</code> and adjusting the bias — then '
        '  issues a single <code>cudnnConvolutionBiasActivationForward</code> '
        '  call. Result: 137.8 ms LARGE (essentially the same as conv2d_'
        '  batched alone), but doing all three operations. The bandwidth-'
        '  bound bn and relu effectively become free; they ride the conv&apos;s '
        '  compute-bound memory pass.'
        '  <br><br>'
        '  Correctness key: <span class="pass">&check; PASS</span> = bit-'
        '  exact match with the CPU stub (maxpool, shortcut are integer-'
        '  like ops); <span class="partial">&asymp; FP-noise</span> = '
        '  cuDNN tensor-core accumulation order differs from CPU naive '
        '  order at the third decimal (expected, not a correctness bug).'
        '</div>'
        + table
    )


def _backend_overview(polybench_stats: dict[str, dict]) -> str:
    launching = [s for s in polybench_stats.values() if s.get("launches", 0) > 0]
    abi_ready = sum(bool(s.get("abi_lowered")) for s in launching)
    cpu_ready = sum(bool(s.get("cpu_backend")) for s in launching)
    cuda_ready = sum(bool(s.get("cuda_backend")) for s in launching)
    cblas_symbols = ", ".join(
        f"<code>{html.escape(s)}</code>" for s in sorted(CPU_CBLAS_CALLS)
    )
    return (
        '<div class="section-header"><h2 class="section-title">Where CPU and GPU lowering diverge</h2></div>'
        '<div class="intro"><b>The frontend, raising, debufferization, Egglog proof, '
        'and <code>kernel.launch</code> selection are shared.</b> The production '
        '<code>--lower-kernel-launch-to-cublas</code> pass then emits one target-neutral '
        '<code>@polygeist_*</code> pointer ABI. Backend choice happens at compile/link time.</div>'
        '<div class="backend-flow">'
        '<div class="backend-common"><b>Shared compiler path</b><br>C → cgeist → Linalg → '
        'Egglog → kernel.launch → func.call @polygeist_*</div>'
        '<div class="backend-arrow">↙</div>'
        '<div class="backend-card"><b>CPU/C executable</b><br>Native host LLVM plus '
        '<code>polygeist_cublas_rt_cpu.c</code>.<br>Reference C by default; set '
        '<code>POLYGEIST_CPU_BLAS=1</code> for CBLAS where supported.</div>'
        '<div class="backend-arrow">↘</div>'
        '<div class="backend-card"><b>GPU executable</b><br>AArch64/host orchestration plus '
        '<code>polygeist_cublas_rt_cuda.c</code> and CUDA vendor libraries. The shim '
        'handles transfers, synchronization, and optional residency/graph paths.</div>'
        '</div>'
        '<div class="intro"><b>Current PolyBench snapshot:</b> '
        f'{len(launching)} kernels emit at least one launch; {abi_ready} lower to the shared ABI; '
        f'{cpu_ready} have all emitted calls implemented by the CPU runtime; {cuda_ready} have all '
        'emitted calls implemented by the CUDA runtime.<br><br>'
        '<b>Optimized CPU CBLAS routes:</b> ' + cblas_symbols + '. Other CPU-runtime symbols '
        'use reference C unless another optimized CPU backend is added.<br><br>'
        '<b>Generic CPU fallback:</b> <code>--lower-kernel-launch</code> is a separate path '
        'that restores the canonical Linalg body and lowers it to loops. It does not call a '
        'vendor library. Unmatched residual Linalg also remains host code.</div>'
    )


def _ginsbach_artifact_stem(source: str) -> str:
    """Mirror ginsbach_asplos18_audit.py's per-source directory naming."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", source)


def _build_ginsbach_detail_pages() -> dict[tuple[str, str], str]:
    """Render program and translation-unit C → Linalg → matcher pages."""
    unit_rows = _read_csv(GINSBACH_AUDIT_ROOT / "translation_units.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in unit_rows:
        key = (row.get("suite", ""), row.get("program", ""))
        grouped.setdefault(key, []).append(row)

    program_links: dict[tuple[str, str], str] = {}
    for (suite, program), units in sorted(grouped.items()):
        program_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{suite}_{program}")
        program_page = f"ginsbach_{program_slug}.html"
        program_links[(suite, program)] = program_page
        unit_table_rows = []

        for unit in sorted(units, key=lambda item: item.get("source", "")):
            source_rel = unit.get("source", "")
            source_path = REPO_ROOT / source_rel
            artifact_dir = (
                GINSBACH_AUDIT_ROOT / "units"
                / _ginsbach_artifact_stem(source_rel)
            )
            unit_slug = _ginsbach_artifact_stem(source_rel)
            unit_page = f"ginsbach_unit_{unit_slug}.html"
            stages = [
                ("source", "Original C/C++ source", source_path),
                ("cgeist", "cgeist output (pre-raise MLIR)",
                 artifact_dir / "affine.mlir"),
                ("linalg", "Raised + debufferized Linalg",
                 artifact_dir / "linalg.mlir"),
                ("matched", "Matcher-generated IR (kernel.launch)",
                 artifact_dir / "matched.mlir"),
            ]
            available = [(name, title, path) for name, title, path in stages
                         if path.exists()]
            stage_links = " · ".join(
                f'<a href="{unit_page}#{name}">{html.escape(title)}</a>'
                for name, title, _ in available
            ) or '<span class="nope">artifacts unavailable</span>'
            unit_name = Path(source_rel).name
            unit_table_rows.append(
                '<tr>'
                f'<td><a class="kernel" href="{unit_page}">'
                f'{html.escape(unit_name)}</a></td>'
                f'<td><code>{html.escape(source_rel)}</code></td>'
                f'<td>{unit.get("linalg_generics", "0")}</td>'
                f'<td>{unit.get("kernel_launches", "0")}</td>'
                f'<td>{stage_links}</td>'
                '</tr>'
            )

            jump_links = " · ".join(
                f'<a href="#{name}">{html.escape(title)}</a>'
                for name, title, _ in available
            )
            unit_body = (
                f'<div class="header"><h1><a href="{program_page}">'
                f'← {html.escape(suite)} / {html.escape(program)}</a> '
                f'&nbsp; {html.escape(unit_name)}</h1></div>'
                '<div class="summary" style="padding:8px 20px; '
                'border-bottom:1px solid #eee;background:#fafafa;font-size:13px;">'
                f'<b>{unit.get("linalg_generics", "0")}</b> linalg.generic '
                f'&nbsp;·&nbsp; <b>{unit.get("kernel_launches", "0")}</b> '
                f'kernel.launch &nbsp;|&nbsp; {jump_links}</div>'
            )
            for name, title, path in available:
                rendered, _ = syntax_highlight(
                    path.read_text(errors="replace"),
                    "c" if name == "source" else "llvm",
                )
                unit_body += (
                    f'<h2 id="{name}">{html.escape(title)}</h2>'
                    f'<div class="container">{rendered}</div>'
                )
            if not available:
                unit_body += (
                    '<div class="intro"><b>Artifacts unavailable.</b> Rerun '
                    '<code>ginsbach_asplos18_audit.py</code> and point '
                    '<code>POLYGEIST_GINSBACH_AUDIT_ROOT</code> at its output.'
                    '</div>'
                )
            OUTPUT_DIR.joinpath(unit_page).write_text(
                render_html(f"{program}: {unit_name}", unit_body, "")
            )

        program_body = (
            '<div class="header"><h1><a href="ginsbach.html">'
            '← Ginsbach ASPLOS\'18</a> &nbsp; '
            f'{html.escape(suite)} / {html.escape(program)}</h1></div>'
            '<div class="intro">Click a translation unit to inspect its '
            'original source, raised Linalg, and exact matcher-generated IR. '
            'The counts are static operations in the corpus audit, not '
            'dynamic runtime invocations.</div>'
            '<table><thead><tr><th>translation unit</th><th>source path</th>'
            '<th>linalg</th><th>launches</th><th>direct views</th>'
            '</tr></thead><tbody>' + ''.join(unit_table_rows)
            + '</tbody></table>'
        )
        OUTPUT_DIR.joinpath(program_page).write_text(
            render_html(f"Ginsbach: {suite}/{program}", program_body, "")
        )
    return program_links


def _ginsbach_page() -> tuple[str, int]:
    """Render the external-library-only ASPLOS'18 corpus audit."""
    rows = _read_csv(GINSBACH_SUMMARY)
    if not rows:
        return (
            '<div class="intro"><b>Ginsbach ASPLOS\'18 audit unavailable.</b> '
            f'Missing <code>{html.escape(str(GINSBACH_SUMMARY))}</code>.</div>',
            0,
        )

    numeric_fields = (
        "units", "frontend_ok", "raise_ok", "linalg_generics",
        "kernel_launches", "structured_fusions", "structured_reductions",
        "structured_stencils", "histogram_candidates", "csr_spmv_candidates",
        "published_idioms",
    )
    totals = {
        field: sum(int(row.get(field, 0) or 0) for row in rows)
        for field in numeric_fields
    }
    # Memory initialization sites are intentionally excluded from this
    # computational comparison. They remain in the raw audit snapshot, but do
    # not count toward library-compute coverage or silicon validation.
    memory_only_launches = {
        ("snu-npb", "BT"): 6,
        ("snu-npb", "LU"): 1,
    }
    computational_launches = totals["kernel_launches"] - sum(
        memory_only_launches.values()
    )
    backend_routes = {
        ("snu-npb", "BT"): (
            "cuBLAS strided-batched subtract GEMM + GEMV "
            "(combined ABI passes 3/3; full-app transform pending)"
        ),
        ("snu-npb", "CG"): "cuSPARSE CSR SpMV ×4",
        ("snu-npb", "FT"): "cuFFT: no executable match yet",
        ("snu-npb", "IS"): "CUB histogram ×6 corpus sites; ×2 in rank",
        ("snu-npb", "UA"): (
            "cuBLAS DAXPBY ×3; tiny Ddot ×14 recognized but "
            "profitability-gated"
        ),
        ("parboil", "sgemm"): "cuBLAS SGEMM ×1",
        ("parboil", "spmv"): (
            "JDS-to-CSR storage adapter + NVIDIA cuSPARSE SpMV ×1"
        ),
        ("parboil", "stencil"): "cuDNN 3D convolution ×1",
    }
    silicon_rows = _read_csv(GINSBACH_SILICON)
    silicon = {
        (row.get("suite", ""), row.get("program", "")): row
        for row in silicon_rows
    }
    program_links = _build_ginsbach_detail_pages()

    metric_specs = (
        ("Translation units", totals["units"]),
        ("Frontend passed", totals["frontend_ok"]),
        ("Raise passed", totals["raise_ok"]),
        ("Linalg generics", totals["linalg_generics"]),
        ("Computational launches", computational_launches),
        ("Silicon-tested programs", len(silicon_rows)),
        ("Published idioms", totals["published_idioms"]),
    )
    metrics = ''.join(
        '<div class="audit-metric"><b>' + str(value) + '</b><span>'
        + html.escape(label) + '</span></div>'
        for label, value in metric_specs
    )
    body_rows = []
    for row in rows:
        key = (row.get("suite", ""), row.get("program", ""))
        raw_launches = int(row.get("kernel_launches", 0) or 0)
        launches = raw_launches - memory_only_launches.get(key, 0)
        launch_class = "pass" if launches else "nope"
        route = backend_routes.get(
            key, "— (memory-only excluded)" if raw_launches else "—"
        )
        silicon_row = silicon.get(key)
        if silicon_row:
            silicon_status = silicon_row.get("status", "PASS")
            validation_class = (
                "pass" if silicon_status == "PASS"
                else "partial" if silicon_status == "PARTIAL"
                else "nope"
            )
            validation = (
                f'<span class="{validation_class}">{html.escape(silicon_status)}</span>'
                f'<br><small>{html.escape(silicon_row.get("validation_scope", ""))}</small>'
            )
            runtime = html.escape(silicon_row.get("runtime", "not measured"))
        else:
            validation = '<span class="nope">not run</span>'
            runtime = "—"
        program_name = html.escape(row.get("program", ""))
        if key in program_links:
            program_cell = (
                f'<a class="kernel" href="{program_links[key]}">'
                f'{program_name}</a>'
            )
        else:
            program_cell = f'<b>{program_name}</b>'
        body_rows.append(
            '<tr>'
            f'<td>{html.escape(row.get("suite", ""))}</td>'
            f'<td>{program_cell}</td>'
            f'<td>{row.get("units", "0")}</td>'
            f'<td>{row.get("raise_ok", "0")}/{row.get("units", "0")}</td>'
            f'<td>{row.get("linalg_generics", "0")}</td>'
            f'<td class="{launch_class}">{launches}</td>'
            f'<td>{html.escape(route)}</td>'
            f'<td>{validation}</td>'
            f'<td>{runtime}</td>'
            f'<td>{row.get("structured_fusions", "0")}</td>'
            f'<td>{row.get("structured_reductions", "0")}</td>'
            f'<td>{row.get("structured_stencils", "0")}</td>'
            f'<td>{row.get("histogram_candidates", "0")}</td>'
            f'<td>{row.get("published_idioms", "0")}</td>'
            '</tr>'
        )

    body = (
        '<div class="intro"><b>Ginsbach et al., ASPLOS 2018 — '
        'external-library-only audit.</b> The current source corpus contains '
        '21 benchmark programs. Structural Egglog detections are shown '
        'separately from executable launches: only matches that lower to a '
        'pre-existing external library or CUDA platform API count as '
        'computational launches. CUDA memset sites are excluded. Silicon '
        'status and runtime scope are reported separately so a library smoke '
        'is never presented as a full-application measurement. Click any '
        'program name to inspect its translation units and their source, '
        'raised Linalg, and matcher-generated IR.</div>'
        f'<div class="audit-metrics">{metrics}</div>'
        '<div class="intro"><b>Analysis-only inventory:</b> '
        f'{totals["structured_fusions"]} Egglog-proved structured regions; '
        f'{totals["structured_reductions"]} reduction-shaped regions; '
        f'{totals["structured_stencils"]} stencil-shaped regions; '
        f'{totals["histogram_candidates"]} histogram candidates; '
        f'{totals["csr_spmv_candidates"]} CSR SpMV candidates. '
        'These candidates do not count as executable matches.</div>'
        '<table class="audit-table"><thead><tr>'
        '<th>suite</th><th>program</th><th>units</th><th>raised</th>'
        '<th>linalg</th><th>compute launches</th><th>external route</th>'
        '<th>silicon validation</th><th>silicon runtime</th>'
        '<th>Egglog regions</th><th>reductions</th><th>stencils</th>'
        '<th>histograms</th><th>paper idioms</th>'
        '</tr></thead><tbody>' + ''.join(body_rows) + '</tbody></table>'
        '<div class="intro"><b>Silicon evidence:</b> NPB CG Class S passed '
        'three post-reboot runs (0.13 s each; 501.82 median Mop/s). '
        'The original Parboil SGEMM application exactly matches the CPU output '
        'and is 3.90× faster in compute (1.99× including I/O). The original '
        'Parboil stencil application is correct and, after caching cuDNN setup, '
        'is currently 1.15× slower. NPB UA verifies with its three DAXPBY '
        'replacements; a profitability guard now keeps its proven-tiny Ddot '
        'operations in Linalg. The BT combined strided-batched GEMM+GEMV ABI '
        'passes 3/3, and Egglog loop lifting lowers the prototype, but full BT '
        'still needs helper-first raising and workspace privatization. NPB FT '
        'passes its CPU baseline but does not yet match '
        'cuFFT. NPB IS passes full '
        'Class-S verification in three post-reboot runs (122.61 median Mop/s) '
        'with the original driver and full_verify around '
        'a source-faithful rank core containing two CUB histogram sites. '
        'Exact sizes, timing scopes, and incomplete outcomes are shown in the '
        'table. See '
        '<code>issues/ginsbach_asplos18/SILICON_STATUS.md</code> for the '
        'exact evidence and remaining gaps.</div>'
    )
    return body, len(rows)


def build_site_pages(polybench_stats: dict[str, dict],
                     aten_stats: dict[str, dict],
                     mfem_stats: list[dict],
                     mfem_application_stats: list[dict],
                     mfem_application_extraction_stats: list[dict],
                     llama_forward_stats: dict[str, dict],
                     whisper_ops_stats: dict[str, dict],
                     stencil_conv2d_stats: dict[str, dict],
                     llmc_stats: dict[str, dict],
                     darknet_stats: dict[str, dict],
                     ex_darknet_stats: dict[str, dict],
                     fopt_stats: dict[str, dict]) -> dict[str, str]:
    ginsbach_body, ginsbach_count = _ginsbach_page()
    modified_body, modified_count = _modified_kernels_page()
    llama_forward_section = _build_section(
        title="Llama forward fixtures (raised C benchmarks)",
        anchor="llama-forward",
        blurb=(
            "Source-level C fixtures in <code>third_party/cnn-extracted</code> "
            "covering the pieces of a one-token Llama decode step. The rows "
            "below include the individual kernels used in the op sweep plus "
            "<code>extended_forward</code>, the fuller one-token, one-layer "
            "benchmark that combines token embedding, attention RMSNorm, "
            "Q/K/V projections, split RoPE, KV cache read/write, attention "
            "scores + softmax, attention value matvec, output projection, "
            "residuals, FFN RMSNorm, gate/up/down projections, SwiGLU, final "
            "RMSNorm, and lm_head logits. Each row has a Compiler Explorer "
            "deep-link and an IR preview for the C benchmark we are raising."
        ),
        kernel_stats=llama_forward_stats,
        notes=LLAMA_FORWARD_NOTES,
        blockers=LLAMA_FORWARD_BLOCKERS,
        extra_html=_llama_forward_runtime_summary(),
        runtimes=LLAMA_FORWARD_RUNTIMES,
        display_names=LLAMA_FORWARD_DISPLAY_NAMES,
        order=LLAMA_FORWARD_ORDER,
        runtime_headers=(
            "Jetson<br>case",
            "Raised pipeline<br>(rt-gpu)",
            "Reference<br>CUDA",
            "comparison",
            "notes",
        ),
    )
    whisper_ops_section = _build_section(
        title="Whisper extracted kernels (raised C fixtures)",
        anchor="whisper-ops",
        blurb=(
            "Source-level C fixtures in <code>third_party/cnn-extracted/"
            "whisper_ops.c</code> covering representative Whisper/ggml "
            "inference compute bodies: vector dot, softmax, RMSNorm-style "
            "normalization, GELU, and encoder-side 1D convolution. These rows "
            "are intentionally the exposed kernel bodies, not full "
            "<code>ggml_tensor</code> framework functions; they show which "
            "algorithmic kernels the linalg raising path can express once the "
            "framework metadata, SIMD dispatch, and helper-call scaffolding "
            "are isolated."
        ),
        kernel_stats=whisper_ops_stats,
        notes=WHISPER_OPS_NOTES,
        blockers=WHISPER_OPS_BLOCKERS,
        display_names=WHISPER_OPS_DISPLAY_NAMES,
        order=WHISPER_OPS_ORDER,
        runtime_headers=(
            "case",
            "raised pipeline",
            "reference",
            "comparison",
            "notes",
        ),
    )
    stencil_conv2d_section = _build_section(
        title="Stencil Conv2D fixtures (cuDNN tensor ntap target)",
        anchor="stencil-conv2d",
        blurb=(
            "Image-processing and finite-difference stencil fixtures written "
            "as plain C neighbourhood expressions. The debufferized tensor "
            "forms raise to one loop-free linalg.generic and match the "
            "generalized packed-weight "
            "<code>@cudnnConvolution2D_ntap_f32_tensor</code> route. The "
            "legacy memref 9/25-tap entries remain available for explicit "
            "no-debufferize runs. Each row links to Compiler Explorer and an "
            "IR preview for the raised C fixture."
        ),
        kernel_stats=stencil_conv2d_stats,
        notes=STENCIL_CONV2D_NOTES,
        blockers=STENCIL_CONV2D_BLOCKERS,
        runtimes=STENCIL_CONV2D_RUNTIMES,
        display_names=STENCIL_CONV2D_DISPLAY_NAMES,
        order=STENCIL_CONV2D_ORDER,
        runtime_headers=(
            "Jetson<br>case",
            "Raised pipeline<br>(cuDNN)",
            "Target<br>library",
            "comparison",
            "notes",
        ),
    )
    llmc_section = _build_section(
        title="llm.c (karpathy/llm.c — GPT-2 in C, forward + backward)",
        anchor="llmc",
        blurb=(
            "15 leaf kernels from train_gpt2.c — the full GPT-2 building "
            "blocks for both inference and training: encoder, layernorm, "
            "matmul, attention, gelu, residual, softmax, crossentropy "
            "(forward + backward where it applies). This is a related C LLM "
            "suite with wider coverage. It stresses the pipeline "
            "in new ways: indirect-index lookups (encoder), math.h ext-call "
            "bodies (gelu/crossentropy via tanhf/logf), full scaled-dot "
            "attention (4 fused generics including softmax-shaped reductions), "
            "and the layernorm dominance issue in both debuf paths. The "
            "<code>matmul_forward_naive</code> reference is used instead of "
            "the tiled <code>matmul_forward</code>."
        ),
        kernel_stats=llmc_stats,
        notes=LLMC_NOTES,
        blockers=LLMC_BLOCKERS,
    )
    darknet_section = _build_section(
        title="darknet (pjreddie/darknet — full source bake)",
        anchor="darknet",
        blurb=(
            "Empirical &quot;matcher coverage survey&quot; over all 46 .c "
            "files in <code>third_party/darknet/src/</code>. cgeist baked "
            "with <code>--function=*</code> and inlining enabled; "
            "every file's debuferized output ran through the matcher. "
            "<br><br>"
            "Outcome (matches my earlier prediction of ~2% hit rate): "
            "<b>1 file matches</b> (<code>gemm.c</code>, 6 kernel.launch "
            "across gemm_nn/nt/tn/tt + gemm_bin variants). The rest splits "
            "into three buckets:"
            "<br>&nbsp;&nbsp;<b>18 raise-OK with 0 matches</b> — produced "
            "linalg.generic but the matcher's template library has no "
            "entries for pooling, batchnorm, LRN, residual-add, RNN gates, "
            "transposed conv, locally-connected layers, dense+bias, etc. "
            "<i>This is the actionable list: each is a matcher template "
            "we could add to expand CNN coverage.</i>"
            "<br>&nbsp;&nbsp;<b>5 raise-failed</b> — cgeist OK but the "
            "raise pass chokes (batchnorm_layer, convolutional_layer, box, "
            "demo, tree). convolutional_layer.c is the painful one because "
            "its body is mostly external-call dispatch (to im2col_cpu + "
            "gemm); the actual gemm work lives in <code>gemm.c</code> which "
            "does match."
            "<br>&nbsp;&nbsp;<b>17 cgeist-failed</b> — framework code "
            "(parser, network, image, data, list, utils, ...) plus a few "
            "layers with IfStmt lowering or function-pointer-dispatch "
            "patterns cgeist can't handle. Most of these don't have "
            "matchable compute anyway."
            "<br><br>"
            "darknet's actual hot path uses <code>gemm_nn</code> (TA=TB=0). "
            "The matcher hits it as <code>@cublasDaxpy</code> (the inner "
            "loop has a scalar-hoisted axpy shape) but doesn't compose the "
            "outer two loops back into gemm. <code>gemm_nt</code> and "
            "<code>gemm_tt</code> use the conventional sum-accumulator form "
            "and match as <code>@cublasDgemm_alpha_only</code> cleanly. "
            "Fixing the gemm_nn composition is a high-value matcher "
            "improvement target — it would auto-cover every conv layer "
            "darknet runs at inference time."
        ),
        kernel_stats=darknet_stats,
        notes=DARKNET_NOTES,
        blockers=DARKNET_BLOCKERS,
    )

    def nav() -> str:
        return (
            '<div class="header"><h1><a href="index.html">'
            'Polygeist IR explorer</a></h1>'
            '<div style="margin-top:6px; font-size:13px;">'
            '<a href="index.html">Overview</a> &middot; '
            '<a href="polybench.html">PolyBench results</a> &middot; '
            '<a href="backends.html">CPU/GPU lowering</a> &middot; '
            '<a href="numerical.html">ATen</a> &middot; '
            '<a href="performance.html">Performance analysis</a> &middot; '
            '<a href="modified-kernels.html">Modified kernels</a> &middot; '
            '<a href="mfem.html">MFEM</a> &middot; '
            '<a href="ginsbach.html">Ginsbach ASPLOS\'18</a> &middot; '
            '<a href="ai.html">AI kernels</a> &middot; '
            '<a href="vision.html">Vision + fusion</a> &middot; '
            '<a href="pva.html">PVA backend</a>'
            '</div></div>'
        )

    def card(href: str, title: str, count: int, description: str) -> str:
        return (
            f'<a class="suite-card" href="{href}">'
            f'<b>{title}</b><span>{count} tracked rows</span>'
            f'<small>{description}</small></a>'
        )

    extra_css = (
        '.section-header { background: #eaeefa; padding: 8px 20px; '
        'border-top: 2px solid #c4cce0; border-bottom: 1px solid #c4cce0; '
        'margin-top: 24px; } '
        '.section-title { margin: 0; font-size: 16px; color: #1f2d3d; } '
        '.suite-grid { display:grid; grid-template-columns:repeat(auto-fit, '
        'minmax(240px,1fr)); gap:14px; padding:18px 20px; max-width:1100px; } '
        '.suite-card { border:1px solid #d8dee8; border-radius:8px; padding:16px; '
        'text-decoration:none; color:#1f2d3d; background:#fafbfc; } '
        '.suite-card:hover { border-color:#7b91bd; background:#f3f6fc; } '
        '.suite-card b,.suite-card span,.suite-card small { display:block; } '
        '.suite-card span { color:#1a7f37; margin-top:5px; font-size:13px; } '
        '.suite-card small { color:#555; margin-top:8px; line-height:1.35; } '
        '.audit-metrics { display:grid; grid-template-columns:repeat(auto-fit, '
        'minmax(145px,1fr)); gap:10px; margin:10px 20px; max-width:1050px; } '
        '.audit-metric { border:1px solid #d8dee8; border-radius:7px; padding:12px; '
        'background:#fafbfc; } .audit-metric b { display:block; color:#1a7f37; '
        'font-size:22px; } .audit-metric span { color:#555; font-size:12px; } '
        '.audit-table { font-size:12px; } .audit-table td { white-space:nowrap; } '
        '.cause-tag { display:inline-block; border-radius:10px; padding:2px 7px; '
        'font-size:11px; font-weight:bold; margin-bottom:4px; } '
        '.cause-memory { background:#ffd9d9; color:#8b1a1a; } '
        '.cause-host { background:#eadcff; color:#53258a; } '
        '.cause-copy { background:#dcecff; color:#174f86; } '
        '.cause-intensity { background:#ffe8c7; color:#7a4300; } '
        '.cause-setup { background:#fff3bd; color:#705900; } '
        '.cause-bandwidth { background:#ffe0ec; color:#842347; } '
        '.cause-amortized { background:#dff5e5; color:#1a6a34; } '
        '.mod-tag { display:inline-block; border-radius:10px; padding:2px 7px; '
        'font-size:11px; font-weight:bold; background:#fff3bd; color:#705900; '
        'margin-bottom:4px; } '
        '.modified-table td { vertical-align:top; min-width:120px; } '
        '.modified-table td:nth-child(2) { min-width:175px; } '
        '.modified-table td:nth-child(5),.modified-table td:nth-child(6),'
        '.modified-table td:nth-child(7) { min-width:220px; white-space:normal; } '
        '.columns { columns:3 300px; } .columns li { break-inside:avoid; '
        'margin-bottom:3px; } '
        '.backend-flow { display:grid; grid-template-columns:minmax(260px,1fr) 30px '
        'minmax(260px,1fr) 30px minmax(260px,1fr); gap:8px; align-items:center; '
        'padding:16px 20px; background:#f4f7fb; } '
        '.backend-common,.backend-card { padding:12px; border:1px solid #c8d2e2; '
        'border-radius:6px; background:white; font-size:12px; line-height:1.55; } '
        '.backend-common { background:#eef4ff; } .backend-arrow { text-align:center; '
        'font-size:22px; color:#65758b; } '
        '@media(max-width:900px) { .backend-flow { display:block; } '
        '.backend-arrow { display:none; } .backend-common,.backend-card { margin:7px 0; } }'
    )

    landing = (
        nav()
        + '<div class="intro"><b>Raising and library-matching tracker.</b> '
          'The explorer is split into focused pages so the large Compiler '
          'Explorer deep-links are loaded only for the suite being inspected. '
          'Each kernel still has a static IR preview and a full CE link.</div>'
        + '<div class="suite-grid">'
        + card("polybench.html", "PolyBench four-runtime results", len(polybench_stats),
               "Native CPU, raised CPU-library, native CUDA, and raised CUDA timings with correctness gates.")
        + card("backends.html", "CPU + GPU lowering",
               sum(s.get("launches", 0) > 0 for s in polybench_stats.values()),
               "Shared ABI, backend branch point, and implementation coverage.")
        + card("numerical.html", "ATen numerical kernels", len(aten_stats),
               "Extracted ATen C algorithms and Jetson comparisons.")
        + card("performance.html", "Why are some kernels slow?",
               sum(row.get("correctness") == "PASS"
                   for row in _read_csv(ATEN_SILICON_RESULTS)),
               "Root-cause groups, highlighted slowdown ratios, and a GEMV deep dive.")
        + card("modified-kernels.html", "Modified kernels", modified_count,
               "Extracted or normalized sources, why direct compilation was not used, "
               "and whether the cause is frontend, raising, or framework structure.")
        + card("mfem.html", "MFEM finite elements",
               len(mfem_stats) + len(mfem_application_stats)
               + len(mfem_application_extraction_stats),
               "Original/normalized FEM kernels and larger application hot paths.")
        + card("ginsbach.html", "Ginsbach ASPLOS'18", ginsbach_count,
               "103/103 units raised; 23 compute launches and 4 silicon-validated program rows.")
        + card("ai.html", "AI kernels",
               len(llama_forward_stats) + len(whisper_ops_stats) + len(llmc_stats),
               "Llama forward, Whisper/ggml, and llm.c forward/backward kernels.")
        + card("vision.html", "Vision + fusion",
               len(stencil_conv2d_stats) + len(darknet_stats)
               + len(ex_darknet_stats) + len(fopt_stats),
               "Stencil Conv2D, darknet, extracted CNN blocks, and fusion experiments.")
        + card("pva.html", "PVA backend", len(PVA_KERNELS),
               "PVA lowering coverage and executable backend experiments.")
        + '</div>'
        + _build_taxonomy_panel()
    )
    backends = nav() + _backend_overview(polybench_stats)
    performance = nav() + _aten_slowness_page(aten_stats)
    modified = nav() + modified_body
    numerical_pages: dict[str, str] = {}
    for sort_by in ("alphabetical", "raised", "resident"):
        ordered = _aten_sorted_kernels(sort_by)
        page_count = max(1, (len(ordered) + ATEN_PAGE_SIZE - 1) // ATEN_PAGE_SIZE)
        for page in range(1, page_count + 1):
            begin = (page - 1) * ATEN_PAGE_SIZE
            subset = ordered[begin:begin + ATEN_PAGE_SIZE]
            filename = _aten_page_filename(sort_by, page)
            numerical_pages[filename] = render_html(
                "Polygeist: ATen numerical kernels",
                nav() + _aten_section(
                    aten_stats, subset, sort_by, page, page_count
                ),
                extra_css,
            )
    mfem = (nav()
            + _mfem_application_extraction_section(
                mfem_application_extraction_stats
            )
            + _mfem_application_section(mfem_application_stats)
            + _mfem_section(mfem_stats))
    ginsbach = nav() + ginsbach_body
    ai = nav() + llama_forward_section + whisper_ops_section + llmc_section
    vision = (
        nav() + stencil_conv2d_section + darknet_section
        + _extracted_darknet_section(ex_darknet_stats)
        + _fusion_opt_section(fopt_stats)
    )
    pva = nav() + _pva_section()
    pages = {
        "index.html": render_html("Polygeist IR explorer", landing, extra_css),
        "backends.html": render_html(
            "Polygeist: CPU and GPU lowering", backends, extra_css
        ),
        "performance.html": render_html(
            "Polygeist: kernel slowness analysis", performance, extra_css
        ),
        "modified-kernels.html": render_html(
            "Polygeist: modified and extracted kernels", modified, extra_css
        ),
        "mfem.html": render_html("Polygeist: MFEM kernels", mfem, extra_css),
        "ginsbach.html": render_html(
            "Polygeist: Ginsbach ASPLOS'18 audit", ginsbach, extra_css
        ),
        "ai.html": render_html("Polygeist: AI kernels", ai, extra_css),
        "vision.html": render_html(
            "Polygeist: vision + fusion", vision, extra_css
        ),
        "pva.html": render_html("Polygeist: PVA backend", pva, extra_css),
    }
    pages.update(numerical_pages)
    return pages


def main():
    mfem_only = "--mfem-only" in sys.argv[1:]
    aten_only = "--aten-only" in sys.argv[1:]
    polybench_only = "--polybench-only" in sys.argv[1:]
    polybench_results_only = "--polybench-results-only" in sys.argv[1:]
    ginsbach_only = "--ginsbach-only" in sys.argv[1:]
    unknown_args = [
        arg for arg in sys.argv[1:]
        if arg not in (
            "--mfem-only", "--aten-only", "--polybench-only",
            "--polybench-results-only",
            "--ginsbach-only",
        )
    ]
    if unknown_args:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown_args)}")
    if sum((mfem_only, aten_only, polybench_only,
            polybench_results_only, ginsbach_only)) > 1:
        raise SystemExit("suite-only arguments are mutually exclusive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if polybench_results_only:
        write_polybench_results_page()
        for obsolete in ("polybenchgpu.html", "polybench-section42.html"):
            OUTPUT_DIR.joinpath(obsolete).unlink(missing_ok=True)
        print(f"Done. Open {OUTPUT_DIR}/polybench.html.")
        return
    if ginsbach_only:
        pages = build_site_pages(
            {}, {}, [], [], [], {}, {}, {}, {}, {}, {}, {},
        )
        OUTPUT_DIR.joinpath("ginsbach.html").write_text(
            pages["ginsbach.html"]
        )
        print(f"Done. Open {OUTPUT_DIR}/ginsbach.html.")
        return
    if mfem_only:
        for stale in OUTPUT_DIR.glob("mfem_*.html"):
            stale.unlink()
        print("Rendering MFEM original and normalized kernels...", flush=True)
        mfem_stats = build_mfem_pages()
        mfem_application_stats = build_mfem_application_pages()
        mfem_application_extraction_stats = (
            build_mfem_application_extraction_pages()
        )
        pages = build_site_pages(
            {}, {}, mfem_stats, mfem_application_stats,
            mfem_application_extraction_stats, {}, {}, {}, {}, {}, {}, {},
        )
        OUTPUT_DIR.joinpath("mfem.html").write_text(pages["mfem.html"])
        print(
            f"  [MFEM] rendered {len(mfem_stats)} kernels and "
            f"{len(mfem_application_stats)} application ports and "
            f"{len(mfem_application_extraction_stats)} larger application paths",
            flush=True,
        )
        print(f"Done. Open {OUTPUT_DIR}/mfem.html.")
        return
    if aten_only:
        for stale in OUTPUT_DIR.glob("aten_*.html"):
            stale.unlink()
        aten_stats = {}
        print(f"Rendering {len(ATEN_C_ORDER)} ATen C kernels...", flush=True)
        for i, kernel in enumerate(ATEN_C_ORDER, 1):
            print(
                f"  [ATEN {i:2d}/{len(ATEN_C_ORDER)}] {kernel}",
                flush=True,
            )
            has_any = any(
                (ATEN_C_MLIR_DIR / f"{kernel}{suffix}").exists()
                for suffix in (".mlir", "_linalg.mlir", "_debuf.mlir")
            )
            if not has_any:
                aten_stats[kernel] = {
                    "launches": 0, "linalg_ops": 0, "matched_symbols": [],
                    "residual": 0, "residual_for": 0, "ce_url": None,
                    "page_filename": "",
                }
                continue
            aten_stats[kernel] = build_kernel_page(
                kernel, mlir_dir=ATEN_C_MLIR_DIR,
                kset="aten_c", file_prefix="",
            )
        build_aten_c_source_pages(aten_stats)
        pages = build_site_pages(
            {}, aten_stats, [], [], [], {}, {}, {}, {}, {}, {}, {},
        )
        for filename, page_html in pages.items():
            if filename.startswith("numerical") or filename == "performance.html":
                OUTPUT_DIR.joinpath(filename).write_text(page_html)
        print(f"Done. Open {OUTPUT_DIR}/numerical.html.")
        return
    if polybench_only:
        polybench_kernels = discover_kernels(MLIR_DIR)
        polybench_stats = {}
        print(
            f"Rendering {len(polybench_kernels)} PolyBench kernels...",
            flush=True,
        )
        for i, kernel in enumerate(polybench_kernels, 1):
            print(
                f"  [PB {i:2d}/{len(polybench_kernels)}] {kernel}",
                flush=True,
            )
            polybench_stats[kernel] = build_kernel_page(
                kernel, mlir_dir=MLIR_DIR,
                kset="polybench", file_prefix="",
            )
        pages = build_site_pages(
            polybench_stats, {}, [], [], [], {}, {}, {}, {}, {}, {}, {},
        )
        OUTPUT_DIR.joinpath("backends.html").write_text(pages["backends.html"])
        write_polybench_results_page()
        for obsolete in ("polybenchgpu.html", "polybench-section42.html"):
            OUTPUT_DIR.joinpath(obsolete).unlink(missing_ok=True)
        print(
            f"Done. Open {OUTPUT_DIR}/polybench.html or "
            f"{OUTPUT_DIR}/backends.html."
        )
        return
    for stale in OUTPUT_DIR.glob("llama_*.html"):
        stale.unlink()
    for stale in OUTPUT_DIR.glob("whisper_*.html"):
        stale.unlink()
    for stale in OUTPUT_DIR.glob("aten_*.html"):
        stale.unlink()
    for stale in OUTPUT_DIR.glob("mfem_*.html"):
        stale.unlink()

    # PolyBench set.
    pb_kernels = discover_kernels(MLIR_DIR)
    print(f"Rendering {len(pb_kernels)} PolyBench kernels...", flush=True)
    pb_stats = {}
    for i, k in enumerate(pb_kernels, 1):
        print(f"  [PB {i:2d}/{len(pb_kernels)}] {k}", flush=True)
        pb_stats[k] = build_kernel_page(k, mlir_dir=MLIR_DIR,
                                         kset="polybench", file_prefix="")

    # Standalone C extractions of representative ATen numerical kernels.
    aten_stats = {}
    print(f"Rendering {len(ATEN_C_ORDER)} ATen C kernels...", flush=True)
    for i, k in enumerate(ATEN_C_ORDER, 1):
        print(f"  [ATEN {i:2d}/{len(ATEN_C_ORDER)}] {k}", flush=True)
        has_any = any((ATEN_C_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir"))
        if not has_any:
            aten_stats[k] = {
                "launches": 0, "linalg_ops": 0, "matched_symbols": [],
                "residual": 0, "residual_for": 0, "ce_url": None,
                "page_filename": "",
            }
            continue
        aten_stats[k] = build_kernel_page(
            k, mlir_dir=ATEN_C_MLIR_DIR, kset="aten_c", file_prefix="",
        )
    build_aten_c_source_pages(aten_stats)

    # MFEM finite-element extractions use a manifest-driven artifact layout.
    print("Rendering MFEM original and normalized kernels...", flush=True)
    mfem_stats = build_mfem_pages()
    print(f"  [MFEM] rendered {len(mfem_stats)} kernels", flush=True)
    mfem_application_stats = build_mfem_application_pages()
    print(
        f"  [MFEM applications] rendered {len(mfem_application_stats)} ports",
        flush=True,
    )
    mfem_application_extraction_stats = build_mfem_application_extraction_pages()
    print(
        "  [MFEM larger applications] rendered "
        f"{len(mfem_application_extraction_stats)} paths",
        flush=True,
    )

    # Llama forward fixtures extracted as C benchmarks.
    llama_forward_kernels_from_files = discover_kernels(LLAMA_FORWARD_MLIR_DIR)
    llama_forward_kernel_set = (
        set(llama_forward_kernels_from_files) | set(LLAMA_FORWARD_KERNELS.keys())
    )
    llama_forward_kernels = [
        k for k in LLAMA_FORWARD_ORDER if k in llama_forward_kernel_set
    ]
    llama_forward_kernels += sorted(
        k for k in llama_forward_kernel_set if k not in set(LLAMA_FORWARD_ORDER)
    )
    print(f"Rendering {len(llama_forward_kernels)} Llama forward fixture kernels...", flush=True)
    llama_forward_stats = {}
    for i, k in enumerate(llama_forward_kernels, 1):
        print(f"  [LLAMA-FWD {i:2d}/{len(llama_forward_kernels)}] {k}", flush=True)
        has_any = any((LLAMA_FORWARD_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            llama_forward_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                                      "ce_url": None, "page_filename": ""}
            continue
        llama_forward_stats[k] = build_kernel_page(
            k, mlir_dir=LLAMA_FORWARD_MLIR_DIR, kset="llama_forward",
            file_prefix="llamafwd_",
        )
        # The paper's extended-forward row deliberately excludes four
        # project-authored computational helpers (mask, two adds, and
        # SwiGLU).  Keep its displayed denominator aligned with the audited
        # external-library-only run instead of the broader exploratory match.
        if k == "extended_forward":
            llama_forward_stats[k]["launches"] = 13
            llama_forward_stats[k]["residual"] = 32

    # Whisper/ggml-style extracted operation fixtures.
    whisper_ops_kernels_from_files = discover_kernels(WHISPER_OPS_MLIR_DIR)
    whisper_ops_kernel_set = (
        set(whisper_ops_kernels_from_files) | set(WHISPER_OPS_KERNELS.keys())
    )
    whisper_ops_kernels = [
        k for k in WHISPER_OPS_ORDER if k in whisper_ops_kernel_set
    ]
    whisper_ops_kernels += sorted(
        k for k in whisper_ops_kernel_set if k not in set(WHISPER_OPS_ORDER)
    )
    print(f"Rendering {len(whisper_ops_kernels)} Whisper extracted kernels...", flush=True)
    whisper_ops_stats = {}
    for i, k in enumerate(whisper_ops_kernels, 1):
        print(f"  [WHISPER {i:2d}/{len(whisper_ops_kernels)}] {k}", flush=True)
        has_any = any((WHISPER_OPS_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            whisper_ops_stats[k] = {"launches": 0, "residual": 0,
                                    "residual_for": 0, "ce_url": None,
                                    "page_filename": ""}
            continue
        whisper_ops_stats[k] = build_kernel_page(
            k, mlir_dir=WHISPER_OPS_MLIR_DIR, kset="whisper_ops",
            file_prefix="",
        )

    # Non-DL stencil fixtures that map to cuDNN 3x3 convolution.
    # This directory also contains scratch artifacts produced by the local
    # smoke tests (`*_matched.mlir`, `*_lowered.mlir`). Keep the website to
    # the explicit fixture list so those files do not become bogus rows.
    stencil_conv2d_kernels = list(STENCIL_CONV2D_ORDER)
    print(f"Rendering {len(stencil_conv2d_kernels)} stencil Conv2D kernels...", flush=True)
    stencil_conv2d_stats = {}
    for i, k in enumerate(stencil_conv2d_kernels, 1):
        print(f"  [STENCIL-CONV2D {i:2d}/{len(stencil_conv2d_kernels)}] {k}", flush=True)
        has_any = any((STENCIL_CONV2D_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            stencil_conv2d_stats[k] = {"launches": 0, "residual": 0,
                                       "residual_for": 0, "ce_url": None,
                                       "page_filename": ""}
            continue
        stencil_conv2d_stats[k] = build_kernel_page(
            k, mlir_dir=STENCIL_CONV2D_MLIR_DIR, kset="stencil_conv2d",
            file_prefix="stencilconv_",
        )

    # llm.c set.
    llmc_kernels_from_files = discover_kernels(LLMC_MLIR_DIR)
    llmc_kernels = sorted(set(llmc_kernels_from_files) | set(LLMC_KERNELS.keys()))
    print(f"Rendering {len(llmc_kernels)} llm.c kernels...", flush=True)
    llmc_stats = {}
    for i, k in enumerate(llmc_kernels, 1):
        print(f"  [LLMC {i:2d}/{len(llmc_kernels)}] {k}", flush=True)
        has_any = any((LLMC_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            llmc_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                              "ce_url": None, "page_filename": ""}
            continue
        llmc_stats[k] = build_kernel_page(
            k, mlir_dir=LLMC_MLIR_DIR, kset="llmc",
            file_prefix="llmc_",
        )

    # darknet (full-source bake). The kernel "name" is each .c file's
    # basename; bake_darknet_mlir.sh emits <name>.mlir + <name>_linalg.mlir
    # + <name>_debuf.mlir using the same naming convention the explorer
    # expects, so build_kernel_page reads them transparently.
    darknet_kernels_from_files = discover_kernels(DARKNET_MLIR_DIR)
    darknet_kernels = sorted(set(darknet_kernels_from_files) | set(DARKNET_KERNELS.keys()))
    print(f"Rendering {len(darknet_kernels)} darknet kernels...", flush=True)
    darknet_stats = {}
    for i, k in enumerate(darknet_kernels, 1):
        print(f"  [DARKNET {i:2d}/{len(darknet_kernels)}] {k}", flush=True)
        has_any = any((DARKNET_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            darknet_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                                 "ce_url": None, "page_filename": ""}
            continue
        darknet_stats[k] = build_kernel_page(
            k, mlir_dir=DARKNET_MLIR_DIR, kset="darknet",
            file_prefix="darknet_",
        )

    # extracted-darknet (polybench-style CNN block kernels for the cuDNN
    # runtime pipeline). Same per-kernel-page machinery as the other
    # sections — bake_extracted_darknet_mlir.sh produces the per-stage
    # MLIR files in /tmp/extracted_darknet_mlir/ that build_kernel_page
    # consumes.
    ex_darknet_kernels = sorted(EXTRACTED_DARKNET_KERNELS.keys())
    print(f"Rendering {len(ex_darknet_kernels)} extracted-darknet kernels...", flush=True)
    ex_darknet_stats = {}
    for i, k in enumerate(ex_darknet_kernels, 1):
        print(f"  [EXTRACTED-DARKNET {i:1d}/{len(ex_darknet_kernels)}] {k}", flush=True)
        has_any = any((EXTRACTED_DARKNET_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir"))
        if not has_any:
            ex_darknet_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                                    "ce_url": None, "page_filename": ""}
            continue
        ex_darknet_stats[k] = build_kernel_page(
            k, mlir_dir=EXTRACTED_DARKNET_MLIR_DIR, kset="extracted_darknet",
            file_prefix="exdark_",
        )

    # Fusion-optimization kernels (algebraic rewrites: conv+bias+relu+add,
    # gemm+bias+relu, AᵀA→syrk, 1×1 conv → batched gemm). Same per-stage
    # MLIR bake pipeline as extracted_darknet.
    fopt_kernel_list = sorted(FUSION_OPT_KERNELS.keys())
    print(f"Rendering {len(fopt_kernel_list)} fusion-optimization kernels...", flush=True)
    fopt_stats = {}
    for i, k in enumerate(fopt_kernel_list, 1):
        print(f"  [FUSION-OPT {i:1d}/{len(fopt_kernel_list)}] {k}", flush=True)
        has_any = any((EXTRACTED_DARKNET_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir"))
        if not has_any:
            fopt_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                              "ce_url": None, "page_filename": ""}
            continue
        fopt_stats[k] = build_kernel_page(
            k, mlir_dir=EXTRACTED_DARKNET_MLIR_DIR, kset="fusion_opt",
            file_prefix="fopt_",
        )

    pages = build_site_pages(
        pb_stats, aten_stats, mfem_stats, mfem_application_stats,
        mfem_application_extraction_stats,
        llama_forward_stats, whisper_ops_stats,
        stencil_conv2d_stats, llmc_stats, darknet_stats, ex_darknet_stats,
        fopt_stats,
    )
    for filename, html in pages.items():
        OUTPUT_DIR.joinpath(filename).write_text(html)
    write_polybench_results_page()
    for obsolete in ("polybenchgpu.html", "polybench-section42.html"):
        OUTPUT_DIR.joinpath(obsolete).unlink(missing_ok=True)
    print(f"\nWrote {len(pages)} explorer pages.")
    print(f"Done. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
