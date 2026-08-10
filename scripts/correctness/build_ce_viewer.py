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
REWRITER = env_path("POLYGEIST_KERNEL_MATCH_REWRITER", SCRIPT_DIR / "kernel_match_rewrite.py")
PYTHON = os.environ.get("PYTHON", sys.executable)

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
    "aten_adaptive_avg_pool2d": "no average-pooling definition",
    "aten_adaptive_avg_pool3d": "no 3D average-pooling definition",
    "aten_avg_pool2d": "no average-pooling definition",
    "aten_avg_pool3d": "no 3D average-pooling definition",
    "aten_binary_cross_entropy": "external logf calls retain a residual loop",
    "aten_bmm": "no true batched-GEMM definition",
    "aten_channel_shuffle": "layout transform; correctly rejected as flat copy",
    "aten_clamp": "no standalone clamp definition",
    "aten_conv1d": "no 1D-convolution definition",
    "aten_conv3d": "shape/template gap in current 3D-convolution definitions",
    "aten_conv_transpose2d": "no transposed-convolution definition",
    "aten_cross": "three elementwise stages; no cross-product composition",
    "aten_cumsum": "loop-carried scan raises, but no scan definition",
    "aten_elu": "no standalone ELU definition",
    "aten_embedding": "indexed gather retains residual loops",
    "aten_gelu": "valid custom CUDA GELU route",
    "aten_hardsigmoid": "no standalone hard-sigmoid definition",
    "aten_hardswish": "no standalone hard-swish definition",
    "aten_hardtanh": "no standalone hard-tanh definition",
    "aten_im2col": "window gather correctly rejected as flat copy",
    "aten_l1_loss": "no L1 reduction composition",
    "aten_layer_norm": "mean/variance/affine composition not in library",
    "aten_leaky_relu": "no standalone leaky-ReLU definition",
    "aten_lerp": "no linear-interpolation definition",
    "aten_mean": "reduction-plus-scale composition not in library",
    "aten_mse_loss": "square-difference plus mean composition not in library",
    "aten_outer": "partial: only output zero-initialization matched",
    "aten_pixel_shuffle": "layout transform correctly rejected as flat copy",
    "aten_prod": "no product-reduction definition",
    "aten_reflection_pad2d": "no reflection-padding definition",
    "aten_relu": "no standalone ReLU definition",
    "aten_replication_pad2d": "no replication-padding definition",
    "aten_sigmoid": "no standalone sigmoid definition",
    "aten_silu": "no standalone SiLU definition",
    "aten_softplus": "external logf/expf and branch retain a residual loop",
    "aten_sum": "partial: only output zero-initialization matched",
    "aten_tanh": "no standalone tanh definition",
    "aten_transpose_copy": "permuted indexing map correctly rejected as flat copy",
    "aten_upsample_bilinear2d": "no bilinear-resampling definition",
    "aten_upsample_nearest2d": "no nearest-neighbor resampling definition",
}

ATEN_C_UNSAFE_MATCHES: set[str] = {"aten_pixel_shuffle"}

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
                      "L·Lᵀ factorization — outer k column update carries "
                      "dependency to all later columns; small inner parallelism"),
    "lu":            ("serial",
                      "LU factorization — same column-sequential pattern as cholesky"),
    "ludcmp":        ("serial",
                      "LU + forward/back substitution — substitution phase is "
                      "strictly sequential"),
    "gramschmidt":   ("serial",
                      "modified Gram-Schmidt — each column projects against ALL "
                      "previously orthogonalized columns; strictly sequential"),
    "trisolv":       ("serial",
                      "triangular solve — y[i] depends on y[0..i-1]; sequential "
                      "row-by-row"),
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
    "cholesky":      ("serial-recurrence", "lower-triangular factorization — column k modifies columns 0..k-1, k+1..N-1 depends on them"),
    "gramschmidt":   ("serial-recurrence", "column-by-column modified Gram-Schmidt — column k+1 reads what column k just wrote"),
    "lu":            ("serial-recurrence", "LU factorization — pivot row k modifies rows >k that subsequent iterations consume"),
    "trisolv":       ("serial-recurrence", "triangular solve — y[i] depends on y[0..i-1]"),
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

# Warmed in-process comparison against handwritten PolyBenchGPU CUDA kernels.
# Method: Jetson Orin, N/NI/NJ/NK/NL/NM=512, double precision, 50 iterations
# in a single process, discard the first 10 warmup iterations, then report a
# 10% trimmed mean over the remaining 40 samples. Raised numbers are summed
# device-event timings from the runtime shims; PolyBenchGPU numbers are CUDA
# event timings around the handwritten kernel sequence. CPU comparison is
# intentionally not rendered in the PolyBench tracker for now.
POLYBENCHGPU_RUNTIMES: dict[str, list[dict]] = {
    "gemm": [
        {"size": "512 warmed", "raised_ms": 3.808535, "pbgpu_ms": 7.696930,
         "notes": "Raised path uses cuBLAS dgemm; first cuBLAS cold-start iteration discarded"},
    ],
    "2mm": [
        {"size": "512 warmed", "raised_ms": 7.639525, "pbgpu_ms": 11.200252,
         "notes": "Raised path is two warmed cuBLAS dgemms plus host helper ops"},
    ],
    "3mm": [
        {"size": "512 warmed", "raised_ms": 11.451146, "pbgpu_ms": 10.500537,
         "notes": "Only current warmed case where handwritten PolyBenchGPU is slightly faster"},
    ],
    "gesummv": [
        {"size": "512 warmed", "raised_ms": 0.069274, "pbgpu_ms": 0.341379,
         "notes": "Raised path is two warmed cuBLAS gemv calls plus host axpby"},
    ],
    "gemver": [
        {"size": "512 warmed", "raised_ms": 0.188384, "pbgpu_ms": 0.312846,
         "notes": "Raised path is warmed ger/gemv/axpy sequence"},
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
        {"size": "7B-size one layer warm", "raised": "host 13.480 ms<br>device 12.273 ms",
         "reference": "ggml CUDA host 9.638 ms", "winner": "ggml 1.40x",
         "notes": "MODEL_DIM=4096, FFN_DIM=11008, VOCAB=32000, SEQ_LEN=2048, HEADS=32; one layer only"},
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
    "extended_forward":       ("none", "Full fixture emits 34 runtime calls after lowering and matches native C logits on Jetson; it uses split RoPE and branchless mask to stay inside today's raising envelope."),
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
    res = subprocess.run(
        [PYTHON, str(REWRITER), str(path)],
        capture_output=True, text=True, timeout=10,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"kernel matcher failed for {path} with {PYTHON}:\n{res.stderr}"
        )
    out = res.stdout
    n_launch = len(re.findall(r"kernel\.launch", out))
    n_lg = len(re.findall(r"linalg\.generic", out))
    return out, [("launches", n_launch), ("residual_lg", n_lg)]


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
    matched_symbols: list[str] = []
    report = [("launches", 0), ("residual_lg", 0)]

    if cgeist_mlir.exists():
        cgeist_text = cgeist_mlir.read_text()
        html, css = syntax_highlight(cgeist_text)
        pages["cgeist"] = html
        if not raised.exists() and not debuf.exists() and not debuf_mr.exists():
            n_for = count_for_loops(cgeist_text)
            report = [
                ("launches", 0),
                ("residual_lg", len(re.findall(r"linalg\.generic", cgeist_text))),
            ]
    if raised.exists():
        raised_text = raised.read_text()
        n_linalg = len(re.findall(r"\blinalg\.generic\b", raised_text))
        html, css = syntax_highlight(raised_text)
        pages["raised"] = html
        if not debuf.exists() and not debuf_mr.exists():
            n_for = count_for_loops(raised_text)
            report = [
                ("launches", 0),
                ("residual_lg", len(re.findall(r"linalg\.generic", raised_text))),
            ]
        if kset == "stencil_conv2d" and not debuf.exists():
            n_for = count_for_loops(raised_text)
            rewritten, report = run_rewriter(raised)
            html, css = syntax_highlight(rewritten)
            pages["matched"] = html
    if debuf.exists():
        debuf_text = debuf.read_text()
        n_for = count_for_loops(debuf_text)
        html, css = syntax_highlight(debuf_text)
        pages["debuf"] = html
        # The exhaustive ATen sweep stores the authoritative matcher output
        # beside its diagnostics. Reuse it instead of starting one Egglog
        # process per page (hundreds of avoidable process launches).
        stored_match = mlir_dir / kernel / "matched.mlir"
        if kset == "aten_c" and stored_match.exists():
            rewritten = stored_match.read_text()
            report = [
                ("launches", len(re.findall(r"kernel\.launch\s+@", rewritten))),
                ("residual_lg", len(re.findall(r"\blinalg\.generic\b", rewritten))),
            ]
        else:
            rewritten, report = run_rewriter(debuf)
        matched_symbols = sorted(set(
            re.findall(r"kernel\.launch\s+@([A-Za-z0-9_]+)", rewritten)
        ))
        html, css = syntax_highlight(rewritten)
        pages["matched"] = html
    if debuf_mr.exists():
        debuf_mr_text = debuf_mr.read_text()
        html, css = syntax_highlight(debuf_mr_text)
        pages["debuf_mr"] = html
        # Fallback: if v2 debuf failed but multi-root succeeded (the
        # common pattern for whole-program-raise suites),
        # run the matcher on the multi-root output so the "matched" tab
        # and the match-status column reflect what's actually achievable.
        if not debuf.exists() and not debuf_mr_text.lstrip().startswith("//"):
            n_for = count_for_loops(debuf_mr_text)
            rewritten, report = run_rewriter(debuf_mr)
            matched_symbols = sorted(set(
                re.findall(r"kernel\.launch\s+@([A-Za-z0-9_]+)", rewritten)
            ))
            html, css = syntax_highlight(rewritten)
            pages["matched"] = html

    ce_url = ce_link(kernel, mlir_dir=mlir_dir, kset=kset)
    open_link = (f'<a href="{ce_url}" target="_blank" '
                 f'style="margin-left:12px; color:#0366d6;">'
                 f'open in Compiler Explorer →</a>') if ce_url else ''

    n_launches = report[0][1]
    n_resid = report[1][1]
    summary = (
        f'<div class="summary" style="padding:8px 20px; '
        f'border-bottom:1px solid #eee; background:#fafafa; font-size:13px;">'
        f'<b>{n_launches}</b> kernel.launch op(s) emitted &nbsp;·&nbsp; '
        f'<b>{n_resid}</b> residual linalg.generic &nbsp;·&nbsp; '
        f'<b>{n_for}</b> residual for-loop(s) &nbsp;|&nbsp; '
        f'jump to: <a href="#cgeist">cgeist</a> · '
        f'<a href="#raised">raised</a> · '
        f'<a href="#debuf">debuferized</a> · '
        f'<a href="#debuf_mr">debuf multi-root</a> · '
        f'<a href="#matched">kernel.launch output</a>'
        f'</div>'
    )
    back_href, back_label = "index.html", "index"
    if kset == "polybench":
        back_href, back_label = "polybench.html", "PolyBench"
    elif kset == "aten_c":
        back_href, back_label = "numerical.html", "ATen"
    header = (
        f'<div class="header"><h1><a href="{back_href}">← {back_label}</a> '
        f'&nbsp; {kernel}{open_link}</h1></div>'
        + summary
    )
    body_blocks = []
    for stage, title in [
        ("cgeist",   "cgeist output (pre-raise MLIR)"),
        ("raised",   "raised (memref linalg, before debuferize)"),
        ("debuf",    "debuferized (tensor linalg, matcher input)"),
        ("debuf_mr", "debuferized — multi-root (--linalg-debufferize=use-multi-root=true)"),
        ("matched",  "kernel.launch (matcher output)"),
    ]:
        if stage not in pages:
            continue
        body_blocks.append(
            f'<h2 id="{stage}">{title}</h2>'
            f'<div class="container">{pages[stage]}</div>'
        )
    body = header + "\n".join(body_blocks)
    OUTPUT_DIR.joinpath(f"{file_prefix}{kernel}.html").write_text(render_html(kernel, body, css))
    return {
        "launches": report[0][1],
        "linalg_ops": n_linalg,
        "matched_symbols": matched_symbols,
        "residual": report[1][1],
        "residual_for": n_for,
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


def _aten_page_filename(sort_by: str, page: int) -> str:
    prefix = "numerical" if sort_by == "alphabetical" else "numerical-correctness"
    return f"{prefix}.html" if page == 1 else f"{prefix}-{page}.html"


def _aten_sorted_kernels(sort_by: str) -> list[str]:
    if sort_by == "alphabetical":
        return sorted(ATEN_C_ORDER)
    performance = {
        row.get("kernel", ""): row for row in _read_csv(ATEN_SILICON_RESULTS)
    }
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
    performance = {
        row.get("kernel", ""): row for row in _read_csv(ATEN_SILICON_RESULTS)
    }
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
        launches = stats.get("launches", 0)
        residual = stats.get("residual", 0)
        loops = stats.get("residual_for", 0)
        linalg_ops = stats.get("linalg_ops", 0)
        if linalg_ops > 0 and loops == 0:
            raise_status_class, raise_status = "pass", "FULL"
        elif linalg_ops > 0:
            raise_status_class, raise_status = "partial", "PARTIAL"
        else:
            raise_status_class, raise_status = "none", "NONE"
        if kernel in ATEN_C_UNSAFE_MATCHES:
            status_class, status = "none", "UNSAFE"
        elif launches and residual == 0 and loops == 0:
            status_class, status = "pass", "FULL"
        elif launches:
            status_class, status = "partial", "PARTIAL"
        else:
            status_class, status = "none", "NONE"
        assessment = ATEN_C_MATCH_ASSESSMENT.get(kernel, "")
        perf = performance.get(kernel, {})
        execution = html.escape(perf.get("executable_status", "—"))
        correctness = html.escape(perf.get("correctness", "—"))
        problem = html.escape(perf.get("problem", "—"))
        raised_us = html.escape(perf.get("raised_us", "—"))
        resident_us = html.escape(perf.get("resident_cuda_us", "—"))
        ratio = perf.get("raised_over_resident", "—")
        ratio = html.escape(f"{ratio}×" if ratio not in ("", "—") else "—")
        baseline = html.escape(perf.get("baseline", "—"))
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
        implementation_form = html.escape(
            audit.get("implementation_form", "—").replace("_", " ")
        )
        audit_finding = html.escape(
            audit.get("compiler_gap", "—").replace("_", " ")
        )
        audit_reason = html.escape(audit.get("rationale", ""))
        audit_scope = html.escape(
            audit.get("current_match_scope", "—").replace("_", " ")
        )
        execution_class = "pass" if execution == "EXECUTED" else "none"
        correctness_class = "pass" if correctness == "PASS" else "none"
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{upstream}</td><td>{extracted_c}</td>"
            f"<td>{linalg_ops}</td><td>{loops}</td>"
            f'<td class="{raise_status_class}">{raise_status}</td>'
            f"<td>{launches}</td>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{symbol_html}</td>"
            f"<td>{audit_scope}</td><td>{candidate}</td>"
            f"<td>{implementation_form}</td>"
            f"<td><b>{audit_finding}</b><br>{audit_reason}</td>"
            f'<td class="{execution_class}">{execution}</td>'
            f'<td class="{correctness_class}">{correctness}</td>'
            f"<td><code>{problem}</code></td><td>{raised_us}</td>"
            f"<td>{resident_us}</td><td>{ratio}</td><td>{baseline}</td>"
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
    complete_matches = sum(
        row.get("current_match_scope") == "COMPLETE_REWRITE_CANDIDATE"
        for row in cuda_audit.values()
    )
    partial_matches = sum(
        row.get("current_match_scope") == "PARTIAL_STAGE_ONLY"
        for row in cuda_audit.values()
    )
    sort_links = (
        f'<b>Sort:</b> <a href="{_aten_page_filename("alphabetical", 1)}">'
        f'{"<b>alphabetical</b>" if sort_by == "alphabetical" else "alphabetical"}</a> &middot; '
        f'<a href="{_aten_page_filename("correctness", 1)}">'
        f'{"<b>correctness</b>" if sort_by == "correctness" else "correctness"}</a>'
    )
    page_links = " &middot; ".join(
        (
            f'<b>{number}</b>' if number == page else
            f'<a href="{_aten_page_filename(sort_by, number)}">{number}</a>'
        )
        for number in range(1, page_count + 1)
    )
    controls = (
        '<div class="intro" style="padding-top:10px;padding-bottom:10px">'
        f'{sort_links}<span style="margin-left:24px"><b>Page:</b> '
        f'{page_links}</span><span style="margin-left:24px">Showing '
        f'{(page - 1) * ATEN_PAGE_SIZE + 1}–'
        f'{(page - 1) * ATEN_PAGE_SIZE + len(kernels)} of '
        f'{len(ATEN_C_ORDER)}</span></div>'
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
        f'checking finds only {complete_matches} complete rewrite candidates and '
        f'{partial_matches} partial stage matches. '
        'Raising FULL/PARTIAL/NONE means Linalg with no residual loops, Linalg '
        'with residual loops, or no raised Linalg, respectively. Match '
        'FULL/PARTIAL/NONE describes semantic library-matcher coverage. Copy matching '
        'rejects known transpose and gather false-positives; the remaining '
        'pixel-shuffle view candidate is explicitly marked unsafe until its '
        'submap layout is proven by the runtime ABI. '
        'The newly available cuTensorNet tensor-product definition produced '
        'no additional ATen match: none of these kernels has its rank-6 '
        'separable 3D tensor-product signature. Existing cuBLAS, cuDNN, and '
        'custom CUDA definitions remain the correct matches. These are '
        'standalone C extractions of ATen mathematics, not the unmodified '
        'PyTorch C++ translation units (whose direct 224-file sweep produced '
        '0 Linalg operations). Large-problem silicon results use a Jetson '
        'Orin in MAXN mode. Raised time is the current host-pointer ABI; the '
        'resident baseline keeps operands on the GPU and times only the '
        'cuBLAS/cuDNN or fused CUDA operation. Both columns are warm medians '
        'of process runs 2–4 and are shown only after correctness passes.'
        ' <a href="performance.html"><b>Why are some kernels slow?</b></a> '
        'groups the measured gaps by cause and starts with a GEMV deep dive.'
        '</div>'
        + controls
        +
        '<table><thead><tr><th>kernel</th><th>original ATen CPU implementation</th>'
        '<th>standalone C form</th><th>Linalg ops</th>'
        '<th>residual loops</th><th>raising status</th>'
        '<th>launches</th><th>match status</th>'
        '<th>matched implementation</th><th>current match scope</th>'
        '<th>NVIDIA library candidate</th><th>implementation form</th>'
        '<th>audit finding</th><th>execution</th><th>correctness</th>'
        '<th>large problem</th><th>raised warm (µs)</th>'
        '<th>resident CUDA (µs)</th><th>raised / resident</th>'
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
        source = MFEM_C_ROOT / row["source"]
        raise_row = raise_rows.get((ident, variant), {})
        match_row = match_rows.get(ident, {}) if variant == "normalized" else {}
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
        ]
        for anchor, title, path in stage_paths:
            if not path.exists():
                continue
            highlighted, css = syntax_highlight(path.read_text())
            blocks.append(
                f'<h2 id="{anchor}">{title}</h2>'
                f'<div class="container">{highlighted}</div>'
            )

        launches = int(match_row.get("kernel_launches", "0") or 0)
        symbols = [
            value for value in match_row.get("launch_symbols", "").split(",")
            if value
        ]
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
            f'<b>{launches}</b> executable library launch(es)'
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
            "matched_symbols": symbols,
            "silicon": silicon_row,
        })
    return stats


def build_mfem_application_pages() -> list[dict]:
    """Render MFEM example hot-operator ports and measured status."""
    stats = []
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
            f'{html.escape(row["library_launches"])} structural launch(es). '
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
        stats.append({**row, "page_filename": page_filename})
    return stats


def build_mfem_application_extraction_pages() -> list[dict]:
    """Render raised hot paths extracted from larger MFEM applications."""
    stats = []
    summary = _read_csv(MFEM_APPLICATION_EXTRACTION_RESULTS_DIR / "summary.csv")
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
        matched_symbols = []
        if matched.exists():
            matched_symbols = sorted(set(re.findall(
                r"kernel\.launch\s+@([A-Za-z0-9_.$-]+)",
                matched.read_text(),
            )))
        matched_implementations = ", ".join(
            f"<code>@{html.escape(symbol)}</code>"
            for symbol in matched_symbols
        ) or "—"
        comparison = comparison_rows.get(function)
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
                    f'{row.get("launches", "0")} raised candidate launches; '
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
            f'&nbsp;·&nbsp; <b>{html.escape(row["launches"])}</b> candidate launch(es)<br>'
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
            "launches_int": int(row.get("launches", "0") or 0),
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
            f'<td>{row["matches_int"]}</td><td>{row["launches_int"]}</td>'
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
        'size and is intentionally not timed. Raised values are medians of warm '
        'process runs 2–4. <b>EXACT_OPERATOR</b> is a directly paired native MFEM '
        'CUDA operator. <b>COMPONENT_SUM</b> sums separately measured resident MFEM '
        'CUDA PA kernels and is a conservative component baseline, not a fused '
        'whole-application timing. PARTIAL_COMPONENT_SUM omits the ex9 PCG algebra. '
        'UNAVAILABLE means this MFEM revision exposes no equivalent native CUDA '
        'microbenchmark path.'
        '</div>'
        '<table><thead><tr><th>extracted entry</th><th>extracted C</th>'
        '<th>application</th>'
        '<th>coverage</th><th>upstream MFEM call site</th><th>Linalg ops</th>'
        '<th>residual loops</th><th>matches</th><th>candidate launches</th>'
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
            f'<td>{html.escape(stats["library_launches"])}</td>'
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
        '<th>structural launches</th><th>library-backed status</th>'
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
            raised_runtime = _fmt_seconds(
                float(silicon["raised_runtime_us"]) / 1.0e6
            )
            native_runtime = _fmt_seconds(
                float(silicon["mfem_native_runtime_us"]) / 1.0e6
            )
            ratio = float(silicon["raised_over_native"])
            ratio_cell = f'MFEM <b>{ratio:.1f}&times;</b> faster'
            before_cache = silicon.get("raised_runtime_us_before_plan_cache", "")
            cache_speedup = silicon.get("plan_cache_speedup", "")
            if before_cache and cache_speedup:
                optimization_cell = (
                    f'<b>{float(cache_speedup):.2f}&times;</b> faster<br>'
                    f'<small>before: {_fmt_seconds(float(before_cache) / 1.0e6)}</small>'
                )
            else:
                optimization_cell = "—"
        else:
            correctness = raised_runtime = native_runtime = ratio_cell = "—"
            optimization_cell = "—"
            correctness_class = ""
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{extracted_c}</td><td>{upstream}</td>"
            f"<td>{html.escape(stats['family'])}</td>"
            f"<td>{html.escape(stats['dimension'])}D</td>"
            f"<td>{html.escape(variant)}</td>"
            f"<td>{stats['linalg_ops']}</td>"
            f"<td>{stats['residual_loops']}</td>"
            f"<td>{launches}</td>"
            f'<td class="{status_class}">{status}</td>'
            f"<td>{symbols}</td>"
            f'<td class="{correctness_class}">{correctness}</td>'
            f"<td>{raised_runtime}</td><td>{optimization_cell}</td>"
            f"<td>{native_runtime}</td>"
            f"<td>{ratio_cell}</td></tr>"
        )

    originals = [row for row in mfem_stats if row["variant"] == "original"]
    normalized = [row for row in mfem_stats if row["variant"] == "normalized"]
    total_linalg = sum(row["linalg_ops"] for row in mfem_stats)
    fully_raised = sum(row["fully_raised"] for row in mfem_stats)
    matched_kernels = sum(row["launches"] > 0 for row in normalized)
    total_matches = sum(row["launches"] for row in normalized)
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
        f'The matcher emitted {total_matches} ABI-lowerable library launches '
        f'across {matched_kernels}/{len(normalized)} normalized kernels. '
        'Each row links to the '
        'stored frontend, raised, debufferized, and matcher-rewritten IR plus '
        'a Compiler Explorer deep link.'
        '<br><b>Silicon comparison:</b> all 18 matcher-covered normalized '
        'kernels (ten PA operators plus eight DFEM interpolation/integration '
        'maps) use f64, NE=1024, D1D=4, Q1D=5, and 20 warm iterations on '
        'Jetson Orin sm_87 in MAXN mode with CUDA 12.6. The raised value is '
        'the median of '
        'process runs 2–4 so the first cold CUDA process is excluded. '
        '“Raised” is the current cached host-pointer ABI (including '
        'host-mapping, correctness snapshots, and synchronization overhead); '
        'prepared cuTensorNet plans, workspaces, and scratch are reused. '
        '“MFEM CUDA” is a synchronized '
        'resident-device native MFEM launch. This intentionally records the '
        'performance gap in the current end-to-end lowering and is not a '
        'kernel-only claim.'
        '</div>'
        '<table><thead><tr><th>kernel</th><th>extracted C</th>'
        '<th>upstream MFEM source</th><th>family</th><th>dim</th>'
        '<th>variant</th><th>Linalg ops</th><th>residual loops</th>'
        '<th>executable launches</th><th>status</th>'
        '<th>matched implementation</th>'
        '<th>silicon correctness</th><th>raised current ABI</th>'
        '<th>plan-cache improvement</th>'
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
        if l > 0 and r == 0 and f == 0:
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
        first_note    = _with_rowspan(note_cell)
        first_block   = _with_rowspan(block_cell)

        rows.append(
            f'<tr>{first_kernel}{first_match}{first_note}{first_block}'
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
        '<th>fixture</th>'
        '<th>math compared</th>'
        '<th>ggml CUDA</th>'
        '<th>raised pipeline</th>'
        '<th>correctness</th>'
        '<th>notes</th>'
        '</tr></thead><tbody>'
        '<tr>'
        '<td><b>extended_forward, 7B-size one layer</b></td>'
        '<td>one token at pos=1024: MODEL_DIM=4096, FFN_DIM=11008, '
        'VOCAB=32000, SEQ_LEN=2048, HEADS=32</td>'
        '<td>warm host median 9.638 ms</td>'
        '<td>warm host median 13.480 ms<br>warm device median 12.273 ms<br>'
        'cold first iter host 447.317 ms</td>'
        '<td>first 4 logits match exactly to printed precision; checksum '
        'diff is about 0.002 over 32000 logits</td>'
        '<td>Same one-layer f32 fixture and dimensions, not the full 32-layer '
        'Llama 2 model and not a quantized GGUF path.</td>'
        '</tr>'
        '<tr>'
        '<td><b>extended_forward, toy one layer</b></td>'
        '<td>one token at pos=16: MODEL_DIM=64, FFN_DIM=128, VOCAB=256, '
        'SEQ_LEN=32, HEADS=4</td>'
        '<td>warm host median 0.098 ms<br>cold one-iter 72.725 ms</td>'
        '<td>warm host median 0.719 ms<br>warm device median 0.447 ms<br>'
        'cold first iter host 269.634 ms</td>'
        '<td>ggml CUDA vs native C max diff 8.46e-06</td>'
        '<td>Kept for fast IR/debug iteration; the 7B-size row is the '
        'headline size comparison.</td>'
        '</tr>'
        '</tbody></table>'
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
    common_legend = (
        '  Click a kernel name to open its static raised / debuferized / '
        '  matcher-rewritten IR snapshot. Each snapshot has an '
        '  <em>open in Compiler Explorer</em> link containing the full C and '
        '  MLIR source; keeping those large URLs off this suite page makes '
        '  the tracker load quickly.'
        '  The <em>residual for-loops</em> column counts imperative-loop ops '
        '  (<code>affine.for</code>, <code>scf.for</code>, '
        '  <code>scf.while</code>, <code>affine.parallel</code>, '
        '  <code>scf.parallel</code>) still present after raise + lower-submap '
        '  + debuferize — a measure of how much of the kernel remains '
        '  imperative rather than expressed as linalg / kernel.launch.'
        '  The <em>blocker</em> column links to the '
        '  <a href="index.html#taxonomy">algorithm taxonomy</a>: yellow tags are '
        '  fixable pipeline gaps, red tags are fundamental cross-iteration '
        '  dependencies that no transformation can remove.'
        '  The <em>parallelism</em> column classifies the kernel by its GPU '
        '  suitability: <span class="pass"><b>highly parallel</b></span> '
        '  (every iter independent), <span class="partial"><b>parallel + T '
        '  loop</b></span> (body parallel, outer time loop serial — stencils), '
        '  <span class="partial"><b>partial parallel</b></span> (mixes '
        '  reductions / serial steps), <span class="none"><b>serial</b></span> '
        '  (cross-iter dependencies, poor naive GPU fit — factorizations, '
        '  recurrences, DPs).'
        '  Runtime columns compare warmed raised-pipeline runtime timings '
        '  against handwritten PolyBenchGPU CUDA timings where available; '
        '  CPU comparison is intentionally hidden for now.'
    )

    polybench_section = _build_section(
        title="PolyBench/C 4.2.1",
        anchor="polybench",
        blurb=(
            "30 numerical kernels from the PolyBench/C 4.2.1 benchmark — "
            "dense linear algebra, stencils, and data-mining bodies. " +
            common_legend
        ),
        kernel_stats=polybench_stats,
        notes=KERNEL_NOTES,
        blockers=POLYBENCH_BLOCKERS,
        runtimes=POLYBENCHGPU_RUNTIMES,
    )
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
            '<a href="polybench.html">PolyBench</a> &middot; '
            '<a href="numerical.html">ATen</a> &middot; '
            '<a href="performance.html">Performance analysis</a> &middot; '
            '<a href="mfem.html">MFEM</a> &middot; '
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
        '.cause-tag { display:inline-block; border-radius:10px; padding:2px 7px; '
        'font-size:11px; font-weight:bold; margin-bottom:4px; } '
        '.cause-memory { background:#ffd9d9; color:#8b1a1a; } '
        '.cause-host { background:#eadcff; color:#53258a; } '
        '.cause-copy { background:#dcecff; color:#174f86; } '
        '.cause-intensity { background:#ffe8c7; color:#7a4300; } '
        '.cause-setup { background:#fff3bd; color:#705900; } '
        '.cause-bandwidth { background:#ffe0ec; color:#842347; } '
        '.cause-amortized { background:#dff5e5; color:#1a6a34; }'
    )

    landing = (
        nav()
        + '<div class="intro"><b>Raising and library-matching tracker.</b> '
          'The explorer is split into focused pages so the large Compiler '
          'Explorer deep-links are loaded only for the suite being inspected. '
          'Each kernel still has a static IR preview and a full CE link.</div>'
        + '<div class="suite-grid">'
        + card("polybench.html", "PolyBench/C", len(polybench_stats),
               "Dense linear algebra, stencils, and data-mining kernels.")
        + card("numerical.html", "ATen numerical kernels", len(aten_stats),
               "Extracted ATen C algorithms and Jetson comparisons.")
        + card("performance.html", "Why are some kernels slow?",
               sum(row.get("correctness") == "PASS"
                   for row in _read_csv(ATEN_SILICON_RESULTS)),
               "Root-cause groups, highlighted slowdown ratios, and a GEMV deep dive.")
        + card("mfem.html", "MFEM finite elements",
               len(mfem_stats) + len(mfem_application_stats)
               + len(mfem_application_extraction_stats),
               "Original/normalized FEM kernels and larger application hot paths.")
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
    polybench = nav() + polybench_section
    performance = nav() + _aten_slowness_page(aten_stats)
    numerical_pages: dict[str, str] = {}
    for sort_by in ("alphabetical", "correctness"):
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
    ai = nav() + llama_forward_section + whisper_ops_section + llmc_section
    vision = (
        nav() + stencil_conv2d_section + darknet_section
        + _extracted_darknet_section(ex_darknet_stats)
        + _fusion_opt_section(fopt_stats)
    )
    pva = nav() + _pva_section()
    pages = {
        "index.html": render_html("Polygeist IR explorer", landing, extra_css),
        "polybench.html": render_html(
            "Polygeist: PolyBench/C", polybench, extra_css
        ),
        "performance.html": render_html(
            "Polygeist: kernel slowness analysis", performance, extra_css
        ),
        "mfem.html": render_html("Polygeist: MFEM kernels", mfem, extra_css),
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
    unknown_args = [
        arg for arg in sys.argv[1:]
        if arg not in ("--mfem-only", "--aten-only", "--polybench-only")
    ]
    if unknown_args:
        raise SystemExit(f"unknown argument(s): {' '.join(unknown_args)}")
    if sum((mfem_only, aten_only, polybench_only)) > 1:
        raise SystemExit("suite-only arguments are mutually exclusive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        OUTPUT_DIR.joinpath("polybench.html").write_text(
            pages["polybench.html"]
        )
        print(f"Done. Open {OUTPUT_DIR}/polybench.html.")
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
    print(f"\nWrote {len(pages)} explorer pages.")
    print(f"Done. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
