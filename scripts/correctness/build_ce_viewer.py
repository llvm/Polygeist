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
}

WHISPER_OPS_ORDER = list(WHISPER_OPS_KERNELS.keys())

WHISPER_OPS_DISPLAY_NAMES: dict[str, str] = {
    "whisper_vec_dot":      "vector dot",
    "whisper_vec_softmax":  "vector softmax",
    "whisper_softmax_full": "full softmax",
    "whisper_rms_norm":     "RMSNorm",
    "whisper_gelu":         "GELU",
    "whisper_conv1d":       "1D convolution",
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
    "raise-crash":       ("polygeist-opt crash during raise",
                          "polygeist-opt segfaults in the raise pipeline; needs deeper investigation"),
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
    "whisper_softmax_full": ("matcher-gap", "Raises as the expected max-reduce + exp/sum + normalize sequence; today only the normalize/scal tail matches, leaving max/exp-sum residual linalg."),
    "whisper_rms_norm":     ("runtime-gap", "Matches the RMSNorm family as unweighted RMSNorm and emits the tensor launch form; runtime/library lowering for the unweighted ABI is still follow-up work."),
    "whisper_gelu":         ("matcher-gap", "Raises to one elementwise tensor linalg.generic with math.tanh; no GELU matcher/library template is wired yet."),
    "whisper_conv1d":       ("matcher-gap", "Raises and matches the per-output inner dot, but still leaves one output-position loop; full 1D conv composition/library routing is future matcher work."),
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


def ce_link(kernel: str, mlir_dir: Path = MLIR_DIR,
            kset: str = "polybench") -> str | None:
    """Construct the CE deep-link URL for a kernel; None if sources missing."""
    c_path = find_kernel_c(kernel, kset=kset)
    mlir_path = mlir_dir / f"{kernel}.mlir"
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
        capture_output=True, text=True, timeout=120,
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

    pages: dict[str, str] = {}
    css = ""
    n_for = 0
    report = [("launches", 0), ("residual_lg", 0)]

    if raised.exists():
        raised_text = raised.read_text()
        html, css = syntax_highlight(raised_text)
        pages["raised"] = html
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
        rewritten, report = run_rewriter(debuf)
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
        f'jump to: <a href="#raised">raised</a> · '
        f'<a href="#debuf">debuferized</a> · '
        f'<a href="#debuf_mr">debuf multi-root</a> · '
        f'<a href="#matched">kernel.launch output</a>'
        f'</div>'
    )
    header = (
        f'<div class="header"><h1><a href="index.html">← index</a> '
        f'&nbsp; {kernel}{open_link}</h1></div>'
        + summary
    )
    body_blocks = []
    for stage, title in [
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
        "residual": report[1][1],
        "residual_for": n_for,
        "ce_url": ce_url,
        "page_filename": f"{file_prefix}{kernel}.html",
    }


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
    "raise-crash":       "none",
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
        l = s["launches"]; r = s["residual"]; f = s["residual_for"]
        if l > 0 and r == 0 and f == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        for_cls = "none" if f > 0 else "pass"

        if s["ce_url"]:
            label = (display_names or {}).get(k, k)
            kernel_link = f'<a class="kernel" href="{s["ce_url"]}" target="_blank">{label}</a>'
        else:
            label = (display_names or {}).get(k, k)
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
                f'<a href="#taxonomy" style="color:inherit; text-decoration:none">'
                f'<b>{block_label}</b></a></td>'
                f'<td style="font-size:12px; color:#555">{block_blurb}</td>'
            )

        page_file = s.get("page_filename", f"{k}.html")
        kernel_cell = (
            f'<td>{kernel_link}'
            f'<a class="viewer" href="{page_file}" style="margin-left:12px">[IR preview]</a>'
            f'</td>'
        )
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


def build_index(polybench_stats: dict[str, dict],
                 llama_forward_stats: dict[str, dict],
                 whisper_ops_stats: dict[str, dict],
                 stencil_conv2d_stats: dict[str, dict],
                 llmc_stats: dict[str, dict],
                 darknet_stats: dict[str, dict],
                 ex_darknet_stats: dict[str, dict],
                 fopt_stats: dict[str, dict]) -> str:
    common_legend = (
        '  Click a kernel name to open the full Polygeist pipeline in '
        '  Compiler Explorer: C source on the left feeds cgeist; the affine '
        '  MLIR on the right feeds <code>polygeist-opt</code> with an '
        '  <em>Opt Pipeline</em> pane showing every internal pass. '
        '  The <code>[IR preview]</code> link opens a static snapshot of the '
        '  raised / debuferized / matcher-rewritten IR for that kernel.'
        '  The <em>residual for-loops</em> column counts imperative-loop ops '
        '  (<code>affine.for</code>, <code>scf.for</code>, '
        '  <code>scf.while</code>, <code>affine.parallel</code>, '
        '  <code>scf.parallel</code>) still present after raise + lower-submap '
        '  + debuferize — a measure of how much of the kernel remains '
        '  imperative rather than expressed as linalg / kernel.launch.'
        '  The <em>blocker</em> column links to the '
        '  <a href="#taxonomy">algorithm taxonomy</a>: yellow tags are '
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

    body = (
        '<div class="header"><h1>Polygeist IR explorer</h1>'
        '<div style="margin-top:6px; font-size:13px;">'
        '  Jump to: '
        '  <a href="#taxonomy">Algorithm taxonomy</a> &middot; '
        '  <a href="#polybench">PolyBench</a> &middot; '
        '  <a href="#llama-forward">Llama forward fixtures</a> &middot; '
        '  <a href="#whisper-ops">Whisper extracted kernels</a> &middot; '
        '  <a href="#stencil-conv2d">Stencil Conv2D</a> &middot; '
        '  <a href="#llmc">llm.c</a> &middot; '
        '  <a href="#darknet">darknet</a> &middot; '
        '  <a href="#extracted-darknet">extracted darknet</a> &middot; '
        '  <a href="#fusion-opt">Fusion optimization</a> &middot; '
        '  <a href="#pva">PVA backend</a>'
        '</div></div>'
        + _build_taxonomy_panel()
        + polybench_section
        + llama_forward_section
        + whisper_ops_section
        + stencil_conv2d_section
        + llmc_section
        + darknet_section
        + _extracted_darknet_section(ex_darknet_stats)
        + _fusion_opt_section(fopt_stats)
        + _pva_section()
    )
    # Extra CSS for section headers.
    extra_css = (
        '.section-header { background: #eaeefa; padding: 8px 20px; '
        'border-top: 2px solid #c4cce0; border-bottom: 1px solid #c4cce0; '
        'margin-top: 24px; } '
        '.section-title { margin: 0; font-size: 16px; color: #1f2d3d; }'
    )
    return render_html("Polygeist IR explorer", body, extra_css)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("llama_*.html"):
        stale.unlink()
    for stale in OUTPUT_DIR.glob("whisper_*.html"):
        stale.unlink()

    # PolyBench set.
    pb_kernels = discover_kernels(MLIR_DIR)
    print(f"Rendering {len(pb_kernels)} PolyBench kernels...", flush=True)
    pb_stats = {}
    for i, k in enumerate(pb_kernels, 1):
        print(f"  [PB {i:2d}/{len(pb_kernels)}] {k}", flush=True)
        pb_stats[k] = build_kernel_page(k, mlir_dir=MLIR_DIR,
                                         kset="polybench", file_prefix="")

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

    OUTPUT_DIR.joinpath("index.html").write_text(
        build_index(pb_stats, llama_forward_stats, whisper_ops_stats,
                    stencil_conv2d_stats,
                    llmc_stats, darknet_stats, ex_darknet_stats, fopt_stats))
    print(f"\nDone. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
