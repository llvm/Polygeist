#!/usr/bin/env python3
"""Build large, correctness-gated ATen cuDNN pointwise-graph benchmarks.

These are host-pointer end-to-end runs.  The generic graph runtime presently
owns the H2D/D2H transfers, so no device-resident number is claimed here.
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
BUILDER = ROOT / "scripts/correctness/polygeist_build.sh"


def ptr(name: str, size: str, output: bool = False, init: str = "normal") -> tuple:
    return (name, "ptr", size, output, init)


def iptr(name: str, size: str, output: bool = False,
         init: str = "index") -> tuple:
    return (name, "iptr", size, output, init)


def bptr(name: str, size: str, output: bool = False) -> tuple:
    return (name, "bptr", size, output, "")


def scalar(name: str, value: float) -> tuple:
    return (name, "scalar", repr(value), False, "")


def iscalar(name: str, value: int) -> tuple:
    return (name, "iscalar", str(value), False, "")


def spec(dims: dict[str, int], args: list[tuple], coverage: str = "full graph",
         rtol: float = 2e-3) -> dict:
    return {"dims": dims, "args": args, "coverage": coverage, "rtol": rtol}


N = 4_194_304
CASES = {
    "aten_mul": spec(
        {"N": N}, [ptr("a", "N"), ptr("b", "N"),
                    ptr("out", "N", True)], "generic graph multiply"),
    "aten_clamp": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True),
                    scalar("lo", -0.5), scalar("hi", 0.75)],
        "comparison plus ternary-select graph"),
    "aten_erf": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True)],
        "cuDNN ERF pointwise node"),
    "aten_exp2": spec(
        {"N": N}, [ptr("x", "N", init="small"),
                    ptr("out", "N", True)],
        "exp2 expanded to multiply plus exp"),
    "aten_pow": spec(
        {"N": N}, [ptr("a", "N", init="positive"),
                    ptr("b", "N", init="positive"),
                    ptr("out", "N", True)], "cuDNN POW pointwise node"),
    "aten_round": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True)],
        "round expanded to compare/select/floor/ceil"),
    "aten_logical_and": spec(
        {"N": N}, [ptr("a", "N"), ptr("b", "N"),
                    ptr("out", "N", True)],
        "boolean comparisons and select with f32 materialization"),
    "aten_gelu_backward_cpu_exact": spec(
        {"N": N}, [ptr("grad", "N"), ptr("x", "N"),
                    ptr("out", "N", True)],
        "sixteen-node exact GELU backward graph"),
    "aten_gelu_backward_cpu_tanh": spec(
        {"N": N}, [ptr("grad", "N"), ptr("x", "N"),
                    ptr("out", "N", True)],
        "twenty-four-node-capable tanh GELU backward graph"),
    "aten_elu_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("output", "N", init="wide"),
                    scalar("alpha", 1.25), scalar("scale", 0.75),
                    ptr("out", "N", True)],
        "ordered select through cuDNN ReLU-backward mask"),
    "aten_softplus_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("self", "N", init="wide"),
                    scalar("beta", 1.1), scalar("threshold", 0.7),
                    ptr("out", "N", True)],
        "softplus derivative and threshold mask graph"),
    "aten_threshold_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("self", "N", init="wide"),
                    scalar("threshold", 0.35), ptr("out", "N", True)],
        "cuDNN ReLU-backward numeric mask"),
    "aten_hardswish_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("self", "N", init="wide"),
                    ptr("out", "N", True)],
        "nested ordered-select graph"),
    "aten_hardtanh_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("self", "N", init="wide"),
                    scalar("minval", -0.7), scalar("maxval", 0.8),
                    ptr("out", "N", True)],
        "compound predicate through two numeric masks"),
    "aten_hardshrink": spec(
        {"N": N}, [ptr("self", "N", init="wide"), scalar("lambd", 0.35),
                    ptr("out", "N", True)],
        "compound predicate through two numeric masks"),
    "aten_huber_backward": spec(
        {"N": N}, [ptr("input", "N", init="wide"),
                    ptr("target", "N", init="wide_shift"),
                    scalar("norm", 0.75), scalar("delta", 0.6),
                    ptr("out", "N", True)],
        "piecewise Huber derivative graph"),
    "aten_huber_elementwise": spec(
        {"N": N}, [ptr("a", "N", init="wide"),
                    ptr("b", "N", init="wide_shift"),
                    scalar("delta", 0.6), ptr("out", "N", True)],
        "piecewise Huber loss graph"),
    "aten_shrink_backward": spec(
        {"N": N}, [ptr("grad", "N"), ptr("self", "N", init="wide"),
                    scalar("lambd", 0.35), ptr("out", "N", True)],
        "compound predicate through two numeric masks"),
    "aten_smooth_l1_backward": spec(
        {"N": N}, [ptr("input", "N", init="wide"),
                    ptr("target", "N", init="wide_shift"),
                    scalar("norm", 0.75), scalar("beta", 0.6),
                    ptr("out", "N", True)],
        "piecewise smooth-L1 derivative graph"),
    "aten_smooth_l1_elementwise": spec(
        {"N": N}, [ptr("a", "N", init="wide"),
                    ptr("b", "N", init="wide_shift"),
                    scalar("beta", 0.6), ptr("out", "N", True)],
        "piecewise smooth-L1 loss graph"),
    "aten_softshrink": spec(
        {"N": N}, [ptr("self", "N", init="wide"), scalar("lambd", 0.35),
                    ptr("out", "N", True)],
        "nested ordered-select graph"),
    "aten_erfc": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True)],
        "erfc expanded to one minus erf"),
    "aten_hypot": spec(
        {"N": N}, [ptr("a", "N"), ptr("b", "N"),
                    ptr("out", "N", True)],
        "hypot expanded to squares add and sqrt"),
    "aten_logaddexp": spec(
        {"N": N}, [ptr("a", "N", init="small"),
                    ptr("b", "N", init="small"),
                    ptr("out", "N", True)],
        "stable max plus log1p-exp graph"),
    "aten_logaddexp2": spec(
        {"N": N}, [ptr("a", "N", init="small"),
                    ptr("b", "N", init="small"),
                    ptr("out", "N", True)],
        "stable base-two logaddexp graph"),
    "aten_leaky_relu": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True),
                    scalar("slope", 0.1)],
        "leaky ReLU rewritten through min-max arithmetic"),
    "aten_elu": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True),
                    scalar("alpha", 1.25), scalar("scale", 0.75)],
        "ELU rewritten through min-max-exp arithmetic"),
    "aten_frac": spec(
        {"N": N}, [ptr("x", "N"), ptr("out", "N", True)],
        "x minus trunc x rewritten to modulo one"),
    "aten_conv1d": spec(
        {"B": 32, "IC": 64, "OC": 128, "W": 4096, "K": 3},
        [ptr("input", "B*IC*W"), ptr("weight", "OC*IC*K"),
         ptr("bias", "OC"), ptr("output", "B*OC*(W-K+1)", True)],
        "full bias plus valid 1d convolution through cuDNN"),
    "aten_dilated_convolution_cpu": spec(
        {"C": 16, "O": 32, "H": 128, "W": 128, "K": 3, "D": 2},
        [ptr("x", "C*H*W"), ptr("w", "O*C*K*K"),
         ptr("out", "O*(H-2*D)*(W-2*D)", True)],
        "full constant-dilation 2d convolution through cuDNN"),
    "aten_batch_norm_transform_cpu": spec(
        {"N": 32, "C": 64, "H": 64, "W": 64},
        [ptr("x", "N*C*H*W"), ptr("mean", "C"),
         ptr("invstd", "C", init="positive"),
         ptr("weight", "C"), ptr("bias", "C"),
         ptr("out", "N*C*H*W", True)],
        "full inference batch normalization through cuDNN"),
    "aten_int_mm_out_cpu": spec(
        {"M": 512, "N": 512, "K": 1024},
        [bptr("a", "M*K"), bptr("b", "K*N"),
         iptr("out", "M*N", True)],
        "full i8 by i8 to i32 matrix multiplication through cuBLAS GemmEx"),
    "aten_sparse_norm_cpu": spec(
        {"N": 16_777_216},
        [ptr("value", "N"), ptr("out", "1", True)],
        "full Euclidean norm through cuBLAS Snrm2"),
    "aten_joint_scaling_cpu": spec(
        {"N": 16_777_216},
        [ptr("a", "N", init="wide"), ptr("b", "N", init="wide_shift"),
         ptr("out", "1", True)],
        "two max-absolute reductions through cuBLAS Isamax"),
    "aten_dropout_feature_noise_cpu": spec(
        {"B": 32, "C": 64, "H": 64, "W": 64},
        [ptr("x", "B*C*H*W"), ptr("mask", "B*C", init="unit"),
         scalar("scale", 1.25), ptr("out", "B*C*H*W", True)],
        "feature-wise broadcast multiply through cuDNN OpTensor"),
    "aten_conv_transpose2d": spec(
        {"B": 2, "IC": 16, "OC": 32, "H": 128, "W": 128, "K": 3},
        [ptr("input", "B*IC*H*W"), ptr("weight", "IC*OC*K*K"),
         ptr("output", "B*OC*(H+K-1)*(W+K-1)", True)],
        "full overlap-add transposed convolution through cuDNN backward-data"),
    "aten_depthwise_conv3x3_cpu": spec(
        {"B": 2, "C": 64, "H": 256, "W": 256},
        [ptr("x", "B*C*H*W"), ptr("weight", "C*3*3"),
         ptr("bias", "C"), ptr("out", "B*C*H*W", True)],
        "bias plus same-padding depthwise convolution through grouped cuDNN"),
    "aten_kron_impl_cpu": spec(
        {"A": 256, "B": 128, "C": 32, "D": 32},
        [ptr("x", "A*B"), ptr("y", "C*D"),
         ptr("out", "A*C*B*D", True)],
        "full Kronecker product through mode-based cuTENSOR multiply"),
    "aten_kron_out_cpu": spec(
        {"A": 256, "B": 128, "C": 32, "D": 32},
        [ptr("x", "A*B"), ptr("y", "C*D"),
         ptr("out", "A*C*B*D", True)],
        "full Kronecker product through mode-based cuTENSOR multiply"),
    "aten_binary_cross_entropy": spec(
        {"N": 8_388_608},
        [ptr("input", "N", init="unit"), ptr("target", "N", init="unit"),
         ptr("out", "1", True)],
        "cuDNN pointwise loss graph followed by cuDNN mean reduction"),
    "aten_conv_tbc_cpu": spec(
        {"T": 4096, "B": 16, "I": 32, "O": 64, "K": 3},
        [ptr("x", "T*B*I"), ptr("w", "K*I*O"),
         ptr("out", "(T-K+1)*B*O", True)],
        "full TBC convolution through cuDNN transform plus convolution"),
    "aten_transform_bias_rescale_qkv_cpu": spec(
        {"B": 8, "S": 512, "H": 16, "D": 64},
        [ptr("qkv", "B*S*3*H*D"), ptr("bias", "3*H*D"),
         scalar("scale", 0.125), ptr("q", "B*H*S*D", True),
         ptr("k", "B*H*S*D", True), ptr("v", "B*H*S*D", True)],
        "three full QKV slice-bias-permute stages through cuDNN OpTensor"),
    "aten_addr_elementwise": spec(
        {"N": 8_388_608},
        [ptr("self", "N"), ptr("x", "N"), ptr("y", "N"),
         scalar("beta", 0.0), scalar("alpha", 0.75),
         ptr("out", "N", True)],
        "full beta-zero addr graph through cuDNN pointwise operations"),
    "aten_log_sigmoid_cpu": spec(
        {"N": 8_388_608},
        [ptr("x", "N"), ptr("out", "N", True),
         ptr("buffer", "N", True)],
        "full stable log-sigmoid and saved buffer through two cuDNN graphs"),
    "aten_softplus": spec(
        {"N": 8_388_608},
        [ptr("x", "N"), ptr("out", "N", True),
         scalar("beta", 1.25), scalar("threshold", 0.5)],
        "full thresholded softplus through a cached cuDNN pointwise graph"),
    "aten_count_nonzero_cpu": spec(
        {"N": 8_388_608},
        [ptr("x", "N"), iptr("out", "1", True)],
        "full CUB transformed count-nonzero reduction"),
    "aten_count_nonzero_impl_cpu": spec(
        {"R": 131_072, "C": 64},
        [ptr("x", "R*C"), iptr("out", "R", True)],
        "full segmented CUB transformed count-nonzero reduction"),
    "aten_equal_cpu": spec(
        {"N": 8_388_608},
        [ptr("a", "N"), ptr("b", "N", init="mismatch"),
         iptr("out", "1", True)],
        "full CUB transformed equality-and reduction"),
    "aten_allany_dims_cpu": spec(
        {"R": 131_072, "C": 64},
        [iptr("x", "R*C", init="bool"), iscalar("all", 1),
         iptr("out", "R", True)],
        "full dynamic CUB segmented all-or-any reduction"),
    "aten_and_reduce_cpu": spec(
        {"R": 131_072, "K": 64},
        [iptr("x", "R*K", init="bool"), iptr("out", "R", True)],
        "full CUB segmented logical-and reduction"),
    "aten_bf16_dot_cpu": spec(
        {"K": 4_194_304},
        [ptr("a", "K"), ptr("b", "K"), ptr("out", "1", True)],
        "full scalarized-f32 dot product through the bufferized cuBLAS Sdot route",
        rtol=1e-2),
    "aten_argmax_cpu": spec(
        {"R": 131_072, "K": 64},
        [ptr("x", "R*K"), iptr("out", "R", True)],
        "full row-wise first-index argmax through CUB segmented reduction"),
    "aten_argmin_cpu": spec(
        {"R": 131_072, "K": 64},
        [ptr("x", "R*K"), iptr("out", "R", True)],
        "full row-wise first-index argmin through CUB segmented reduction"),
    "aten_bf16_gemv_trans_cpu": spec(
        {"M": 4096, "K": 8192},
        [ptr("matrix", "M*K"), ptr("vector", "M"),
         ptr("out", "K", True)],
        "full scalarized-f32 transposed GEMV through bufferized cuBLAS Sgemv"),
    "aten_sinc": spec(
        {"N": 8_388_608},
        [ptr("x", "N"), ptr("out", "N", True)],
        "full normalized sinc through a cached cuDNN pointwise graph"),
    "aten_avg_pool2d": spec(
        {"B": 2, "C": 4, "H": 16, "W": 16},
        [ptr("input", "B*C*H*W"), ptr("output", "B*C*(H/2)*(W/2)", True)],
        "full fixed average pool 2d forward"),
    "aten_avg_pool2d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "I1": 7},
        [ptr("input", "B*C*I0*I1"), ptr("output", "B*C*(I0/2)*(I1/2)", True)],
        "full fixed average pool 2d forward"),
    "aten_avg_pool2d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "I1": 7},
        [ptr("grad_output", "B*C*(I0/2)*(I1/2)"),
         ptr("grad_input", "B*C*I0*I1", True)],
        "full fixed average pool 2d backward"),
    "aten_avg_pool3d": spec(
        {"B": 2, "C": 3, "D": 8, "H": 8, "W": 8},
        [ptr("input", "B*C*D*H*W"),
         ptr("output", "B*C*(D/2)*(H/2)*(W/2)", True)],
        "full fixed average pool 3d forward"),
    "aten_avg_pool3d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "I1": 7, "I2": 8},
        [ptr("input", "B*C*I0*I1*I2"),
         ptr("output", "B*C*(I0/2)*(I1/2)*(I2/2)", True)],
        "full fixed average pool 3d forward"),
    "aten_avg_pool3d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "I1": 7, "I2": 8},
        [ptr("grad_output", "B*C*(I0/2)*(I1/2)*(I2/2)"),
         ptr("grad_input", "B*C*I0*I1*I2", True)],
        "full fixed average pool 3d backward"),
    "aten_batch_norm_backward_cpu": spec(
        {"B": 4, "C": 8, "S": 32},
        [ptr("grad", "B*C*S"), ptr("x", "B*C*S"),
         ptr("mean", "C"), ptr("invstd", "C", init="positive"),
         ptr("weight", "C"), ptr("dx", "B*C*S", True),
         ptr("dweight", "C", True), ptr("dbias", "C", True)],
        "full cuDNN batch normalization backward"),
    "aten_batch_norm_backward_template_cpu": spec(
        {"N": 8, "C": 16, "H": 16, "W": 16},
        [ptr("grad", "N*C*H*W"), ptr("x", "N*C*H*W"),
         ptr("mean", "C"), ptr("invstd", "C", init="positive"),
         ptr("out", "N*C*H*W", True)],
        "full cuDNN batch normalization input gradient"),
    "aten_adaptive_avg_pool2d": spec(
        {"B": 4, "C": 32, "H": 256, "W": 256, "OH": 128, "OW": 128},
        [ptr("input", "B*C*H*W"), ptr("output", "B*C*OH*OW", True)],
        "full regular 2x2 uniform-window convolution"),
    "aten_adaptive_avg_pool2d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3},
        [ptr("input", "B*C*I0*I1"), ptr("output", "B*C*O0*O1", True)],
        "full fractional adaptive average forward"),
    "aten_adaptive_avg_pool2d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3},
        [ptr("grad_output", "B*C*O0*O1"),
         ptr("grad_input", "B*C*I0*I1", True)],
        "full fractional adaptive average backward"),
    "aten_adaptive_avg_pool3d": spec(
        {"B": 2, "C": 3, "D": 8, "H": 8, "W": 8},
        [ptr("input", "B*C*D*H*W"), ptr("output", "B*C*4*4*4", True)],
        "full regular adaptive average 3d forward"),
    "aten_adaptive_avg_pool3d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3,
         "I2": 8, "O2": 3},
        [ptr("input", "B*C*I0*I1*I2"),
         ptr("output", "B*C*O0*O1*O2", True)],
        "full fractional adaptive average 3d forward"),
    "aten_adaptive_avg_pool3d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3,
         "I2": 8, "O2": 3},
        [ptr("grad_output", "B*C*O0*O1*O2"),
         ptr("grad_input", "B*C*I0*I1*I2", True)],
        "full fractional adaptive average 3d backward"),
    "aten_adaptive_max_pool1d_cpu": spec(
        {"C": 4, "I": 32, "O": 7},
        [ptr("x", "C*I"), ptr("out", "C*O", True),
         iptr("index", "C*O", True)],
        "full fractional adaptive max 1d forward"),
    "aten_adaptive_max_pool2d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3},
        [ptr("input", "B*C*I0*I1"), ptr("output", "B*C*O0*O1", True),
         iptr("indices", "B*C*O0*O1", True)],
        "full fractional adaptive max 2d forward"),
    "aten_adaptive_max_pool2d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3},
        [ptr("grad_output", "B*C*O0*O1"),
         iptr("indices", "B*C*O0*O1", init="index42"),
         ptr("grad_input", "B*C*I0*I1", True)],
        "full saved-index adaptive max 2d backward"),
    "aten_adaptive_max_pool3d_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3,
         "I2": 8, "O2": 3},
        [ptr("input", "B*C*I0*I1*I2"),
         ptr("output", "B*C*O0*O1*O2", True),
         iptr("indices", "B*C*O0*O1*O2", True)],
        "full fractional adaptive max 3d forward"),
    "aten_adaptive_max_pool3d_backward_cpu": spec(
        {"B": 1, "C": 2, "I0": 6, "O0": 3, "I1": 7, "O1": 3,
         "I2": 8, "O2": 3},
        [ptr("grad_output", "B*C*O0*O1*O2"),
         iptr("indices", "B*C*O0*O1*O2", init="index336"),
         ptr("grad_input", "B*C*I0*I1*I2", True)],
        "full saved-index adaptive max 3d backward"),
    "aten_adaptive_max_pool3d_legacy_cpu": spec(
        {"C": 2, "ID": 8, "IH": 9, "IW": 10, "OD": 3, "OH": 4, "OW": 5},
        [ptr("x", "C*ID*IH*IW"), ptr("out", "C*OD*OH*OW", True),
         iptr("idx", "C*OD*OH*OW", True)],
        "full fractional adaptive max 3d legacy forward"),
    "aten_adaptive_max_pool3d_legacy_backward_cpu": spec(
        {"C": 2, "ID": 8, "IH": 9, "IW": 10, "OD": 3, "OH": 4, "OW": 5},
        [ptr("g", "C*OD*OH*OW"),
         iptr("idx", "C*OD*OH*OW", init="index720"),
         ptr("out", "C*ID*IH*IW", True)],
        "full saved-index adaptive max 3d legacy backward"),
    "aten_addcdiv": spec({"N": N}, [ptr("self", "N"), ptr("x", "N"), ptr("y", "N", init="positive"), scalar("value", .75), ptr("out", "N", True)]),
    "aten_addcmul": spec({"N": N}, [ptr("self", "N"), ptr("x", "N"), ptr("y", "N"), scalar("value", .75), ptr("out", "N", True)]),
    "aten_batch_norm_cpu_entry": spec({"N": N}, [ptr("x", "N"), scalar("scale", 1.25), scalar("bias", -.2), ptr("out", "N", True)]),
    "aten_cross": spec({"N": N // 3}, [ptr("a", "N*3"), ptr("b", "N*3"), ptr("out", "N*3", True)], "three graph stages"),
    "aten_cross_cpu_backend": spec({"V": N // 3}, [ptr("a", "V*3"), ptr("b", "V*3"), ptr("out", "V*3", True)], "three graph stages"),
    "aten_dirichlet_transform_cpu": spec({"R": 65_536, "C": 64}, [ptr("gamma", "R*C", init="positive"), ptr("out", "R*C", True)], "partial graph epilogue"),
    "aten_div": spec({"N": N}, [ptr("a", "N"), ptr("b", "N", init="positive"), ptr("out", "N", True)]),
    "aten_glu": spec({"N": N}, [ptr("a", "N"), ptr("b", "N"), ptr("out", "N", True)]),
    "aten_glu_backward": spec({"N": N}, [ptr("sigmoid_b", "N", init="unit"), ptr("grad", "N"), ptr("a", "N"), ptr("out", "N", True)]),
    "aten_gradient_cpu": spec({"N": N}, [ptr("x", "N"), scalar("h", .125), ptr("out", "N", True)], "partial graph interior"),
    "aten_gradient_float_cpu": spec({"N": N}, [ptr("x", "N"), ptr("coord", "N", init="coord"), ptr("out", "N", True)], "partial graph interior"),
    "aten_grid_sampler_2d_backward_cpu": spec({"B": 2, "C": 16, "IH": 128, "IW": 128, "OH": 96, "OW": 96}, [ptr("x", "B*C*IH*IW"), ptr("grid", "B*OH*OW*2", init="grid"), ptr("grad", "B*C*OH*OW"), ptr("dx", "B*C*IH*IW", True), ptr("dgrid", "B*OH*OW*2", True)], "partial graph stages"),
    "aten_host_softmax_backward_cpu": spec({"R": 65_536, "K": 64}, [ptr("grad", "R*K"), ptr("output", "R*K", init="unit"), ptr("out", "R*K", True)], "partial graph epilogue"),
    "aten_layer_norm": spec({"N": N}, [ptr("x", "N"), ptr("weight", "N"), ptr("bias", "N"), ptr("out", "N", True), scalar("eps", 1e-5)], "partial graph epilogue"),
    "aten_lerp": spec({"N": N}, [ptr("a", "N"), ptr("b", "N"), ptr("weight", "N", init="unit"), ptr("out", "N", True)]),
    "aten_lerp_scalar": spec({"N": N}, [ptr("self", "N"), ptr("end", "N"), scalar("weight", .3), ptr("out", "N", True)]),
    "aten_lerp_scalar_cpu": spec({"N": N}, [ptr("self", "N"), ptr("end", "N"), scalar("weight", .3), ptr("out", "N", True)]),
    "aten_lerp_tensor_cpu": spec({"N": N}, [ptr("self", "N"), ptr("end", "N"), ptr("weight", "N", init="unit"), ptr("out", "N", True)]),
    "aten_log_normal_cpu": spec({"N": N}, [ptr("standard_normal", "N", init="small"), scalar("mean", .1), scalar("std", .25), ptr("out", "N", True)]),
    "aten_mse_backward": spec({"N": N}, [ptr("input", "N"), ptr("target", "N"), scalar("value", .5), ptr("out", "N", True)]),
    "aten_mse_elementwise": spec({"N": N}, [ptr("a", "N"), ptr("b", "N"), ptr("out", "N", True)]),
    "aten_mse_loss": spec({"N": N}, [ptr("input", "N"), ptr("target", "N"), ptr("scratch", "N", True), ptr("out", "1", True)], "partial graph plus reduction"),
    "aten_nested_softmax_backward_cpu": spec({"B": 65_536, "N": 64}, [ptr("grad", "B*N"), ptr("y", "B*N", init="unit"), ptr("out", "B*N", True)], "partial graph epilogue"),
    "aten_normal_cpu": spec({"N": N}, [ptr("standard_normal", "N"), scalar("mean", .1), scalar("std", .75), ptr("out", "N", True)]),
    "aten_rsqrt": spec({"N": N}, [ptr("x", "N", init="positive"), ptr("out", "N", True)]),
    "aten_sigmoid_backward": spec({"N": N}, [ptr("grad", "N"), ptr("output", "N", init="unit"), ptr("out", "N", True)]),
    "aten_sparse_coo_softmax_backward_cpu": spec({"R": 524_288, "K": 8}, [ptr("grad", "R*K"), ptr("y", "R*K", init="unit"), ptr("out", "R*K", True)], "partial graph epilogue"),
    "aten_square": spec({"N": N}, [ptr("x", "N"), ptr("out", "N", True)]),
    "aten_tanh_backward": spec({"N": N}, [ptr("grad", "N"), ptr("output", "N", init="unit"), ptr("out", "N", True)]),
    "aten_uniform_cpu": spec({"N": N}, [ptr("uniform01", "N", init="unit"), scalar("from", -2.), scalar("to", 3.), ptr("out", "N", True)]),
}


_POSITIVE = ("log", "sqrt", "rsqrt", "acosh", "reciprocal", "digamma", "lgamma",
             # domain-sensitive: non-zero divisor for div/mod, positive base
             # for pow(., frac) — otherwise inf/nan breaks the correctness check
             "pow", "fmod", "remainder", "div_floor", "div_trunc",
             "floor_divide")
_UNIT = ("acos", "asin", "atanh")


def _auto_init(kernel: str, name: str) -> str:
    base = kernel[5:] if kernel.startswith("aten_") else kernel
    if any(t in base for t in _POSITIVE):
        return "positive"
    if any(t in base for t in _UNIT):
        return "unit"
    return "normal"


def auto_spec(kernel: str) -> dict | None:
    """Synthesize a harness spec from a kernel's extracted C signature.
    Handles float/int/signed-char array params + scalar params; skips doubles
    (harness is f32) and anything it can't parse cleanly. Output params are the
    ones written in the body; dims are scaled to ~4M total elements."""
    f = ATEN / f"{kernel}.c"
    if not f.exists():
        return None
    txt = f.read_text()
    sig = re.search(rf"void\s+{re.escape(kernel)}\s*\(([^)]*)\)", txt)
    if not sig:
        return None
    body = txt[sig.end():]
    defined = {k: int(v) for k, v in
               re.findall(r"#\s*define\s+(\w+)\s+(\d+)", txt)}
    args, used_dims = [], set()
    for p in (x.strip() for x in sig.group(1).split(",") if x.strip()):
        # array dims may be arithmetic expressions (e.g. [B*N], [A*B], [R+M]).
        marr = re.fullmatch(
            r"(float|double|int|signed char)\s+(\w+)\s*((?:\[[\w*+ \-]+\])+)", p)
        msc = re.fullmatch(r"(float|double|int)\s+(\w+)", p)
        if marr:
            typ, nm, dimspec = marr.group(1), marr.group(2), marr.group(3)
            if typ == "double":
                return None  # harness is f32-only
            exprs = [e.strip() for e in re.findall(r"\[([\w*+ \-]+)\]", dimspec)]
            for ident in set(re.findall(r"[A-Za-z_]\w*", " ".join(exprs))):
                if ident not in defined:
                    return None
                used_dims.add(ident)
            size = "*".join(f"({e})" for e in exprs)
            # output = written in the body (handles multi-dim out[a][b][c] = ...)
            is_out = bool(re.search(
                rf"\b{re.escape(nm)}\s*(?:\[[^\]]*\])+\s*[-+*/]?=(?!=)", body))
            if typ == "float":
                args.append(ptr(nm, size, is_out, _auto_init(kernel, nm)))
            elif typ == "int":
                args.append(iptr(nm, size, is_out))
            else:
                args.append(bptr(nm, size, is_out))
        elif msc:
            typ, nm = msc.group(1), msc.group(2)
            if typ == "double":
                return None
            args.append(scalar(nm, 0.5) if typ == "float" else iscalar(nm, 2))
        else:
            return None
    if not used_dims or not any(a[3] for a in args if a[1] in
                                ("ptr", "iptr", "bptr")):
        return None  # need at least one dim and one output
    base = {d: defined[d] for d in sorted(used_dims)}  # deterministic order
    arrays = [a[2] for a in args if a[1] in ("ptr", "iptr", "bptr")]

    def footprint(dd):  # largest array's true element count at these dims
        best = 0
        for expr in arrays:
            try:
                best = max(best, eval(expr, {"__builtins__": {}}, dd))
            except Exception:
                pass
        return best

    # Numerically binary-search a uniform dim factor so the largest array is
    # ~4M elements. Robust for ANY size expression (products, sums, nesting) —
    # no fragile analytical rank (footprint is monotonic in the factor).
    dims = dict(base)
    if footprint(base) > 0:
        lo, hi = 1e-4, 1e7
        for _ in range(50):
            mid = (lo * hi) ** 0.5
            dd = {d: max(2, round(v * mid)) for d, v in base.items()}
            if footprint(dd) < 4_194_304:
                lo = mid
            else:
                hi = mid
        f = (lo * hi) ** 0.5
        dims = {d: max(2, round(v * f)) for d, v in base.items()}
    return spec(dims, args, f"auto: {kernel}")


def scaled_source(kernel: str, cfg: dict, out: Path) -> None:
    text = (ATEN / f"{kernel}.c").read_text()
    for name, value in cfg["dims"].items():
        pattern = rf"(^\s*#\s*define\s+{re.escape(name)}\s+)[^\n]+"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)
        if not count:
            text = f"#define {name} {value}\n" + text
    out.write_text(text)


def harness_text(kernel: str, cfg: dict) -> str:
    decls, call_ref, call_got, allocations, init, comparisons, frees = [], [], [], [], [], [], []
    # Device-resident path: cudaMalloc buffers, copy in/out OUTSIDE the timed
    # region so timing reflects the op on device DRAM (torch's methodology).
    call_dev, dev_alloc, dev_h2d, dev_d2h, dev_free = [], [], [], [], []
    for name, kind, value, output, init_kind in cfg["args"]:
        if kind in ("scalar", "iscalar"):
            decls.append((f"float {name} = {value}f;" if kind == "scalar"
                          else f"int {name} = {value};"))
            call_ref.append(name); call_got.append(name); call_dev.append(name)
            continue
        allocations.append(f"size_t {name}_n = (size_t)({value});")
        ctype = ("int" if kind == "iptr" else
                 "signed char" if kind == "bptr" else "float")
        allocations.append(f"{ctype} *{name}_ref = aligned_alloc(64, (({name}_n*sizeof({ctype})+63)/64)*64);")
        allocations.append(f"{ctype} *{name}_got = aligned_alloc(64, (({name}_n*sizeof({ctype})+63)/64)*64);")
        allocations.append(f"{ctype} *{name}_dev = 0;")
        dev_alloc.append(f"cudaMalloc((void**)&{name}_dev, {name}_n*sizeof({ctype}));")
        dev_h2d.append(f"cudaMemcpy({name}_dev, {name}_got, {name}_n*sizeof({ctype}), 1);")
        call_dev.append(f"{name}_dev")
        dev_free.append(f"cudaFree({name}_dev);")
        if output:
            dev_d2h.append(f"cudaMemcpy({name}_got, {name}_dev, {name}_n*sizeof({ctype}), 2);")
        if kind == "bptr":
            init.append(f"for(size_t i=0;i<{name}_n;++i) {name}_ref[i]=(signed char)((int)(i%13)-6);")
            init.append(f"memcpy({name}_got,{name}_ref,{name}_n*sizeof(signed char));")
            call_ref.append(f"{name}_ref"); call_got.append(f"{name}_got")
            if output:
                comparisons.append(f"CHECK_BARRAY({name});")
            frees.extend([f"free({name}_ref);", f"free({name}_got);"])
            continue
        if kind == "iptr":
            modulus = re.fullmatch(r"index(\d+)", init_kind)
            expr = (f"(int)(i%{modulus.group(1)})" if modulus else
                    "(int)((i%7)!=0)" if init_kind == "bool" else "0")
            init.append(f"for(size_t i=0;i<{name}_n;++i) {name}_ref[i]={expr};")
            init.append(f"memcpy({name}_got,{name}_ref,{name}_n*sizeof(int));")
            call_ref.append(f"{name}_ref"); call_got.append(f"{name}_got")
            if output:
                comparisons.append(f"CHECK_IARRAY({name});")
            frees.extend([f"free({name}_ref);", f"free({name}_got);"])
            continue
        if init_kind == "coord":
            expr = "0.01f*(float)i"
        elif init_kind == "grid":
            expr = "-0.8f + 1.6f*(float)(i%97)/96.0f"
        elif init_kind == "positive":
            expr = "0.25f + (float)(i%101)/101.0f"
        elif init_kind == "unit":
            expr = "0.05f + 0.9f*(float)(i%101)/101.0f"
        elif init_kind == "small":
            expr = "((float)(i%101)-50.0f)/100.0f"
        elif init_kind == "wide":
            expr = "(float)((int)(i%11)-5)"
        elif init_kind == "wide_shift":
            expr = "(float)((int)(i%13)-6)"
        elif init_kind == "mismatch":
            expr = "i == 12345 ? 99.0f : ((float)(i%101)-50.0f)/37.0f"
        else:
            expr = "((float)(i%101)-50.0f)/37.0f"
        init.append(f"for(size_t i=0;i<{name}_n;++i) {name}_ref[i]={expr};")
        init.append(f"memcpy({name}_got,{name}_ref,{name}_n*sizeof(float));")
        call_ref.append(f"{name}_ref"); call_got.append(f"{name}_got")
        if output:
            comparisons.append(f"CHECK_ARRAY({name});")
        frees.extend([f"free({name}_ref);", f"free({name}_got);"])
    types = [
        "float" if a[1] == "scalar" else
        "int" if a[1] == "iscalar" else
        "int *" if a[1] == "iptr" else
        "signed char *" if a[1] == "bptr" else "float *"
        for a in cfg["args"]
    ]
    signature = ", ".join(types)
    ref_args = ", ".join(call_ref); got_args = ", ".join(call_got)
    dev_args = ", ".join(call_dev)
    shape_str = "_".join(f"{k}={v}" for k, v in cfg["dims"].items())
    dimension_defines = "\n".join(f"#define {k} {v}" for k, v in cfg["dims"].items())
    return f'''#define _POSIX_C_SOURCE 200809L
{dimension_defines}
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void {kernel}({signature});
extern void {kernel}_reference({signature});
extern int cudaMalloc(void**, unsigned long);
extern int cudaMemcpy(void*, const void*, unsigned long, int);
extern int cudaFree(void*);
extern int cudaDeviceSynchronize(void);
static double now_us(void) {{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return 1e6*t.tv_sec+1e-3*t.tv_nsec; }}
#define CHECK_ARRAY(name) do {{ for(size_t i=0;i<name##_n;++i) {{ float r=name##_ref[i], g=name##_got[i]; float e=fabsf(r-g); if(!isfinite(g)||e>{cfg.get('rtol', 2e-3):.9g}f*(1.0f+fabsf(r))) {{ if(errors++<8) fprintf(stderr,"mismatch " #name "[%zu]: ref=%g got=%g err=%g\\n",i,r,g,e); }} if(e>max_error) max_error=e; }} }} while(0)
#define CHECK_IARRAY(name) do {{ for(size_t i=0;i<name##_n;++i) {{ int r=name##_ref[i], g=name##_got[i]; if(r!=g) {{ if(errors++<8) fprintf(stderr,"mismatch " #name "[%zu]: ref=%d got=%d\\n",i,r,g); }} }} }} while(0)
#define CHECK_BARRAY(name) do {{ for(size_t i=0;i<name##_n;++i) {{ int r=(int)name##_ref[i], g=(int)name##_got[i]; if(r!=g) {{ if(errors++<8) fprintf(stderr,"mismatch " #name "[%zu]: ref=%d got=%d\\n",i,r,g); }} }} }} while(0)
int main(void) {{
  {' '.join(decls)}
  {' '.join(allocations)}
  {' '.join(init)}
  {kernel}_reference({ref_args});
  /* Correctness on a SINGLE run, BEFORE the timing loops mutate the buffers.
     In-place ops (e.g. out+=src) would otherwise accumulate over ~36 calls. */
  {kernel}({got_args});
  int errors=0; float max_error=0; {' '.join(comparisons)}
  for(int i=0;i<3;++i) {kernel}({got_args});
  double total=0; for(int i=0;i<10;++i) {{ double t=now_us(); {kernel}({got_args}); total += now_us()-t; }}
  /* Device-resident timing: operands in cudaMalloc'd device DRAM, copy in/out
     ONCE outside the timed loop, so only the op is measured (matches torch). */
  double resident_us = -1.0;
#ifndef BENCH_MAPPED_ONLY
  {' '.join(dev_alloc)}
  {' '.join(dev_h2d)}
  cudaDeviceSynchronize();
  for(int i=0;i<3;++i) {kernel}({dev_args});
  cudaDeviceSynchronize();
  /* best-of-20, wall-clock + full device sync (needed: shims run async on a
     private stream). best-of matches torch's cudaEvent best-of statistic. */
  {{ double best=1e30; for(int i=0;i<20;++i) {{ double t=now_us(); {kernel}({dev_args}); cudaDeviceSynchronize(); double d=now_us()-t; if(d<best) best=d; }} resident_us = best; }}
  {' '.join(dev_d2h)}
  {' '.join(dev_free)}
#endif
  printf("RESULT kernel={kernel} warm_us=%.6f resident_us=%.6f errors=%d max_error=%g shape={shape_str} coverage={cfg['coverage'].replace(' ', '_')}\\n",total/10.0,resident_us,errors,max_error);
  {' '.join(frees)}
  return errors ? 1 : 0;
}}
'''


def run(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log}")


def build_one(kernel: str, cfg: dict, output: Path) -> dict:
    work = output / kernel; work.mkdir(parents=True, exist_ok=True)
    source = work / f"{kernel}_large.c"; scaled_source(kernel, cfg, source)
    harness = work / "harness.c"; harness.write_text(harness_text(kernel, cfg))
    reference = work / "reference.o"
    run(["aarch64-linux-gnu-gcc", "-O3", f"-D{kernel}={kernel}_reference", "-c", str(source), "-o", str(reference)], work / "reference.build.log")
    exe = work / kernel
    env = os.environ.copy()
    env.update({"PYTHON": "/usr/bin/python3", "POLYGEIST_CUSTOM_CUDA_OBJ": str(reference), "POLYGEIST_MINIMAL_CUDNN_RUNTIME": "1"})
    _ct = "/home/arjaiswal/cutensor_sbsa"
    if os.path.isdir(_ct):  # enable cutensorUnary etc. when the SDK is staged
        env["POLYGEIST_CUTENSOR_ROOT"] = _ct
    run([str(BUILDER), "--target=jetson", f"--function={kernel}", f"--harness={harness}", "-o", str(exe), str(source)], work / "raised.build.log", env)
    return {"kernel": kernel, "problem": " ".join(f"{k}={v}" for k,v in cfg["dims"].items()), "coverage": cfg["coverage"], "executable": str(exe)}


def _matched_kernels() -> list[str]:
    """Every kernel whose matched.mlir emits a library kernel.launch."""
    out = []
    for mm in sorted((ATEN / "results").glob("*/matched.mlir")):
        try:
            if "kernel.launch @" in mm.read_text():
                out.append(mm.parent.name)
        except OSError:
            pass
    return out


def _cfg_for(kernel: str) -> dict | None:
    return CASES.get(kernel) or auto_spec(kernel)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/tmp/aten_pointwise_graph_large"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--kernel", action="append", choices=sorted(CASES))
    p.add_argument("--all-matched", action="store_true",
                   help="build every matched kernel: CASES specs, else auto_spec")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.all_matched:
        selected = [k for k in _matched_kernels() if _cfg_for(k)]
        print(f"[all-matched] {len(selected)} kernels have a usable spec", flush=True)
    else:
        selected = args.kernel or sorted(CASES)
    rows=[]; failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs={pool.submit(build_one,k,_cfg_for(k),args.output):k for k in selected}
        for future in concurrent.futures.as_completed(jobs):
            k=jobs[future]
            try: rows.append(future.result()); print(f"[BUILT] {k}", flush=True)
            except Exception as exc: failures.append({"kernel":k,"error":str(exc)}); print(f"[FAIL] {k}: {exc}",file=sys.stderr,flush=True)
    manifest={"cases":sorted(rows,key=lambda x:x["kernel"]),"failures":failures}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"built={len(rows)} failed={len(failures)} output={args.output}")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
