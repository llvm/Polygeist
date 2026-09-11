#!/usr/bin/env python3
"""Generate standalone C specializations of simple ATen CPU scalar kernels.

The formulas below are transcribed from the scalar lambdas in the pinned
PyTorch checkout.  This removes TensorIterator, dispatch, vectorization, and
dynamic-shape plumbing while retaining the numerical operation being tested.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "issues/aten_c_kernels"
MANIFEST = OUT / "generated_provenance.csv"

UNARY = {
    # name: (C expression, upstream token)
    "abs": ("x[i] < 0.0f ? -x[i] : x[i]", "abs_kernel"),
    "neg": ("-x[i]", "neg_kernel"),
    "reciprocal": ("1.0f / x[i]", "reciprocal_kernel"),
    "sign": ("(float)((0.0f < x[i]) - (x[i] < 0.0f))", "sign_kernel"),
    "square": ("x[i] * x[i]", "square_kernel"),
    "logical_not_f32": ("(float)(!x[i])", "logical_not_kernel"),
}

BINARY = {
    "mul": ("a[i] * b[i]", "mul_kernel"),
    "div": ("a[i] / b[i]", "div_true_kernel"),
    "maximum": ("a[i] > b[i] ? a[i] : b[i]", "maximum_kernel"),
    "minimum": ("a[i] < b[i] ? a[i] : b[i]", "minimum_kernel"),
    "lt": ("(float)(a[i] < b[i])", "lt_kernel"),
    "le": ("(float)(a[i] <= b[i])", "le_kernel"),
    "gt": ("(float)(a[i] > b[i])", "gt_kernel"),
    "ge": ("(float)(a[i] >= b[i])", "ge_kernel"),
    "eq": ("(float)(a[i] == b[i])", "eq_kernel"),
    "ne": ("(float)(a[i] != b[i])", "ne_kernel"),
    "mse_elementwise": ("(a[i] - b[i]) * (a[i] - b[i])", "mse_kernel"),
}

# Math-heavy TensorIterator lambdas. External calls are kept deliberately:
# cgeist can represent them faithfully, and the sweep records whether such a
# call prevents loop-to-Linalg conversion instead of silently simplifying the
# ATen operation.
MATH_UNARY = {
    "frac": ("x[i] - truncf(x[i])", "frac_kernel", "truncf"),
    "sinc": ("x[i] == 0.0f ? 1.0f : sinf(3.14159265358979323846f * x[i]) / (3.14159265358979323846f * x[i])", "sinc_kernel", "sinf"),
    "sinh": ("sinhf(x[i])", "sinh_kernel", "sinhf"),
    "cosh": ("coshf(x[i])", "cosh_kernel", "coshf"),
    "acosh": ("acoshf(x[i])", "acosh_kernel", "acoshf"),
    "asinh": ("asinhf(x[i])", "asinh_kernel", "asinhf"),
    "atanh": ("atanhf(x[i])", "atanh_kernel", "atanhf"),
    "exp2": ("exp2f(x[i])", "exp2_kernel", "exp2f"),
    "rsqrt": ("1.0f / sqrtf(x[i])", "rsqrt_kernel", "sqrtf"),
    "ceil": ("ceilf(x[i])", "ceil_kernel", "ceilf"),
    "floor": ("floorf(x[i])", "floor_kernel", "floorf"),
    "round": ("roundf(x[i])", "round_kernel", "roundf"),
    "sqrt": ("sqrtf(x[i])", "sqrt_kernel", "sqrtf"),
    "trunc": ("truncf(x[i])", "trunc_kernel", "truncf"),
    "sin": ("sinf(x[i])", "sin_kernel", "sinf"),
    "cos": ("cosf(x[i])", "cos_kernel", "cosf"),
    "tan": ("tanf(x[i])", "tan_kernel", "tanf"),
    "acos": ("acosf(x[i])", "acos_kernel", "acosf"),
    "asin": ("asinf(x[i])", "asin_kernel", "asinf"),
    "atan": ("atanf(x[i])", "atan_kernel", "atanf"),
    "erf": ("erff(x[i])", "erf_kernel", "erff"),
    "erfc": ("erfcf(x[i])", "erfc_kernel", "erfcf"),
    "exp": ("expf(x[i])", "exp_kernel", "expf"),
    "expm1": ("expm1f(x[i])", "expm1_kernel", "expm1f"),
    "log": ("logf(x[i])", "log_kernel", "logf"),
    "log10": ("log10f(x[i])", "log10_kernel", "log10f"),
    "log1p": ("log1pf(x[i])", "log1p_kernel", "log1pf"),
    "log2": ("log2f(x[i])", "log2_kernel", "log2f"),
    "lgamma": ("lgammaf(x[i])", "lgamma_kernel", "lgammaf"),
    "digamma": ("calc_digammaf(x[i])", "digamma_kernel", "calc_digammaf"),
    "trigamma": ("calc_trigammaf(x[i])", "trigamma_kernel", "calc_trigammaf"),
    "ndtri": ("calc_ndtrif(x[i])", "ndtri_kernel", "calc_ndtrif"),
    "log_ndtr": ("calc_log_ndtrf(x[i])", "log_ndtr_kernel", "calc_log_ndtrf"),
    "i0": ("calc_i0f(x[i])", "i0_kernel", "calc_i0f"),
    "i0e": ("calc_i0ef(x[i])", "i0e_kernel", "calc_i0ef"),
    "i1": ("calc_i1f(x[i])", "i1_kernel", "calc_i1f"),
    "i1e": ("calc_i1ef(x[i])", "i1e_kernel", "calc_i1ef"),
    "erfcx": ("calc_erfcxf(x[i])", "erfcx_kernel", "calc_erfcxf"),
    "erfinv": ("calc_erfinvf(x[i])", "erfinv_kernel", "calc_erfinvf"),
    "bessel_j0": ("bessel_j0_forwardf(x[i])", "bessel_j0_kernel", "bessel_j0_forwardf"),
    "bessel_j1": ("bessel_j1_forwardf(x[i])", "bessel_j1_kernel", "bessel_j1_forwardf"),
    "bessel_y0": ("bessel_y0_forwardf(x[i])", "bessel_y0_kernel", "bessel_y0_forwardf"),
    "bessel_y1": ("bessel_y1_forwardf(x[i])", "bessel_y1_kernel", "bessel_y1_forwardf"),
    "modified_bessel_i0": ("modified_bessel_i0_forwardf(x[i])", "modified_bessel_i0_kernel", "modified_bessel_i0_forwardf"),
    "modified_bessel_i1": ("modified_bessel_i1_forwardf(x[i])", "modified_bessel_i1_kernel", "modified_bessel_i1_forwardf"),
    "modified_bessel_k0": ("modified_bessel_k0_forwardf(x[i])", "modified_bessel_k0_kernel", "modified_bessel_k0_forwardf"),
    "modified_bessel_k1": ("modified_bessel_k1_forwardf(x[i])", "modified_bessel_k1_kernel", "modified_bessel_k1_forwardf"),
}

MATH_BINARY = {
    "atan2": ("atan2f(a[i], b[i])", "atan2_kernel", "atan2f"),
    "fmod": ("fmodf(a[i], b[i])", "fmod_kernel", "fmodf"),
    "remainder": ("remainderf(a[i], b[i])", "remainder_kernel", "remainderf"),
    "fmax": ("fmaxf(a[i], b[i])", "fmax_kernel", "fmaxf"),
    "fmin": ("fminf(a[i], b[i])", "fmin_kernel", "fminf"),
    "hypot": ("hypotf(a[i], b[i])", "hypot_kernel", "hypotf"),
    "nextafter": ("nextafterf(a[i], b[i])", "nextafter_kernel", "nextafterf"),
    "copysign": ("copysignf(a[i], b[i])", "copysign_kernel", "copysignf"),
    "pow": ("powf(a[i], b[i])", "pow_tensor_tensor_kernel", "powf"),
    "igamma": ("calc_igammaf(a[i], b[i])", "igamma_kernel", "calc_igammaf"),
    "igammac": ("calc_igammacf(a[i], b[i])", "igammac_kernel", "calc_igammacf"),
    "zeta": ("calc_zetaf(a[i], b[i])", "zeta_kernel", "calc_zetaf"),
    "chebyshev_polynomial_t": ("calc_chebyshev_tf(a[i], b[i])", "chebyshev_polynomial_t_kernel", "calc_chebyshev_tf"),
    "chebyshev_polynomial_u": ("calc_chebyshev_uf(a[i], b[i])", "chebyshev_polynomial_u_kernel", "calc_chebyshev_uf"),
    "chebyshev_polynomial_v": ("calc_chebyshev_vf(a[i], b[i])", "chebyshev_polynomial_v_kernel", "calc_chebyshev_vf"),
    "chebyshev_polynomial_w": ("calc_chebyshev_wf(a[i], b[i])", "chebyshev_polynomial_w_kernel", "calc_chebyshev_wf"),
    "laguerre_polynomial_l": ("calc_laguerre_lf(a[i], b[i])", "laguerre_polynomial_l_kernel", "calc_laguerre_lf"),
    "legendre_polynomial_p": ("calc_legendre_pf(a[i], b[i])", "legendre_polynomial_p_kernel", "calc_legendre_pf"),
    "hermite_polynomial_h": ("calc_hermite_hf(a[i], b[i])", "hermite_polynomial_h_kernel", "calc_hermite_hf"),
    "hermite_polynomial_he": ("calc_hermite_hef(a[i], b[i])", "hermite_polynomial_he_kernel", "calc_hermite_hef"),
}

INT_BINARY = {
    "bitwise_and_i32": ("a[i] & b[i]", "bitwise_and_kernel"),
    "bitwise_or_i32": ("a[i] | b[i]", "bitwise_or_kernel"),
    "bitwise_xor_i32": ("a[i] ^ b[i]", "bitwise_xor_kernel"),
    "lshift_i32": ("a[i] << b[i]", "lshift_kernel"),
    "rshift_i32": ("a[i] >> b[i]", "rshift_kernel"),
}

CUSTOM = {
    "smooth_l1_elementwise": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "smooth_l1_kernel",
        "float z = a[i] - b[i];\n"
        "    float az = z < 0.0f ? -z : z;\n"
        "    out[i] = az < beta ? 0.5f * z * z / beta : az - 0.5f * beta;",
        "float a[N], float b[N], float beta, float out[N]",
    ),
    "huber_elementwise": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "huber_kernel",
        "float z = a[i] - b[i];\n"
        "    float az = z < 0.0f ? -z : z;\n"
        "    out[i] = az < delta ? 0.5f * z * z : delta * (az - 0.5f * delta);",
        "float a[N], float b[N], float delta, float out[N]",
    ),
    "sigmoid_backward": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "sigmoid_backward_kernel",
        "out[i] = grad[i] * (1.0f - output[i]) * output[i];",
        "float grad[N], float output[N], float out[N]",
    ),
    "tanh_backward": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "tanh_backward_kernel",
        "out[i] = grad[i] * (1.0f - output[i] * output[i]);",
        "float grad[N], float output[N], float out[N]",
    ),
    "threshold_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "threshold_kernel",
        "out[i] = self[i] <= threshold ? 0.0f : grad[i];",
        "float grad[N], float self[N], float threshold, float out[N]",
    ),
    "elu_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "elu_backward_kernel",
        "out[i] = output[i] <= 0.0f ? grad[i] * (output[i] + alpha) * scale : grad[i] * scale;",
        "float grad[N], float output[N], float alpha, float scale, float out[N]",
    ),
    "softplus_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "softplus_backward_kernel",
        "float z = beta * self[i];\n"
        "    out[i] = z > threshold ? grad[i] : grad[i] * (1.0f - 1.0f / (1.0f + expf(z)));",
        "float grad[N], float self[N], float beta, float threshold, float out[N]",
    ),
    "addcmul": (
        "aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp", "addcmul_cpu",
        "out[i] = self[i] + value * x[i] * y[i];",
        "float self[N], float x[N], float y[N], float value, float out[N]",
    ),
    "addcdiv": (
        "aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp", "addcdiv_cpu",
        "out[i] = self[i] + value * x[i] / y[i];",
        "float self[N], float x[N], float y[N], float value, float out[N]",
    ),
    "fill": (
        "aten/src/ATen/native/cpu/FillKernel.cpp", "fill_kernel",
        "out[i] = value;", "float value, float out[N]",
    ),
    "linspace": (
        "aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp", "linspace_kernel",
        "out[i] = start + (float)i * step;",
        "float start, float step, float out[N]",
    ),
    "masked_scale": (
        "aten/src/ATen/native/cpu/AmpGradScalerKernels.cpp", "_amp_foreach_non_finite_check_and_unscale_cpu_kernel",
        "out[i] = x[i] * inv_scale;",
        "float x[N], float inv_scale, float out[N]",
    ),
    "lerp_scalar": (
        "aten/src/ATen/native/cpu/LerpKernel.cpp", "lerp_kernel_scalar",
        "out[i] = self[i] + weight * (end[i] - self[i]);",
        "float self[N], float end[N], float weight, float out[N]",
    ),
    "heaviside": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "heaviside_kernel",
        "out[i] = a[i] == 0.0f ? b[i] : (float)(a[i] > 0.0f);",
        "float a[N], float b[N], float out[N]",
    ),
    "logical_and": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logical_and_kernel",
        "out[i] = (float)((a[i] != 0.0f) && (b[i] != 0.0f));",
        "float a[N], float b[N], float out[N]",
    ),
    "logical_or": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logical_or_kernel",
        "out[i] = (float)((a[i] != 0.0f) || (b[i] != 0.0f));",
        "float a[N], float b[N], float out[N]",
    ),
    "logical_xor": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logical_xor_kernel",
        "out[i] = (float)((a[i] != 0.0f) != (b[i] != 0.0f));",
        "float a[N], float b[N], float out[N]",
    ),
    "addr_elementwise": (
        "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp", "addr_kernel",
        "out[i] = beta == 0.0f ? alpha * x[i] * y[i] : beta * self[i] + alpha * x[i] * y[i];",
        "float self[N], float x[N], float y[N], float beta, float alpha, float out[N]",
    ),
    "xlogy": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "xlogy_kernel",
        "out[i] = y[i] != y[i] ? y[i] : (x[i] == 0.0f ? 0.0f : x[i] * logf(y[i]));",
        "float x[N], float y[N], float out[N]",
    ),
    "xlog1py": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "xlog1py_kernel",
        "out[i] = y[i] != y[i] ? y[i] : (x[i] == 0.0f ? 0.0f : x[i] * log1pf(y[i]));",
        "float x[N], float y[N], float out[N]",
    ),
    "hardsigmoid_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "hardsigmoid_backward_kernel",
        "out[i] = self[i] > -3.0f && self[i] < 3.0f ? grad[i] * (1.0f / 6.0f) : 0.0f;",
        "float grad[N], float self[N], float out[N]",
    ),
    "hardtanh_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "hardtanh_backward_kernel",
        "out[i] = self[i] <= minval || self[i] >= maxval ? 0.0f : grad[i];",
        "float grad[N], float self[N], float minval, float maxval, float out[N]",
    ),
    "hardshrink": (
        "aten/src/ATen/native/cpu/Activation.cpp", "hardshrink_kernel",
        "out[i] = self[i] >= -lambd && self[i] <= lambd ? 0.0f : self[i];",
        "float self[N], float lambd, float out[N]",
    ),
    "softshrink": (
        "aten/src/ATen/native/cpu/Activation.cpp", "softshrink_kernel",
        "out[i] = self[i] > lambd ? self[i] - lambd : (self[i] < -lambd ? self[i] + lambd : self[i] * 0.0f);",
        "float self[N], float lambd, float out[N]",
    ),
    "shrink_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "shrink_backward_kernel",
        "out[i] = self[i] >= -lambd && self[i] <= lambd ? 0.0f : grad[i];",
        "float grad[N], float self[N], float lambd, float out[N]",
    ),
    "hardswish_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "hardswish_backward_kernel",
        "out[i] = self[i] <= -3.0f ? 0.0f : (self[i] < 3.0f ? grad[i] * (self[i] / 3.0f + 0.5f) : grad[i]);",
        "float grad[N], float self[N], float out[N]",
    ),
    "glu": (
        "aten/src/ATen/native/cpu/Activation.cpp", "glu_kernel",
        "out[i] = a[i] / (1.0f + expf(-b[i]));",
        "float a[N], float b[N], float out[N]",
    ),
    "glu_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "glu_backward_kernel",
        "out[i] = (1.0f - sigmoid_b[i]) * sigmoid_b[i] * grad[i] * a[i];",
        "float sigmoid_b[N], float grad[N], float a[N], float out[N]",
    ),
    "glu_jvp": (
        "aten/src/ATen/native/cpu/Activation.cpp", "glu_jvp_kernel",
        "float s = 1.0f / (1.0f + expf(-b[i]));\n    out[i] = da[i] * s + result[i] * (db[i] - s * db[i]);",
        "float result[N], float b[N], float da[N], float db[N], float out[N]",
    ),
    "silu_cpu": (
        "aten/src/ATen/native/cpu/Activation.cpp", "silu_kernel",
        "out[i] = x[i] / (1.0f + expf(-x[i]));",
        "float x[N], float out[N]",
    ),
    "silu_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "silu_backward_kernel",
        "float s = 1.0f / (1.0f + expf(-x[i]));\n    out[i] = grad[i] * s * (1.0f + x[i] * (1.0f - s));",
        "float grad[N], float x[N], float out[N]",
    ),
    "mish": (
        "aten/src/ATen/native/cpu/Activation.cpp", "mish_kernel",
        "out[i] = x[i] * tanhf(log1pf(expf(x[i])));",
        "float x[N], float out[N]",
    ),
    "mish_backward": (
        "aten/src/ATen/native/cpu/Activation.cpp", "mish_backward_kernel",
        "float s = 1.0f / (1.0f + expf(-x[i]));\n    float t = tanhf(log1pf(expf(x[i])));\n    out[i] = grad[i] * (t + x[i] * s * (1.0f - t * t));",
        "float grad[N], float x[N], float out[N]",
    ),
    "add_clamp": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "add_clamp_kernel",
        "float v = a[i] + alpha * b[i];\n    out[i] = v < minval ? minval : (v > maxval ? maxval : v);",
        "float a[N], float b[N], float alpha, float minval, float maxval, float out[N]",
    ),
    "div_trunc": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "div_trunc_kernel",
        "out[i] = truncf(a[i] / b[i]);",
        "float a[N], float b[N], float out[N]",
    ),
    "div_floor": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "div_floor_kernel",
        "out[i] = floorf(a[i] / b[i]);",
        "float a[N], float b[N], float out[N]",
    ),
    "logit_backward": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logit_backward_kernel",
        "out[i] = self[i] < eps || self[i] > 1.0f - eps ? 0.0f : grad[i] / (self[i] * (1.0f - self[i]));",
        "float grad[N], float self[N], float eps, float out[N]",
    ),
    "logaddexp": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logaddexp_kernel",
        "float m = a[i] > b[i] ? a[i] : b[i];\n    float d = a[i] - b[i];\n    if (d < 0.0f) d = -d;\n    out[i] = m + log1pf(expf(-d));",
        "float a[N], float b[N], float out[N]",
    ),
    "logaddexp2": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "logaddexp2_kernel",
        "float m = a[i] > b[i] ? a[i] : b[i];\n    float d = a[i] - b[i];\n    if (d < 0.0f) d = -d;\n    out[i] = m + log1pf(exp2f(-d)) * 1.4426950408889634f;",
        "float a[N], float b[N], float out[N]",
    ),
    "gcd_i32": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "gcd_kernel",
        "int x = a[i] < 0 ? -a[i] : a[i];\n    int y = b[i] < 0 ? -b[i] : b[i];\n    while (y != 0) { int r = x % y; x = y; y = r; }\n    out[i] = x;",
        "int a[N], int b[N], int out[N]",
    ),
    "lcm_i32": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "lcm_kernel",
        "int x = a[i] < 0 ? -a[i] : a[i];\n    int y = b[i] < 0 ? -b[i] : b[i];\n    int aa = x, bb = y;\n    while (y != 0) { int r = x % y; x = y; y = r; }\n    out[i] = x == 0 ? 0 : (aa / x) * bb;",
        "int a[N], int b[N], int out[N]",
    ),
    "ldexp": (
        "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp", "ldexp_kernel",
        "out[i] = ldexpf(a[i], exponent[i]);",
        "float a[N], int exponent[N], float out[N]",
    ),
    "log_sigmoid_cpu": (
        "aten/src/ATen/native/cpu/Activation.cpp", "log_sigmoid_cpu_kernel",
        "float ax = x[i] < 0.0f ? -x[i] : x[i];\n    buffer[i] = expf(-ax);\n    out[i] = (x[i] < 0.0f ? x[i] : 0.0f) - log1pf(buffer[i]);",
        "float x[N], float out[N], float buffer[N]",
    ),
    "log_sigmoid_backward_cpu": (
        "aten/src/ATen/native/cpu/Activation.cpp", "log_sigmoid_backward_cpu_kernel",
        "float neg = input[i] < 0.0f;\n    float max_deriv = neg ? 1.0f : 0.0f;\n    float sign = neg ? 1.0f : -1.0f;\n    out[i] = (max_deriv - sign * (buffer[i] / (1.0f + buffer[i]))) * grad[i];",
        "float input[N], float buffer[N], float grad[N], float out[N]",
    ),
    "gelu_cpu_tanh": (
        "aten/src/ATen/native/cpu/Activation.cpp", "GeluKernelImpl",
        "float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);\n    out[i] = 0.5f * x[i] * (1.0f + tanhf(inner));",
        "float x[N], float out[N]",
    ),
    "gelu_cpu_exact": (
        "aten/src/ATen/native/cpu/Activation.cpp", "GeluKernelImpl",
        "out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865475f));",
        "float x[N], float out[N]",
    ),
    "gelu_backward_cpu_tanh": (
        "aten/src/ATen/native/cpu/Activation.cpp", "GeluBackwardKernelImpl",
        "float x2 = x[i] * x[i];\n    float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x2);\n    float t = tanhf(inner);\n    float deriv = 0.5f * (1.0f + t) + 0.5f * x[i] * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x2);\n    out[i] = grad[i] * deriv;",
        "float grad[N], float x[N], float out[N]",
    ),
    "gelu_backward_cpu_exact": (
        "aten/src/ATen/native/cpu/Activation.cpp", "GeluBackwardKernelImpl",
        "float cdf = 0.5f * (1.0f + erff(x[i] * 0.7071067811865475f));\n    float pdf = 0.3989422804014327f * expf(-0.5f * x[i] * x[i]);\n    out[i] = grad[i] * (cdf + x[i] * pdf);",
        "float grad[N], float x[N], float out[N]",
    ),
    "round_decimals": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "round_decimals_kernel",
        "out[i] = roundf(x[i] * scale) / scale;",
        "float x[N], float scale, float out[N]",
    ),
    "angle_real": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "angle_kernel",
        "out[i] = x[i] < 0.0f ? 3.14159265358979323846f : 0.0f;",
        "float x[N], float out[N]",
    ),
    "angle_complex_scalarized": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "angle_kernel",
        "out[i] = atan2f(im[i], re[i]);",
        "float re[N], float im[N], float out[N]",
    ),
    "signbit": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "signbit_kernel",
        "out[i] = (float)(x[i] < 0.0f);",
        "float x[N], float out[N]",
    ),
    "bitwise_not_i32": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "bitwise_not_kernel",
        "out[i] = ~x[i];",
        "int x[N], int out[N]",
    ),
    "nan_to_num": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "nan_to_num_kernel",
        "out[i] = x[i] != x[i] ? nan_value : (x[i] > max_finite ? posinf_value : (x[i] < -max_finite ? neginf_value : x[i]));",
        "float x[N], float nan_value, float posinf_value, float neginf_value, float max_finite, float out[N]",
    ),
    "conj_complex_scalarized": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "conj_kernel",
        "out_re[i] = re[i];\n    out_im[i] = -im[i];",
        "float re[N], float im[N], float out_re[N], float out_im[N]",
    ),
    "entr": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "entr_kernel",
        "out[i] = x[i] < 0.0f ? nan_value : (x[i] == 0.0f ? 0.0f : (x[i] <= 1.0f ? -x[i] * logf(x[i]) : neg_inf));",
        "float x[N], float nan_value, float neg_inf, float out[N]",
    ),
    "sgn_complex_scalarized": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "sgn_kernel",
        "float mag = hypotf(re[i], im[i]);\n    out_re[i] = mag == 0.0f ? 0.0f : re[i] / mag;\n    out_im[i] = mag == 0.0f ? 0.0f : im[i] / mag;",
        "float re[N], float im[N], float out_re[N], float out_im[N]",
    ),
    "logit": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "logit_kernel",
        "float z = x[i] < eps ? eps : (x[i] > 1.0f - eps ? 1.0f - eps : x[i]);\n    out[i] = logf(z / (1.0f - z));",
        "float x[N], float eps, float out[N]",
    ),
    "polygamma": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "polygamma_kernel",
        "out[i] = calc_polygammaf(order, x[i]);",
        "float x[N], int order, float out[N]",
    ),
    "kaiser_window": (
        "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp", "kaiser_window_kernel",
        "out[i] = calc_kaiserf(x[i], beta);",
        "float x[N], float beta, float out[N]",
    ),
    "where_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "where_kernel_impl",
        "out[i] = condition[i] ? a[i] : b[i];",
        "int condition[N], float a[N], float b[N], float out[N]",
    ),
    "isposinf": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "isposinf_kernel_impl",
        "out[i] = (float)(x[i] > max_finite);",
        "float x[N], float max_finite, float out[N]",
    ),
    "isneginf": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "isneginf_kernel_impl",
        "out[i] = (float)(x[i] < -max_finite);",
        "float x[N], float max_finite, float out[N]",
    ),
    "clamp_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "clamp_kernel_impl",
        "out[i] = x[i] < minval[i] ? minval[i] : (x[i] > maxval[i] ? maxval[i] : x[i]);",
        "float x[N], float minval[N], float maxval[N], float out[N]",
    ),
    "clamp_scalar_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "clamp_scalar_kernel_impl",
        "out[i] = x[i] < minval ? minval : (x[i] > maxval ? maxval : x[i]);",
        "float x[N], float minval, float maxval, float out[N]",
    ),
    "clamp_min_scalar_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "clamp_min_scalar_kernel_impl",
        "out[i] = x[i] < minval ? minval : x[i];",
        "float x[N], float minval, float out[N]",
    ),
    "clamp_max_scalar_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "clamp_max_scalar_kernel_impl",
        "out[i] = x[i] > maxval ? maxval : x[i];",
        "float x[N], float maxval, float out[N]",
    ),
    "complex_scalarized": (
        "aten/src/ATen/native/cpu/ComplexKernel.cpp", "complex_kernel",
        "out_re[i] = real[i];\n    out_im[i] = imag[i];",
        "float real[N], float imag[N], float out_re[N], float out_im[N]",
    ),
    "polar_scalarized": (
        "aten/src/ATen/native/cpu/ComplexKernel.cpp", "polar_kernel",
        "out_re[i] = magnitude[i] * cosf(angle[i]);\n    out_im[i] = magnitude[i] * sinf(angle[i]);",
        "float magnitude[N], float angle[N], float out_re[N], float out_im[N]",
    ),
    "copy_cpu": (
        "aten/src/ATen/native/cpu/CopyKernel.cpp", "copy_kernel",
        "out[i] = input[i];",
        "float input[N], float out[N]",
    ),
    "linear_combination_cpu": (
        "aten/src/ATen/native/cpu/FunctionOfAMatrixUtilsKernel.cpp", "_compute_linear_combination_cpu_kernel",
        "float value = 0.0f;\n    for (int j = 0; j < 4; ++j) value += coefficients[j] * input[j][i];\n    out[i] = value;",
        "float input[4][N], float coefficients[4], float out[N]",
    ),
    "lerp_scalar_cpu": (
        "aten/src/ATen/native/cpu/LerpKernel.cpp", "lerp_scalar_kernel",
        "out[i] = self[i] + weight * (end[i] - self[i]);",
        "float self[N], float end[N], float weight, float out[N]",
    ),
    "lerp_tensor_cpu": (
        "aten/src/ATen/native/cpu/LerpKernel.cpp", "lerp_tensor_kernel",
        "out[i] = self[i] + weight[i] * (end[i] - self[i]);",
        "float self[N], float end[N], float weight[N], float out[N]",
    ),
    "smooth_l1_backward": (
        "aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp", "smooth_l1_backward_cpu_kernel",
        "float z = input[i] - target[i];\n    out[i] = z <= -beta ? -norm : (z >= beta ? norm : norm * z / beta);",
        "float input[N], float target[N], float norm, float beta, float out[N]",
    ),
    "huber_backward": (
        "aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp", "huber_backward_cpu_kernel",
        "float z = input[i] - target[i];\n    out[i] = z < -delta ? -norm * delta : (z > delta ? norm * delta : norm * z);",
        "float input[N], float target[N], float norm, float delta, float out[N]",
    ),
    "mse_backward": (
        "aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp", "mse_backward_cpu_kernel",
        "out[i] = value * (input[i] - target[i]);",
        "float input[N], float target[N], float value, float out[N]",
    ),
    "pow_tensor_scalar": (
        "aten/src/ATen/native/cpu/PowKernel.cpp", "pow_tensor_scalar_kernel",
        "out[i] = powf(input[i], exponent);",
        "float input[N], float exponent, float out[N]",
    ),
    "arange_cpu": (
        "aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp", "arange_kernel",
        "out[i] = start + (float)i * step;",
        "float start, float step, float out[N]",
    ),
    "renorm_scale_factor": (
        "aten/src/ATen/native/cpu/RenormKernel.cpp", "renorm_scale_factor_impl",
        "out[i] = norm[i] > maxnorm ? maxnorm / (norm[i] + 1.0e-7f) : 1.0f;",
        "float norm[N], float maxnorm, float out[N]",
    ),
    "airy_ai": (
        "aten/src/ATen/native/cpu/airy_ai.cpp", "airy_ai_kernel",
        "out[i] = calc_airy_aif(x[i]);",
        "float x[N], float out[N]",
    ),
    "scaled_modified_bessel_k0": (
        "aten/src/ATen/native/cpu/scaled_modified_bessel_k0.cpp", "scaled_modified_bessel_k0_kernel",
        "out[i] = calc_scaled_bessel_k0f(x[i]);",
        "float x[N], float out[N]",
    ),
    "scaled_modified_bessel_k1": (
        "aten/src/ATen/native/cpu/scaled_modified_bessel_k1.cpp", "scaled_modified_bessel_k1_kernel",
        "out[i] = calc_scaled_bessel_k1f(x[i]);",
        "float x[N], float out[N]",
    ),
    "spherical_bessel_j0": (
        "aten/src/ATen/native/cpu/spherical_bessel_j0.cpp", "spherical_bessel_j0_kernel",
        "out[i] = calc_spherical_bessel_j0f(x[i]);",
        "float x[N], float out[N]",
    ),
}

FULL = {
    "max_reduce_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "max_kernel_impl",
        """#ifndef N
#define N 4096
#endif
void aten_max_reduce_cpu(float x[N], float out[1]) {
#pragma scop
  float value = x[0];
  for (int i = 1; i < N; ++i) value = x[i] > value ? x[i] : value;
  out[0] = value;
#pragma endscop
}
""",
    ),
    "min_reduce_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "min_kernel_impl",
        """#ifndef N
#define N 4096
#endif
void aten_min_reduce_cpu(float x[N], float out[1]) {
#pragma scop
  float value = x[0];
  for (int i = 1; i < N; ++i) value = x[i] < value ? x[i] : value;
  out[0] = value;
#pragma endscop
}
""",
    ),
    "aminmax_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "aminmax_kernel",
        """#ifndef N
#define N 4096
#endif
void aten_aminmax_cpu(float x[N], float out_min[1], float out_max[1]) {
#pragma scop
  float lo = x[0], hi = x[0];
  for (int i = 1; i < N; ++i) {
    lo = x[i] < lo ? x[i] : lo;
    hi = x[i] > hi ? x[i] : hi;
  }
  out_min[0] = lo; out_max[0] = hi;
#pragma endscop
}
""",
    ),
    "mode_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "mode_kernel_impl",
        """#ifndef N
#define N 64
#endif
void aten_mode_cpu(float x[N], float out[1], int out_index[1]) {
  float work[N];
  int indices[N];
  for (int i = 0; i < N; ++i) { work[i] = x[i]; indices[i] = i; }
  for (int i = 1; i < N; ++i) {
    float v = work[i]; int idx = indices[i]; int j = i - 1;
    while (j >= 0 && work[j] > v) {
      work[j + 1] = work[j]; indices[j + 1] = indices[j]; --j;
    }
    work[j + 1] = v; indices[j + 1] = idx;
  }
  int best_count = 1, count = 1, best = 0;
  for (int i = 1; i < N; ++i) {
    if (work[i] == work[i - 1]) ++count; else count = 1;
    if (count > best_count) { best_count = count; best = i; }
  }
  out[0] = work[best]; out_index[0] = indices[best];
}
""",
    ),
    "isin_default_cpu": (
        "aten/src/ATen/native/cpu/TensorCompareKernel.cpp", "isin_default_kernel_cpu",
        """#ifndef N
#define N 4096
#endif
#ifndef M
#define M 257
#endif
void aten_isin_default_cpu(float elements[N], float test[M], int out[N]) {
#pragma scop
  for (int i = 0; i < N; ++i) {
    int found = 0;
    for (int j = 0; j < M; ++j) found |= elements[i] == test[j];
    out[i] = found;
  }
#pragma endscop
}
""",
    ),
    "min_all_cpu": (
        "aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp", "min_all_kernel_impl",
        """#ifndef N
#define N 4096
#endif
void aten_min_all_cpu(float x[N], float out[1]) {
  float value = x[0];
  for (int i = 1; i < N; ++i) value = x[i] < value ? x[i] : value;
  out[0] = value;
}
""",
    ),
    "max_all_cpu": (
        "aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp", "max_all_kernel_impl",
        """#ifndef N
#define N 4096
#endif
void aten_max_all_cpu(float x[N], float out[1]) {
  float value = x[0];
  for (int i = 1; i < N; ++i) value = x[i] > value ? x[i] : value;
  out[0] = value;
}
""",
    ),
    "aminmax_allreduce_cpu": (
        "aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp", "aminmax_allreduce_kernel",
        """#ifndef N
#define N 4096
#endif
void aten_aminmax_allreduce_cpu(float x[N], float out_min[1], float out_max[1]) {
  float lo = x[0], hi = x[0];
  for (int i = 1; i < N; ++i) {
    lo = x[i] < lo ? x[i] : lo; hi = x[i] > hi ? x[i] : hi;
  }
  out_min[0] = lo; out_max[0] = hi;
}
""",
    ),
    "amp_update_scale_cpu": (
        "aten/src/ATen/native/cpu/AmpGradScalerKernels.cpp", "_amp_update_scale_cpu_kernel",
        """void aten_amp_update_scale_cpu(float scale[1], int tracker[1],
    float found_inf[1], float growth, float backoff, int interval) {
  if (found_inf[0] != 0.0f) { scale[0] *= backoff; tracker[0] = 0; }
  else {
    int successful = tracker[0] + 1;
    if (successful == interval) { scale[0] *= growth; tracker[0] = 0; }
    else tracker[0] = successful;
  }
}
""",
    ),
    "fused_adagrad_cpu": (
        "aten/src/ATen/native/cpu/FusedAdagradKernel.cpp", "fused_adagrad_kernel",
        """#ifndef N
#define N 4096
#endif
extern float sqrtf(float);
void aten_fused_adagrad_cpu(float param[N], float grad[N], float state_sum[N],
    float lr, float lr_decay, float weight_decay, float eps, float step,
    float grad_scale, int maximize) {
  float clr = lr / (1.0f + (step - 1.0f) * lr_decay);
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale;
    grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    state_sum[i] += g * g;
    param[i] -= clr * g / (sqrtf(state_sum[i]) + eps);
  }
}
""",
    ),
    "fused_sgd_cpu": (
        "aten/src/ATen/native/cpu/FusedSGDKernel.cpp", "fused_sgd_kernel",
        """#ifndef N
#define N 4096
#endif
void aten_fused_sgd_cpu(float param[N], float grad[N], float momentum_buffer[N],
    float lr, float momentum, float dampening, float weight_decay,
    float grad_scale, int maximize, int first_step, int nesterov) {
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale; grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    if (momentum != 0.0f) {
      momentum_buffer[i] = first_step ? g :
          momentum_buffer[i] * momentum + g * (1.0f - dampening);
      g = nesterov ? g + momentum * momentum_buffer[i] : momentum_buffer[i];
    }
    param[i] -= lr * g;
  }
}
""",
    ),
    "fused_adam_cpu": (
        "aten/src/ATen/native/cpu/FusedAdamKernel.cpp", "fused_adam_kernel",
        """#ifndef N
#define N 4096
#endif
extern float sqrtf(float);
void aten_fused_adam_cpu(float param[N], float grad[N], float exp_avg[N],
    float exp_avg_sq[N], float max_exp_avg_sq[N], float lr, float beta1,
    float beta2, float bias1, float bias2_sqrt, float weight_decay, float eps,
    float grad_scale, int maximize, int amsgrad) {
  float step_size = lr / bias1;
  for (int i = 0; i < N; ++i) {
    float g = grad[i] / grad_scale; grad[i] = g;
    if (maximize) g = -g;
    if (weight_decay != 0.0f) g += param[i] * weight_decay;
    exp_avg[i] += (1.0f - beta1) * (g - exp_avg[i]);
    exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
    float variance = exp_avg_sq[i];
    if (amsgrad) {
      max_exp_avg_sq[i] = max_exp_avg_sq[i] > variance ? max_exp_avg_sq[i] : variance;
      variance = max_exp_avg_sq[i];
    }
    param[i] -= step_size * exp_avg[i] / (sqrtf(variance) / bias2_sqrt + eps);
  }
}
""",
    ),
}


def body(name: str, args: str, statement: str, needs_exp: bool = False,
         extern: str | None = None, binary_extern: bool = False) -> str:
    unary_calls = set()
    if needs_exp or "expf(" in statement:
        unary_calls.add("expf")
    for function in ("logf", "log1pf", "tanhf", "truncf", "floorf", "roundf", "exp2f", "erff", "sinf", "cosf"):
        if f"{function}(" in statement:
            unary_calls.add(function)
    prefix = "".join(
        f"extern ATEN_CONST float {function}(float);\n"
        for function in sorted(unary_calls)
    )
    if "ldexpf(" in statement:
        prefix += "extern ATEN_CONST float ldexpf(float, int);\n"
    if "powf(" in statement:
        prefix += "extern ATEN_CONST float powf(float, float);\n"
    if "atan2f(" in statement:
        prefix += "extern ATEN_CONST float atan2f(float, float);\n"
    if "hypotf(" in statement:
        prefix += "extern ATEN_CONST float hypotf(float, float);\n"
    if "calc_polygammaf(" in statement:
        prefix += "extern ATEN_CONST float calc_polygammaf(int, float);\n"
    if "calc_kaiserf(" in statement:
        prefix += "extern ATEN_CONST float calc_kaiserf(float, float);\n"
    for special in (
        "calc_airy_aif", "calc_scaled_bessel_k0f",
        "calc_scaled_bessel_k1f", "calc_spherical_bessel_j0f",
    ):
        if f"{special}(" in statement:
            prefix += f"extern ATEN_CONST float {special}(float);\n"
    if extern:
        parameters = "float, float" if binary_extern else "float"
        if extern not in unary_calls:
            prefix += f"extern ATEN_CONST float {extern}({parameters});\n"
    return (
        f"/* Fixed-shape scalar specialization extracted from pinned ATen. */\n"
        f"#ifndef N\n#define N 4096\n#endif\n"
        f"#define ATEN_CONST __attribute__((const))\n{prefix}"
        f"void aten_{name}({args}) {{\n#pragma scop\n"
        f"  for (int i = 0; i < N; ++i) {{\n    {statement}\n  }}\n"
        f"#pragma endscop\n}}\n"
    )


def main() -> None:
    rows: list[dict[str, str]] = []
    unary_source = "aten/src/ATen/native/cpu/UnaryOpsKernel.cpp"
    binary_source = "aten/src/ATen/native/cpu/BinaryOpsKernel.cpp"
    for name, (expr, token) in UNARY.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(body(name, "float x[N], float out[N]", f"out[i] = {expr};"))
        rows.append({"kernel": kernel, "source": unary_source, "token": token})
    for name, (expr, token) in BINARY.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(body(name, "float a[N], float b[N], float out[N]", f"out[i] = {expr};"))
        rows.append({"kernel": kernel, "source": binary_source, "token": token})
    for name, (expr, token, extern) in MATH_UNARY.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(
            body(name, "float x[N], float out[N]", f"out[i] = {expr};",
                 extern=extern)
        )
        rows.append({"kernel": kernel, "source": unary_source, "token": token})
    for name, (expr, token, extern) in MATH_BINARY.items():
        kernel = f"aten_{name}"
        source = (
            "aten/src/ATen/native/cpu/PowKernel.cpp"
            if name == "pow" else binary_source
        )
        (OUT / f"{kernel}.c").write_text(
            body(name, "float a[N], float b[N], float out[N]",
                 f"out[i] = {expr};", extern=extern, binary_extern=True)
        )
        rows.append({"kernel": kernel, "source": source, "token": token})
    for name, (expr, token) in INT_BINARY.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(
            body(name, "int a[N], int b[N], int out[N]", f"out[i] = {expr};")
        )
        rows.append({"kernel": kernel, "source": binary_source, "token": token})
    for name, (source, token, statement, args) in CUSTOM.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(body(name, args, statement, "expf(" in statement))
        rows.append({"kernel": kernel, "source": source, "token": token})
    for name, (source, token, source_text) in FULL.items():
        kernel = f"aten_{name}"
        (OUT / f"{kernel}.c").write_text(source_text)
        rows.append({"kernel": kernel, "source": source, "token": token})
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kernel", "source", "token"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} C fixtures and {MANIFEST}")


if __name__ == "__main__":
    main()
