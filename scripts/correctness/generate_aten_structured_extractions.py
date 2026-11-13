#!/usr/bin/env python3
"""Generate fixed-shape standalone C forms for structured ATen CPU kernels."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "issues/aten_c_kernels"
MANIFEST = OUT / "generated_structured_provenance.csv"
UPSAMPLE = "aten/src/ATen/native/cpu/UpSampleKernel.cpp"
UPSAMPLE_MORE = "aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp"
PADDING = "aten/src/ATen/native/cpu/PaddingKernel.cpp"
REDUCE = "aten/src/ATen/native/cpu/ReduceOpsKernel.cpp"
DIST = "aten/src/ATen/native/cpu/DistributionKernels.cpp"
INDEX = "aten/src/ATen/native/cpu/IndexKernel.cpp"
SCATTER = "aten/src/ATen/native/cpu/ScatterGatherKernel.cpp"
ADAPTIVE_AVG_POOL = "aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp"
ADAPTIVE_MAX_POOL = "aten/src/ATen/native/cpu/AdaptiveMaxPoolKernel.cpp"
AVG_POOL = "aten/src/ATen/native/cpu/AvgPoolKernel.cpp"
MAX_POOL = "aten/src/ATen/native/cpu/MaxPoolKernel.cpp"
MAX_POOL_1D = "aten/src/ATen/native/cpu/MaxPooling.cpp"


def macros(rank: int) -> str:
    lines = ["#ifndef B", "#define B 1", "#endif", "#ifndef C", "#define C 2", "#endif"]
    for d in range(rank):
        lines += [
            f"#ifndef I{d}", f"#define I{d} {4 + d}", "#endif",
            f"#ifndef O{d}", f"#define O{d} {7 + d}", "#endif",
        ]
    return "\n".join(lines) + "\n"


def array_size(prefix: str, rank: int) -> str:
    return "*".join(["B", "C"] + [f"{prefix}{d}" for d in range(rank)])


def flatten(indices: list[str], dims: list[str]) -> str:
    value = indices[0]
    for index, dim in zip(indices[1:], dims[1:]):
        value = f"(({value})*{dim}+{index})"
    return value


def nearest(rank: int, exact: bool, backward: bool) -> str:
    suffix = f"{rank}d"
    exact_name = "_exact" if exact else ""
    direction = "_backward" if backward else ""
    name = f"aten_upsample_nearest{exact_name}{suffix}{direction}_cpu"
    token = (
        f"_upsample_nearest_exact{suffix}{direction}_kernel_impl"
        if exact else f"upsample_nearest{suffix}{direction}_kernel_impl"
    )
    source = UPSAMPLE_MORE if backward else UPSAMPLE
    inputs = [f"i{d}" for d in range(rank)]
    outputs = [f"o{d}" for d in range(rank)]
    spatial_loops = "\n".join(
        f"{'  ' * (2 + d)}for (int o{d} = 0; o{d} < O{d}; ++o{d}) {{"
        for d in range(rank)
    )
    indent = "  " * (2 + rank)
    index_lines = []
    for d in range(rank):
        expr = (
            f"((2 * o{d} + 1) * I{d}) / (2 * O{d})"
            if exact else f"(o{d} * I{d}) / O{d}"
        )
        index_lines.append(f"{indent}int i{d} = {expr};")
        index_lines.append(f"{indent}if (i{d} >= I{d}) i{d} = I{d} - 1;")
    out_index = flatten(["n", "c", *outputs], ["B", "C", *[f"O{d}" for d in range(rank)]])
    in_index = flatten(["n", "c", *inputs], ["B", "C", *[f"I{d}" for d in range(rank)]])
    if backward:
        signature = (
            f"float grad_output[{array_size('O', rank)}], "
            f"float grad_input[{array_size('I', rank)}]"
        )
        init = (
            f"  for (int p = 0; p < {array_size('I', rank)}; ++p) "
            "grad_input[p] = 0.0f;\n"
        )
        operation = f"{indent}grad_input[{in_index}] += grad_output[{out_index}];"
    else:
        signature = (
            f"float input[{array_size('I', rank)}], "
            f"float output[{array_size('O', rank)}]"
        )
        init = ""
        operation = f"{indent}output[{out_index}] = input[{in_index}];"
    code = (
        f"/* Fixed-shape ATen nearest{'-exact' if exact else ''} {rank}D"
        f"{' backward' if backward else ''}. */\n{macros(rank)}"
        f"void {name}({signature}) {{\n#pragma scop\n{init}"
        "  for (int n = 0; n < B; ++n) {\n"
        "    for (int c = 0; c < C; ++c) {\n"
        f"{spatial_loops}\n" + "\n".join(index_lines) + "\n"
        f"{operation}\n"
        + "\n".join(
            f"{'  ' * depth}}}" for depth in range(1 + rank, -1, -1)
        )
        + "\n#pragma endscop\n}\n"
    )
    return name, source, token, code


def linear(rank: int, backward: bool) -> tuple[str, str, str, str]:
    kind = {1: "linear", 2: "bilinear", 3: "trilinear"}[rank]
    direction = "_backward" if backward else ""
    name = f"aten_upsample_{kind}{rank}d{direction}_cpu"
    token = f"upsample_{kind}{rank}d{direction}_kernel_impl"
    source = UPSAMPLE_MORE if backward else UPSAMPLE
    outputs = [f"o{d}" for d in range(rank)]
    spatial_loops = "\n".join(
        f"{'  ' * (2 + d)}for (int o{d} = 0; o{d} < O{d}; ++o{d}) {{"
        for d in range(rank)
    )
    indent = "  " * (2 + rank)
    coordinates = []
    for d in range(rank):
        coordinates += [
            f"{indent}float s{d} = ((float)o{d} + 0.5f) * (float)I{d} / (float)O{d} - 0.5f;",
            f"{indent}if (s{d} < 0.0f) s{d} = 0.0f;",
            f"{indent}int i{d}0 = (int)s{d};",
            f"{indent}int i{d}1 = i{d}0 + 1 < I{d} ? i{d}0 + 1 : i{d}0;",
            f"{indent}float w{d}1 = s{d} - (float)i{d}0;",
            f"{indent}float w{d}0 = 1.0f - w{d}1;",
        ]
    out_index = flatten(
        ["n", "c", *outputs], ["B", "C", *[f"O{d}" for d in range(rank)]]
    )
    terms = []
    for choices in itertools.product((0, 1), repeat=rank):
        indices = [f"i{d}{choice}" for d, choice in enumerate(choices)]
        in_index = flatten(
            ["n", "c", *indices], ["B", "C", *[f"I{d}" for d in range(rank)]]
        )
        weight = "*".join(f"w{d}{choice}" for d, choice in enumerate(choices))
        terms.append((in_index, weight))
    if backward:
        signature = (
            f"float grad_output[{array_size('O', rank)}], "
            f"float grad_input[{array_size('I', rank)}]"
        )
        init = (
            f"  for (int p = 0; p < {array_size('I', rank)}; ++p) "
            "grad_input[p] = 0.0f;\n"
        )
        operations = "\n".join(
            f"{indent}grad_input[{index}] += grad_output[{out_index}] * {weight};"
            for index, weight in terms
        )
    else:
        signature = (
            f"float input[{array_size('I', rank)}], "
            f"float output[{array_size('O', rank)}]"
        )
        init = ""
        expression = " + ".join(
            f"input[{index}] * {weight}" for index, weight in terms
        )
        operations = f"{indent}output[{out_index}] = {expression};"
    code = (
        f"/* Fixed-shape ATen {kind} {rank}D align_corners=false"
        f"{' backward' if backward else ''}. */\n{macros(rank)}"
        f"void {name}({signature}) {{\n#pragma scop\n{init}"
        "  for (int n = 0; n < B; ++n) {\n"
        "    for (int c = 0; c < C; ++c) {\n"
        f"{spatial_loops}\n" + "\n".join(coordinates) + "\n"
        f"{operations}\n"
        + "\n".join(
            f"{'  ' * depth}}}" for depth in range(1 + rank, -1, -1)
        )
        + "\n#pragma endscop\n}\n"
    )
    return name, source, token, code


def filtered_2d(kind: str, backward: bool) -> tuple[str, str, str, str]:
    """Separable fixed-shape antialiased 2D resampling."""
    direction = "_backward" if backward else ""
    name = f"aten_upsample_{kind}2d_aa{direction}_cpu"
    token = f"upsample_{kind}2d_aa{direction}_kernel_impl"
    source = UPSAMPLE if True else UPSAMPLE_MORE
    # ATen keeps AA backward implementations in UpSampleKernel.cpp.
    if kind == "bilinear":
        radius = "1.0f"
        kernel = "ax < 1.0f ? 1.0f - ax : 0.0f"
    elif kind == "bicubic":
        radius = "2.0f"
        kernel = (
            "ax < 1.0f ? ((1.5f * ax - 2.5f) * ax * ax + 1.0f) : "
            "(ax < 2.0f ? ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f : 0.0f)"
        )
    else:
        radius = "3.0f"
        kernel = (
            "ax == 0.0f ? 1.0f : (ax < 3.0f ? "
            "sinf(3.14159265358979323846f * ax) * "
            "sinf(3.14159265358979323846f * ax / 3.0f) / "
            "(3.289868133696453f * ax * ax) : 0.0f)"
        )
    operation = (
        "        grad_input[((n*C+c)*I0+iy)*I1+ix] += "
        "grad_output[((n*C+c)*O0+oy)*O1+ox] * wy * wx / norm;\n"
        if backward else
        "        value += input[((n*C+c)*I0+iy)*I1+ix] * wy * wx;\n"
    )
    final = (
        "" if backward else
        "      output[((n*C+c)*O0+oy)*O1+ox] = value / norm;\n"
    )
    signature = (
        "float grad_output[B*C*O0*O1], float grad_input[B*C*I0*I1]"
        if backward else
        "float input[B*C*I0*I1], float output[B*C*O0*O1]"
    )
    init = (
        "  for (int p = 0; p < B*C*I0*I1; ++p) grad_input[p] = 0.0f;\n"
        if backward else ""
    )
    code = f"""/* Fixed-shape ATen antialiased {kind} 2D{' backward' if backward else ''}. */
{macros(2)}extern float sinf(float);
void {name}({signature}) {{
{init}  float sy_scale = (float)I0 / (float)O0;
  float sx_scale = (float)I1 / (float)O1;
  float fy_scale = sy_scale > 1.0f ? sy_scale : 1.0f;
  float fx_scale = sx_scale > 1.0f ? sx_scale : 1.0f;
  for (int n = 0; n < B; ++n) for (int c = 0; c < C; ++c)
  for (int oy = 0; oy < O0; ++oy) for (int ox = 0; ox < O1; ++ox) {{
    float sy = ((float)oy + 0.5f) * sy_scale - 0.5f;
    float sx = ((float)ox + 0.5f) * sx_scale - 0.5f;
    float norm = 0.0f;
    {'float value = 0.0f;' if not backward else ''}
    for (int iy = 0; iy < I0; ++iy) for (int ix = 0; ix < I1; ++ix) {{
      float ay = sy - (float)iy; if (ay < 0.0f) ay = -ay; ay /= fy_scale;
      float ax = sx - (float)ix; if (ax < 0.0f) ax = -ax; ax /= fx_scale;
      float wy = ay < {radius} ? ({kernel.replace('ax', 'ay')}) : 0.0f;
      float wx = ax < {radius} ? ({kernel}) : 0.0f;
      norm += wy * wx;
    }}
    for (int iy = 0; iy < I0; ++iy) for (int ix = 0; ix < I1; ++ix) {{
      float ay = sy - (float)iy; if (ay < 0.0f) ay = -ay; ay /= fy_scale;
      float ax = sx - (float)ix; if (ax < 0.0f) ax = -ax; ax /= fx_scale;
      float wy = ay < {radius} ? ({kernel.replace('ax', 'ay')}) : 0.0f;
      float wx = ax < {radius} ? ({kernel}) : 0.0f;
{operation}    }}
{final}  }}
}}
"""
    return name, source, token, code


def bicubic_2d() -> tuple[str, str, str, str]:
    name = "aten_upsample_bicubic2d_cpu"
    token = "upsample_bicubic2d_kernel_impl"
    code = f"""/* Fixed-shape ATen bicubic 2D, align_corners=false, a=-0.75. */
{macros(2)}static float aten_cubic_weight(float x) {{
  if (x < 0.0f) x = -x;
  if (x < 1.0f) return ((1.25f * x - 2.25f) * x * x + 1.0f);
  if (x < 2.0f) return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
  return 0.0f;
}}
void {name}(float input[B*C*I0*I1], float output[B*C*O0*O1]) {{
  for (int n = 0; n < B; ++n) for (int c = 0; c < C; ++c)
  for (int oy = 0; oy < O0; ++oy) for (int ox = 0; ox < O1; ++ox) {{
    float sy = ((float)oy + 0.5f) * (float)I0 / (float)O0 - 0.5f;
    float sx = ((float)ox + 0.5f) * (float)I1 / (float)O1 - 0.5f;
    int by = (int)sy, bx = (int)sx;
    if (sy < 0.0f && sy != (float)by) --by;
    if (sx < 0.0f && sx != (float)bx) --bx;
    float value = 0.0f;
    for (int ky = -1; ky <= 2; ++ky) for (int kx = -1; kx <= 2; ++kx) {{
      int iy = by + ky; if (iy < 0) iy = 0; if (iy >= I0) iy = I0 - 1;
      int ix = bx + kx; if (ix < 0) ix = 0; if (ix >= I1) ix = I1 - 1;
      value += input[((n*C+c)*I0+iy)*I1+ix] *
          aten_cubic_weight(sy-(float)(by+ky)) *
          aten_cubic_weight(sx-(float)(bx+kx));
    }}
    output[((n*C+c)*O0+oy)*O1+ox] = value;
  }}
}}
"""
    return name, UPSAMPLE, token, code


def padding(rank: int, mode: str, backward: bool) -> tuple[str, str, str, str]:
    direction = "_backward" if backward else ""
    name = f"aten_{mode}_pad{rank}d{direction}_cpu"
    token = f"{mode}_pad{rank}d{direction}_kernel_impl"
    macro_text = ["#ifndef B\n#define B 1\n#endif", "#ifndef C\n#define C 2\n#endif"]
    for d in range(rank):
        macro_text += [
            f"#ifndef I{d}\n#define I{d} {4+d}\n#endif",
            f"#ifndef P{d}\n#define P{d} 2\n#endif",
            f"#define O{d} (I{d}+2*P{d})",
        ]
    outputs = [f"o{d}" for d in range(rank)]
    inputs = [f"i{d}" for d in range(rank)]
    loops = "\n".join(
        f"{'  ' * (2+d)}for (int o{d}=0; o{d}<O{d}; ++o{d}) {{"
        for d in range(rank)
    )
    indent = "  " * (2 + rank)
    maps = []
    for d in range(rank):
        maps.append(f"{indent}int i{d} = o{d} - P{d};")
        if mode == "reflection":
            maps += [
                f"{indent}if (i{d} < 0) i{d} = -i{d};",
                f"{indent}if (i{d} >= I{d}) i{d} = 2*I{d}-2-i{d};",
            ]
        else:
            maps += [
                f"{indent}if (i{d} < 0) i{d} = 0;",
                f"{indent}if (i{d} >= I{d}) i{d} = I{d}-1;",
            ]
    in_index = flatten(["n", "c", *inputs], ["B", "C", *[f"I{d}" for d in range(rank)]])
    out_index = flatten(["n", "c", *outputs], ["B", "C", *[f"O{d}" for d in range(rank)]])
    if backward:
        signature = f"float grad_output[{array_size('O',rank)}], float grad_input[{array_size('I',rank)}]"
        init = f"  for (int p=0; p<{array_size('I',rank)}; ++p) grad_input[p]=0.0f;\n"
        op = f"{indent}grad_input[{in_index}] += grad_output[{out_index}];"
    else:
        signature = f"float input[{array_size('I',rank)}], float output[{array_size('O',rank)}]"
        init = ""
        op = f"{indent}output[{out_index}] = input[{in_index}];"
    closes = "\n".join(f"{'  '*depth}}}" for depth in range(1+rank,-1,-1))
    code = (
        f"/* Fixed-shape ATen {mode} padding {rank}D{direction}. */\n"
        + "\n".join(macro_text) + "\n"
        + f"void {name}({signature}) {{\n{init}"
        + "  for (int n=0; n<B; ++n) {\n    for (int c=0; c<C; ++c) {\n"
        + loops + "\n" + "\n".join(maps) + "\n" + op + "\n"
        + closes + "\n}\n"
    )
    return name, PADDING, token, code


def reduction_entries() -> list[tuple[str, str, str, str]]:
    header = "#ifndef R\n#define R 32\n#endif\n#ifndef K\n#define K 64\n#endif\n"
    specs = {
        "std_var_cpu": ("std_var_kernel_impl", """extern float sqrtf(float);
void aten_std_var_cpu(float x[R][K], int correction, float out[R]) {
  for (int r=0;r<R;++r) { float sum=0.0f; for(int k=0;k<K;++k) sum+=x[r][k];
    float mean=sum/(float)K, sq=0.0f;
    for(int k=0;k<K;++k){float d=x[r][k]-mean;sq+=d*d;}
    out[r]=sqrtf(sq/(float)(K-correction)); }
}"""),
        "norm_cpu": ("norm_kernel_tensor_iterator_impl", """extern float sqrtf(float);
void aten_norm_cpu(float x[R][K], float out[R]) {
  for(int r=0;r<R;++r){float s=0.0f;for(int k=0;k<K;++k)s+=x[r][k]*x[r][k];out[r]=sqrtf(s);}
}"""),
        "powsum_cpu": ("powsum_kernel_tensor_iterator_impl", """extern float powf(float,float);
void aten_powsum_cpu(float x[R][K], float p, float out[R]) {
  for(int r=0;r<R;++r){float s=0.0f;for(int k=0;k<K;++k){float a=x[r][k]<0?-x[r][k]:x[r][k];s+=powf(a,p);}out[r]=s;}
}"""),
        "and_reduce_cpu": ("and_kernel_impl", """void aten_and_reduce_cpu(int x[R][K], int out[R]) {
  for(int r=0;r<R;++r){int v=1;for(int k=0;k<K;++k)v=v&&(x[r][k]!=0);out[r]=v;}
}"""),
        "or_reduce_cpu": ("or_kernel_impl", """void aten_or_reduce_cpu(int x[R][K], int out[R]) {
  for(int r=0;r<R;++r){int v=0;for(int k=0;k<K;++k)v=v||(x[r][k]!=0);out[r]=v;}
}"""),
        "min_values_cpu": ("min_values_kernel_impl", """void aten_min_values_cpu(float x[R][K], float out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];for(int k=1;k<K;++k)v=x[r][k]<v?x[r][k]:v;out[r]=v;}
}"""),
        "max_values_cpu": ("max_values_kernel_impl", """void aten_max_values_cpu(float x[R][K], float out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];for(int k=1;k<K;++k)v=x[r][k]>v?x[r][k]:v;out[r]=v;}
}"""),
        "argmax_cpu": ("argmax_kernel_impl", """void aten_argmax_cpu(float x[R][K], int out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];int best=0;for(int k=1;k<K;++k)if(x[r][k]>v){v=x[r][k];best=k;}out[r]=best;}
}"""),
        "argmin_cpu": ("argmin_kernel_impl", """void aten_argmin_cpu(float x[R][K], int out[R]) {
  for(int r=0;r<R;++r){float v=x[r][0];int best=0;for(int k=1;k<K;++k)if(x[r][k]<v){v=x[r][k];best=k;}out[r]=best;}
}"""),
        "xor_sum_cpu": ("xor_sum_kernel_impl", """void aten_xor_sum_cpu(int x[R][K], int out[R]) {
  for(int r=0;r<R;++r){int v=0;for(int k=0;k<K;++k)v^=x[r][k];out[r]=v;}
}"""),
        "cumprod_cpu": ("cumprod_cpu_kernel", """void aten_cumprod_cpu(float x[R][K], float out[R][K]) {
  for(int r=0;r<R;++r){float v=1.0f;for(int k=0;k<K;++k){v*=x[r][k];out[r][k]=v;}}
}"""),
        "logcumsumexp_cpu": ("logcumsumexp_cpu_kernel", """extern float expf(float); extern float log1pf(float);
void aten_logcumsumexp_cpu(float x[R][K], float out[R][K]) {
  for(int r=0;r<R;++r){float v=x[r][0];out[r][0]=v;for(int k=1;k<K;++k){float m=v>x[r][k]?v:x[r][k];float d=v-x[r][k];if(d<0)d=-d;v=m+log1pf(expf(-d));out[r][k]=v;}}
}"""),
    }
    return [
        (f"aten_{name}", REDUCE, token, header + code + "\n")
        for name, (token, code) in specs.items()
    ]


def distribution_entries() -> list[tuple[str, str, str, str]]:
    h = "#ifndef N\n#define N 4096\n#endif\n"
    specs = {
        "bernoulli_tensor_cpu": ("bernoulli_tensor_kernel",
            "void aten_bernoulli_tensor_cpu(float uniform[N],float probability[N],float out[N]){for(int i=0;i<N;++i)out[i]=(float)(uniform[i]<probability[i]);}"),
        "bernoulli_scalar_cpu": ("bernoulli_scalar_kernel",
            "void aten_bernoulli_scalar_cpu(float uniform[N],float probability,float out[N]){for(int i=0;i<N;++i)out[i]=(float)(uniform[i]<probability);}"),
        "cauchy_cpu": ("cauchy_kernel",
            "extern float tanf(float);void aten_cauchy_cpu(float uniform[N],float median,float sigma,float out[N]){for(int i=0;i<N;++i)out[i]=median+sigma*tanf(3.14159265358979323846f*(uniform[i]-0.5f));}"),
        "exponential_cpu": ("exponential_kernel",
            "extern float log1pf(float);void aten_exponential_cpu(float uniform[N],float lambda,float out[N]){for(int i=0;i<N;++i)out[i]=-log1pf(-uniform[i])/lambda;}"),
        "geometric_cpu": ("geometric_kernel",
            "extern float logf(float);extern float ceilf(float);void aten_geometric_cpu(float uniform[N],float probability,float out[N]){float d=logf(1.0f-probability);for(int i=0;i<N;++i)out[i]=ceilf(logf(1.0f-uniform[i])/d);}"),
        "log_normal_cpu": ("log_normal_kernel",
            "extern float expf(float);void aten_log_normal_cpu(float standard_normal[N],float mean,float std,float out[N]){for(int i=0;i<N;++i)out[i]=expf(mean+std*standard_normal[i]);}"),
        "normal_cpu": ("normal_kernel",
            "void aten_normal_cpu(float standard_normal[N],float mean,float std,float out[N]){for(int i=0;i<N;++i)out[i]=mean+std*standard_normal[i];}"),
        "uniform_cpu": ("uniform_kernel",
            "void aten_uniform_cpu(float uniform01[N],float from,float to,float out[N]){for(int i=0;i<N;++i)out[i]=from+(to-from)*uniform01[i];}"),
        "random_from_to_cpu": ("random_from_to_kernel",
            "void aten_random_from_to_cpu(unsigned bits[N],int from,int to,int out[N]){unsigned range=(unsigned)(to-from);for(int i=0;i<N;++i)out[i]=from+(int)(bits[i]%range);}"),
        "random_full_64_bits_range_cpu": ("random_full_64_bits_range_kernel",
            "void aten_random_full_64_bits_range_cpu(unsigned long bits[N],unsigned long out[N]){for(int i=0;i<N;++i)out[i]=bits[i];}"),
        "random_cpu": ("random_kernel",
            "void aten_random_cpu(unsigned bits[N],unsigned upper,int out[N]){for(int i=0;i<N;++i)out[i]=(int)(bits[i]%upper);}"),
    }
    return [(f"aten_{n}", DIST, token, h + code + "\n") for n,(token,code) in specs.items()]


def index_entries() -> list[tuple[str, str, str, str]]:
    h = "#ifndef R\n#define R 32\n#endif\n#ifndef K\n#define K 64\n#endif\n#ifndef S\n#define S 128\n#endif\n"
    specs = {
        "index_cpu": ("index_kernel", "void aten_index_cpu(float input[S][K],int index[R],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=input[index[r]][k];}"),
        "index_fill_cpu": ("index_fill_kernel", "void aten_index_fill_cpu(float out[S][K],int index[R],float value){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[index[r]][k]=value;}"),
        "index_copy_cpu": ("index_copy_kernel", "void aten_index_copy_cpu(float out[S][K],int index[R],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[index[r]][k]=source[r][k];}"),
        "index_put_cpu": ("index_put_kernel", "void aten_index_put_cpu(float out[S],int index[R],float source[R],int accumulate){for(int r=0;r<R;++r){if(accumulate)out[index[r]]+=source[r];else out[index[r]]=source[r];}}"),
        "put_cpu": ("put_kernel", "void aten_put_cpu(float out[S],int index[R],float source[R],int accumulate){for(int r=0;r<R;++r){if(accumulate)out[index[r]]+=source[r];else out[index[r]]=source[r];}}"),
        "take_cpu": ("take_kernel", "void aten_take_cpu(float input[S],int index[R],float out[R]){for(int r=0;r<R;++r)out[r]=input[index[r]];}"),
        "masked_fill_cpu": ("masked_fill_kernel", "void aten_masked_fill_cpu(float out[S],int mask[S],float value){for(int i=0;i<S;++i)if(mask[i])out[i]=value;}"),
        "masked_select_serial_cpu": ("masked_select_serial_kernel", "void aten_masked_select_serial_cpu(float input[S],int mask[S],float out[S],int count[1]){int p=0;for(int i=0;i<S;++i)if(mask[i])out[p++]=input[i];count[0]=p;}"),
        "masked_select_cpu": ("masked_select_kernel", "void aten_masked_select_cpu(float input[S],int mask[S],float out[S],int prefix[S]){int p=0;for(int i=0;i<S;++i){prefix[i]=p;if(mask[i])out[p++]=input[i];}}"),
        "masked_scatter_cpu": ("masked_scatter_kernel", "void aten_masked_scatter_cpu(float out[S],int mask[S],float source[S]){int p=0;for(int i=0;i<S;++i)if(mask[i])out[i]=source[p++];}"),
        "flip_cpu": ("flip_kernel", "void aten_flip_cpu(float input[R][K],float out[R][K],int flip_rows,int flip_cols){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int rr=flip_rows?R-1-r:r;int kk=flip_cols?K-1-k:k;out[r][k]=input[rr][kk];}}"),
    }
    rows = [(f"aten_{n}", INDEX, token, h + code + "\n") for n,(token,code) in specs.items()]
    scatter = {
        "gather_cpu": ("gather_cpu_kernel", "void aten_gather_cpu(float input[R][S],int index[R][K],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=input[r][index[r][k]];}"),
        "scatter_cpu": ("scatter_cpu_kernel", "void aten_scatter_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]=source[r][k];}"),
        "scatter_fill_cpu": ("scatter_fill_cpu_kernel", "void aten_scatter_fill_cpu(float out[R][S],int index[R][K],float value){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]=value;}"),
        "scatter_add_cpu": ("scatter_add_cpu_kernel", "void aten_scatter_add_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]+=source[r][k];}"),
        "scatter_reduce_cpu": ("scatter_reduce_cpu_kernel", "void aten_scatter_reduce_cpu(float out[R][S],int index[R][K],float source[R][K],int reduce){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];float x=source[r][k];if(reduce==0)out[r][j]+=x;else if(reduce==1)out[r][j]*=x;else if(reduce==2)out[r][j]=out[r][j]>x?out[r][j]:x;else out[r][j]=out[r][j]<x?out[r][j]:x;}}"),
        "scatter_scalar_reduce_cpu": ("scatter_scalar_reduce_cpu_kernel", "void aten_scatter_scalar_reduce_cpu(float out[R][S],int index[R][K],float value,int reduce){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];if(reduce==0)out[r][j]+=value;else if(reduce==1)out[r][j]*=value;else if(reduce==2)out[r][j]=out[r][j]>value?out[r][j]:value;else out[r][j]=out[r][j]<value?out[r][j]:value;}}"),
        "scatter_reduce_two_cpu": ("scatter_reduce_two_cpu_kernel", "void aten_scatter_reduce_two_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]+=source[r][k];}"),
        "scatter_add_expanded_index_cpu": ("scatter_add_expanded_index_kernel", "void aten_scatter_add_expanded_index_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]+=source[r][k];}"),
        "scatter_reduce_expanded_index_cpu": ("scatter_reduce_expanded_index_kernel", "void aten_scatter_reduce_expanded_index_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];out[r][j]=out[r][j]>source[r][k]?out[r][j]:source[r][k];}}"),
        "gather_expanded_index_cpu": ("gather_expanded_index_kernel", "void aten_gather_expanded_index_cpu(float input[R][S],int index[R][K],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=input[r][index[r][k]];}"),
    }
    rows += [(f"aten_{n}", SCATTER, token, h + code + "\n") for n,(token,code) in scatter.items()]
    return rows


def pool(rank: int, family: str, backward: bool) -> tuple[str, str, str, str]:
    adaptive = family.startswith("adaptive")
    is_max = family.endswith("max")
    direction = "_backward" if backward else ""
    if adaptive:
        opname = f"adaptive_{'max' if is_max else 'avg'}_pool{rank}d"
        token = (
            f"adaptive_max_pool{rank}d{direction}_kernel_impl"
            if is_max else
            f"{'adapative' if backward else 'adaptive'}_avg_pool{rank}d{direction}_kernel_impl"
        )
        source = ADAPTIVE_MAX_POOL if is_max else ADAPTIVE_AVG_POOL
    else:
        opname = f"{'max' if is_max else 'avg'}_pool{rank}d"
        token = f"{opname}{direction}_kernel_impl"
        source = MAX_POOL_1D if is_max and rank == 1 else (MAX_POOL if is_max else AVG_POOL)
        if is_max and rank == 1:
            token = "max_pool1d_impl"
    name = f"aten_{opname}{direction}_cpu"
    defs = ["#ifndef B\n#define B 1\n#endif", "#ifndef C\n#define C 2\n#endif"]
    for d in range(rank):
        defs.append(f"#ifndef I{d}\n#define I{d} {6+d}\n#endif")
        defs.append(
            f"#ifndef O{d}\n#define O{d} 3\n#endif"
            if adaptive else f"#define O{d} (I{d}/2)"
        )
    out_coords = [f"o{d}" for d in range(rank)]
    in_coords = [f"i{d}" for d in range(rank)]
    out_loops = "\n".join(
        f"{'  '*(2+d)}for(int o{d}=0;o{d}<O{d};++o{d}){{"
        for d in range(rank)
    )
    indent = "  " * (2 + rank)
    bounds = []
    for d in range(rank):
        if adaptive:
            bounds += [
                f"{indent}int s{d}=o{d}*I{d}/O{d};",
                f"{indent}int e{d}=((o{d}+1)*I{d}+O{d}-1)/O{d};",
            ]
        else:
            bounds += [f"{indent}int s{d}=o{d}*2;", f"{indent}int e{d}=s{d}+2;"]
    inner_loops = "\n".join(
        f"{indent}{'  '*d}for(int i{d}=s{d};i{d}<e{d};++i{d}){{"
        for d in range(rank)
    )
    inner_indent = indent + "  " * rank
    in_idx = flatten(["n", "c", *in_coords], ["B", "C", *[f"I{d}" for d in range(rank)]])
    out_idx = flatten(["n", "c", *out_coords], ["B", "C", *[f"O{d}" for d in range(rank)]])
    spatial_idx = flatten(in_coords, [f"I{d}" for d in range(rank)])
    inner_closes = "\n".join(
        f"{indent}{'  '*d}}}" for d in range(rank-1, -1, -1)
    )
    output_closes = "\n".join(
        f"{'  '*d}}}" for d in range(1+rank, -1, -1)
    )
    if backward:
        signature = f"float grad_output[{array_size('O',rank)}], "
        if is_max:
            signature += f"int indices[{array_size('O',rank)}], "
        signature += f"float grad_input[{array_size('I',rank)}]"
        init = f"  for(int p=0;p<{array_size('I',rank)};++p)grad_input[p]=0.0f;\n"
        if is_max:
            operation = (
                f"{indent}int flat=indices[{out_idx}];\n"
                f"{indent}grad_input[(n*C+c)*{'*'.join(f'I{d}' for d in range(rank))}+flat]+="
                f"grad_output[{out_idx}];"
            )
            inner = ""
        else:
            count = "*".join(f"(e{d}-s{d})" for d in range(rank))
            operation = (
                f"{inner_indent}grad_input[{in_idx}]+="
                f"grad_output[{out_idx}]/(float)({count});"
            )
            inner = inner_loops + "\n" + operation + "\n" + inner_closes
            operation = ""
    else:
        signature = f"float input[{array_size('I',rank)}], float output[{array_size('O',rank)}]"
        if is_max:
            signature += f", int indices[{array_size('O',rank)}]"
            pre = f"{indent}float value=-3.402823466e38f;int best=0;"
            update = (
                f"{inner_indent}if(input[{in_idx}]>value){{"
                f"value=input[{in_idx}];best={spatial_idx};}}"
            )
            post = f"{indent}output[{out_idx}]=value;indices[{out_idx}]=best;"
        else:
            pre = f"{indent}float value=0.0f;int count=0;"
            update = f"{inner_indent}value+=input[{in_idx}];++count;"
            post = f"{indent}output[{out_idx}]=value/(float)count;"
        init = ""
        inner = pre + "\n" + inner_loops + "\n" + update + "\n" + inner_closes
        operation = post
    code = (
        f"/* Fixed-shape ATen {opname}{direction}. */\n" + "\n".join(defs) + "\n"
        f"void {name}({signature}){{\n{init}"
        "  for(int n=0;n<B;++n){\n    for(int c=0;c<C;++c){\n"
        + out_loops + "\n" + "\n".join(bounds) + "\n"
        + inner + ("\n" if inner else "") + operation + "\n"
        + output_closes + "\n}\n"
    )
    return name, source, token, code


def pooling_entries() -> list[tuple[str, str, str, str]]:
    entries = []
    for rank in (2, 3):
        for family in ("adaptive_avg", "adaptive_max", "avg"):
            for backward in (False, True):
                entries.append(pool(rank, family, backward))
    for rank in (1, 3):
        for backward in (False, True):
            # max_pool1d has no separate registered backward kernel.
            if rank == 1 and backward:
                continue
            entries.append(pool(rank, "max", backward))
    return entries


def misc_entries() -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    def add(name: str, source: str, token: str, code: str) -> None:
        specs.append((f"aten_{name}", source, token, code + "\n"))

    distance = "aten/src/ATen/native/cpu/DistanceOpsKernel.cpp"
    dh = "#ifndef N\n#define N 16\n#endif\n#ifndef M\n#define M 12\n#endif\n#ifndef D\n#define D 32\n#endif\n"
    add("pdist_forward_cpu", distance, "pdist_forward_kernel_impl", dh + """extern float sqrtf(float);
void aten_pdist_forward_cpu(float x[N][D],float out[N*(N-1)/2]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);out[p]=0;for(int d=0;d<D;++d){float z=x[i][d]-x[j][d];out[p]+=z*z;}}for(int p=0;p<N*(N-1)/2;++p)out[p]=sqrtf(out[p]);}""")
    add("pdist_backward_cpu", distance, "pdist_backward_kernel_impl", dh + """extern float sqrtf(float);
void aten_pdist_backward_cpu(float x[N][D],float grad[N*(N-1)/2],float out[N][D]){float dist[N*(N-1)/2];for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);dist[p]=0;for(int d=0;d<D;++d){float z=x[i][d]-x[j][d];dist[p]+=z*z;}dist[p]=sqrtf(dist[p]);}for(int i=0;i<N;++i)for(int d=0;d<D;++d)out[i][d]=0;for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);float g=dist[p]==0?0:grad[p]/dist[p];for(int d=0;d<D;++d){float v=g*(x[i][d]-x[j][d]);out[i][d]+=v;out[j][d]-=v;}}}""")
    add("cdist_cpu", distance, "cdist_kernel_impl", dh + """extern float sqrtf(float);
void aten_cdist_cpu(float x[N][D],float y[M][D],float out[N][M]){for(int i=0;i<N;++i)for(int j=0;j<M;++j){float s=0;for(int d=0;d<D;++d){float z=x[i][d]-y[j][d];s+=z*z;}out[i][j]=sqrtf(s);}}""")
    add("cdist_backward_cpu", distance, "cdist_backward_kernel_impl", dh + """extern float sqrtf(float);
void aten_cdist_backward_cpu(float x[N][D],float y[M][D],float grad[N][M],float out[N][D]){for(int i=0;i<N;++i)for(int d=0;d<D;++d)out[i][d]=0;for(int i=0;i<N;++i)for(int j=0;j<M;++j){float s=0;for(int d=0;d<D;++d){float z=x[i][d]-y[j][d];s+=z*z;}float n=sqrtf(s),g=n==0?0:grad[i][j]/n;for(int d=0;d<D;++d)out[i][d]+=g*(x[i][d]-y[j][d]);}}""")

    hist = "aten/src/ATen/native/cpu/HistogramKernel.cpp"
    hh = "#ifndef N\n#define N 4096\n#endif\n#ifndef B0\n#define B0 16\n#endif\n#ifndef B1\n#define B1 12\n#endif\n"
    add("histogramdd_cpu", hist, "histogramdd_kernel_impl", hh + """void aten_histogramdd_cpu(float x[N][2],float weight[N],float lo0,float hi0,float lo1,float hi1,float out[B0][B1]){for(int a=0;a<B0;++a)for(int b=0;b<B1;++b)out[a][b]=0;for(int i=0;i<N;++i){int a=(int)((x[i][0]-lo0)*B0/(hi0-lo0));int b=(int)((x[i][1]-lo1)*B1/(hi1-lo1));if(a>=0&&a<B0&&b>=0&&b<B1)out[a][b]+=weight[i];}}""")
    add("histogramdd_linear_cpu", hist, "histogramdd_linear_kernel_impl", hh + """void aten_histogramdd_linear_cpu(int bin[N],float weight[N],float out[B0*B1]){for(int b=0;b<B0*B1;++b)out[b]=0;for(int i=0;i<N;++i)if(bin[i]>=0&&bin[i]<B0*B1)out[bin[i]]+=weight[i];}""")
    add("histogram_select_outer_bin_edges_cpu", hist, "histogram_select_outer_bin_edges_impl", hh + """void aten_histogram_select_outer_bin_edges_cpu(float x[N],float out_min[1],float out_max[1]){float lo=x[0],hi=x[0];for(int i=1;i<N;++i){lo=x[i]<lo?x[i]:lo;hi=x[i]>hi?x[i]:hi;}out_min[0]=lo;out_max[0]=hi;}""")

    sortsrc = "aten/src/ATen/native/cpu/SortingKernel.cpp"
    sh = "#ifndef R\n#define R 16\n#endif\n#ifndef K\n#define K 64\n#endif\n#ifndef TOP\n#define TOP 8\n#endif\n"
    sort_body = """for(int r=0;r<R;++r){for(int k=0;k<K;++k){values[r][k]=input[r][k];indices[r][k]=k;}for(int k=1;k<K;++k){float v=values[r][k];int idx=indices[r][k],j=k-1;while(j>=0&&values[r][j]<v){values[r][j+1]=values[r][j];indices[r][j+1]=indices[r][j];--j;}values[r][j+1]=v;indices[r][j+1]=idx;}}"""
    add("sort_cpu", sortsrc, "sort_kernel", sh + "void aten_sort_cpu(float input[R][K],float values[R][K],int indices[R][K]){" + sort_body + "}")
    add("topk_cpu", sortsrc, "topk_kernel", sh + "void aten_topk_cpu(float input[R][K],float out[R][TOP],int out_idx[R][TOP]){float values[R][K];int indices[R][K];" + sort_body + "for(int r=0;r<R;++r)for(int k=0;k<TOP;++k){out[r][k]=values[r][k];out_idx[r][k]=indices[r][k];}}")

    sumsrc = "aten/src/ATen/native/cpu/SumKernel.cpp"
    add("sum_cpu_backend", sumsrc, "sum_kernel_impl", sh + "void aten_sum_cpu_backend(float x[R][K],float out[R]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)s+=x[r][k];out[r]=s;}}")
    add("nansum_cpu", sumsrc, "nansum_kernel_impl", sh + "void aten_nansum_cpu(float x[R][K],float out[R]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)if(x[r][k]==x[r][k])s+=x[r][k];out[r]=s;}}")

    add("cat_serial_cpu", "aten/src/ATen/native/cpu/CatKernel.cpp", "cat_serial_kernel", sh + "#define M 12\nvoid aten_cat_serial_cpu(float a[R][K],float b[M][K],float out[R+M][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=a[r][k];for(int r=0;r<M;++r)for(int k=0;k<K;++k)out[R+r][k]=b[r][k];}")
    add("channel_shuffle_cpu", "aten/src/ATen/native/cpu/ChannelShuffleKernel.cpp", "channel_shuffle_kernel_impl", "#define B 2\n#define G 4\n#define CPG 3\n#define S 32\nvoid aten_channel_shuffle_cpu(float x[B][G*CPG][S],float out[B][G*CPG][S]){for(int n=0;n<B;++n)for(int g=0;g<G;++g)for(int c=0;c<CPG;++c)for(int s=0;s<S;++s)out[n][c*G+g][s]=x[n][g*CPG+c][s];}")
    add("cross_cpu_backend", "aten/src/ATen/native/cpu/CrossKernel.cpp", "cross_kernel_impl", "#define V 256\nvoid aten_cross_cpu_backend(float a[V][3],float b[V][3],float out[V][3]){for(int i=0;i<V;++i){out[i][0]=a[i][1]*b[i][2]-a[i][2]*b[i][1];out[i][1]=a[i][2]*b[i][0]-a[i][0]*b[i][2];out[i][2]=a[i][0]*b[i][1]-a[i][1]*b[i][0];}}")

    pix = "aten/src/ATen/native/cpu/PixelShuffleKernel.cpp"
    ph = "#define B 1\n#define C 3\n#define H 8\n#define W 8\n#define RATIO 2\n"
    add("pixel_shuffle_cpu_backend", pix, "pixel_shuffle_kernel_impl", ph + "void aten_pixel_shuffle_cpu_backend(float x[B][C*RATIO*RATIO][H][W],float out[B][C][H*RATIO][W*RATIO]){for(int n=0;n<B;++n)for(int c=0;c<C;++c)for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int ry=0;ry<RATIO;++ry)for(int rx=0;rx<RATIO;++rx)out[n][c][h*RATIO+ry][w*RATIO+rx]=x[n][c*RATIO*RATIO+ry*RATIO+rx][h][w];}")
    add("pixel_unshuffle_cpu_backend", pix, "pixel_unshuffle_kernel_impl", ph + "void aten_pixel_unshuffle_cpu_backend(float x[B][C][H*RATIO][W*RATIO],float out[B][C*RATIO*RATIO][H][W]){for(int n=0;n<B;++n)for(int c=0;c<C;++c)for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int ry=0;ry<RATIO;++ry)for(int rx=0;rx<RATIO;++rx)out[n][c*RATIO*RATIO+ry*RATIO+rx][h][w]=x[n][c][h*RATIO+ry][w*RATIO+rx];}")

    add("stack_serial_cpu", "aten/src/ATen/native/cpu/StackKernel.cpp", "stack_serial_kernel", "#define T 4\n#define R 16\n#define K 32\nvoid aten_stack_serial_cpu(float x[T][R][K],float out[R][T][K]){for(int t=0;t<T;++t)for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][t][k]=x[t][r][k];}")
    unfsrc = "aten/src/ATen/native/cpu/Unfold2d.cpp"
    uh = "#define C 2\n#define H 8\n#define W 8\n#define KH 3\n#define KW 3\n#define OH 6\n#define OW 6\n"
    add("unfolded2d_copy_cpu", unfsrc, "unfolded2d_copy_kernel", uh + "void aten_unfolded2d_copy_cpu(float x[C][H][W],float out[C][KH][KW][OH][OW]){for(int c=0;c<C;++c)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox)out[c][ky][kx][oy][ox]=x[c][oy+ky][ox+kx];}")
    add("unfolded2d_acc_cpu", unfsrc, "unfolded2d_acc_kernel", uh + "void aten_unfolded2d_acc_cpu(float col[C][KH][KW][OH][OW],float out[C][H][W]){for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int x=0;x<W;++x)out[c][y][x]=0;for(int c=0;c<C;++c)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox)out[c][oy+ky][ox+kx]+=col[c][ky][kx][oy][ox];}")
    add("unfold_backward_cpu", "aten/src/ATen/native/cpu/UnfoldBackwardKernel.cpp", "unfold_backward_cpu_kernel", "#define N 32\n#define SIZE 64\n#define STEP 2\nvoid aten_unfold_backward_cpu(float grad[N][SIZE],float out[N*STEP+SIZE]){for(int i=0;i<N*STEP+SIZE;++i)out[i]=0;for(int w=0;w<N;++w)for(int k=0;k<SIZE;++k)out[w*STEP+k]+=grad[w][k];}")
    unpool = "aten/src/ATen/native/cpu/MaxUnpoolKernel.cpp"
    add("max_unpool2d_cpu", unpool, "max_unpool2d_kernel_impl", "#define C 2\n#define N 64\n#define O 256\nvoid aten_max_unpool2d_cpu(float x[C][N],int index[C][N],float out[C][O]){for(int p=0;p<C*O;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int i=0;i<N;++i)out[c][index[c][i]]=x[c][i];}")
    add("max_unpool3d_cpu", unpool, "max_unpool3d_kernel_impl", "#define C 2\n#define N 64\n#define O 512\nvoid aten_max_unpool3d_cpu(float x[C][N],int index[C][N],float out[C][O]){for(int p=0;p<C*O;++p)((float*)out)[p]=0;for(int c=0;c<C;++c)for(int i=0;i<N;++i)out[c][index[c][i]]=x[c][i];}")
    return specs


def specialized_entries() -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    def add(name: str, source: str, token: str, code: str) -> None:
        specs.append((f"aten_{name}", source, token, code + "\n"))

    spmm = "aten/src/ATen/native/cpu/SpmmReduceKernel.cpp"
    sp = "#define ROWS 16\n#define INNER 32\n#define COLS 24\n#define NNZ 96\n"
    add("spmm_reduce_cpu", spmm, "spmm_reduce_kernel", sp + """void aten_spmm_reduce_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float other[INNER][COLS],int reduce,float out[ROWS][COLS]){for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){float acc=reduce==2?-3.402823466e38f:(reduce==3?3.402823466e38f:0.0f);for(int p=crow[r];p<crow[r+1];++p){float x=val[p]*other[col[p]][n];if(reduce==0||reduce==1)acc+=x;else if(reduce==2)acc=acc>x?acc:x;else acc=acc<x?acc:x;}if(reduce==1&&crow[r+1]>crow[r])acc/=(float)(crow[r+1]-crow[r]);out[r][n]=acc;}}""")
    add("spmm_reduce_arg_cpu", spmm, "spmm_reduce_arg_kernel", sp + """void aten_spmm_reduce_arg_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float other[INNER][COLS],int choose_max,float out[ROWS][COLS],int arg[ROWS][COLS]){for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){float acc=choose_max?-3.402823466e38f:3.402823466e38f;int best=-1;for(int p=crow[r];p<crow[r+1];++p){float x=val[p]*other[col[p]][n];if((choose_max&&x>acc)||(!choose_max&&x<acc)){acc=x;best=p;}}out[r][n]=acc;arg[r][n]=best;}}""")
    add("spmm_reduce_backward_input_cpu", spmm, "spmm_reduce_backward_input_kernel", sp + """void aten_spmm_reduce_backward_input_cpu(int crow[ROWS+1],int col[NNZ],float grad[ROWS][COLS],float other[INNER][COLS],float out[NNZ]){for(int r=0;r<ROWS;++r)for(int p=crow[r];p<crow[r+1];++p){float s=0;for(int n=0;n<COLS;++n)s+=grad[r][n]*other[col[p]][n];out[p]=s;}}""")
    add("spmm_reduce_backward_input_arg_cpu", spmm, "spmm_reduce_backward_input_arg_kernel", sp + """void aten_spmm_reduce_backward_input_arg_cpu(int crow[ROWS+1],int col[NNZ],int arg[ROWS][COLS],float grad[ROWS][COLS],float other[INNER][COLS],float out[NNZ]){for(int p=0;p<NNZ;++p)out[p]=0;for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){int p=arg[r][n];if(p>=0)out[p]+=grad[r][n]*other[col[p]][n];}}""")
    add("spmm_reduce_backward_other_cpu", spmm, "spmm_reduce_backward_other_kernel", sp + """void aten_spmm_reduce_backward_other_cpu(int crow[ROWS+1],int col[NNZ],float val[NNZ],float grad[ROWS][COLS],float out[INNER][COLS]){for(int k=0;k<INNER;++k)for(int n=0;n<COLS;++n)out[k][n]=0;for(int r=0;r<ROWS;++r)for(int p=crow[r];p<crow[r+1];++p)for(int n=0;n<COLS;++n)out[col[p]][n]+=val[p]*grad[r][n];}""")
    add("spmm_reduce_backward_other_arg_cpu", spmm, "spmm_reduce_backward_other_arg_kernel", sp + """void aten_spmm_reduce_backward_other_arg_cpu(int col[NNZ],float val[NNZ],int arg[ROWS][COLS],float grad[ROWS][COLS],float out[INNER][COLS]){for(int k=0;k<INNER;++k)for(int n=0;n<COLS;++n)out[k][n]=0;for(int r=0;r<ROWS;++r)for(int n=0;n<COLS;++n){int p=arg[r][n];if(p>=0)out[col[p]][n]+=val[p]*grad[r][n];}}""")

    gemv = "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp"
    gh = "#define M 64\n#define K 128\n"
    add("fp16_gemv_trans_cpu", gemv, "fp16_gemv_trans", gh + "void aten_fp16_gemv_trans_cpu(float matrix[M][K],float vector[M],float out[K]){for(int k=0;k<K;++k){float s=0;for(int m=0;m<M;++m)s+=matrix[m][k]*vector[m];out[k]=s;}}")
    add("bf16_gemv_trans_cpu", gemv, "bf16_gemv_trans", gh + "void aten_bf16_gemv_trans_cpu(float matrix[M][K],float vector[M],float out[K]){for(int k=0;k<K;++k){float s=0;for(int m=0;m<M;++m)s+=matrix[m][k]*vector[m];out[k]=s;}}")
    add("fp16_dot_cpu", gemv, "fp16_dot", gh + "void aten_fp16_dot_cpu(float a[K],float b[K],float out[1]){float s=0;for(int k=0;k<K;++k)s+=a[k]*b[k];out[0]=s;}")
    add("bf16_dot_cpu", gemv, "bf16_dot", gh + "void aten_bf16_dot_cpu(float a[K],float b[K],float out[1]){float s=0;for(int k=0;k<K;++k)s+=a[k]*b[k];out[0]=s;}")

    int4 = "aten/src/ATen/native/cpu/int4mm_kernel.cpp"
    ih = "#define M 32\n#define K 64\n#define N 48\n"
    add("weight_to_int4pack_cpu", int4, "weight_to_int4pack_kernel", ih + "void aten_weight_to_int4pack_cpu(unsigned char weight[N][K],unsigned char packed[N][K/2]){for(int n=0;n<N;++n)for(int k=0;k<K;k+=2)packed[n][k/2]=(weight[n][k]&15)|((weight[n][k+1]&15)<<4);}")
    add("int4pack_mm_cpu", int4, "int4pack_mm_kernel", ih + "void aten_int4pack_mm_cpu(float a[M][K],unsigned char packed[N][K/2],float scale[N],float zero[N],float out[M][N]){for(int m=0;m<M;++m)for(int n=0;n<N;++n){float s=0;for(int k=0;k<K;++k){int q=(packed[n][k/2]>>(4*(k&1)))&15;s+=a[m][k]*((float)q-zero[n])*scale[n];}out[m][n]=s;}}")
    add("dyn_quant_pack_4bit_weight_cpu", int4, "dyn_quant_pack_4bit_weight_kernel", ih + "void aten_dyn_quant_pack_4bit_weight_cpu(float weight[N][K],unsigned char packed[N][K/2],float scale[N],float zero[N]){for(int n=0;n<N;++n){float lo=weight[n][0],hi=lo;for(int k=1;k<K;++k){lo=weight[n][k]<lo?weight[n][k]:lo;hi=weight[n][k]>hi?weight[n][k]:hi;}scale[n]=(hi-lo)/15.0f;zero[n]=-lo/scale[n];for(int k=0;k<K;k+=2){int q0=(int)(weight[n][k]/scale[n]+zero[n]+0.5f);int q1=(int)(weight[n][k+1]/scale[n]+zero[n]+0.5f);if(q0<0)q0=0;if(q0>15)q0=15;if(q1<0)q1=0;if(q1>15)q1=15;packed[n][k/2]=(unsigned char)(q0|(q1<<4));}}}")
    add("dyn_quant_matmul_4bit_cpu", int4, "dyn_quant_matmul_4bit_kernel", ih + "void aten_dyn_quant_matmul_4bit_cpu(float a[M][K],unsigned char packed[N][K/2],float scale[N],float zero[N],float out[M][N]){for(int m=0;m<M;++m)for(int n=0;n<N;++n){float s=0;for(int k=0;k<K;++k){int q=(packed[n][k/2]>>(4*(k&1)))&15;s+=a[m][k]*((float)q-zero[n])*scale[n];}out[m][n]=s;}}")
    add("int8pack_mm_cpu", "aten/src/ATen/native/cpu/int8mm_kernel.cpp", "int8pack_mm_kernel", ih + "void aten_int8pack_mm_cpu(float a[M][K],signed char weight[N][K],float scale[N],float out[M][N]){for(int m=0;m<M;++m)for(int n=0;n<N;++n){out[m][n]=0;for(int k=0;k<K;++k)out[m][n]+=a[m][k]*(float)weight[n][k]*scale[n];}}")

    add("depthwise_conv3x3_cpu", "aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp", "_convolution_depthwise3x3_winograd", "#define B 1\n#define C 8\n#define H 16\n#define W 16\nvoid aten_depthwise_conv3x3_cpu(float x[B][C][H][W],float weight[C][3][3],float bias[C],float out[B][C][H][W]){for(int n=0;n<B;++n)for(int c=0;c<C;++c)for(int y=0;y<H;++y)for(int z=0;z<W;++z){float s=bias[c];for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int iy=y+ky-1,ix=z+kx-1;if(iy>=0&&iy<H&&ix>=0&&ix<W)s+=x[n][c][iy][ix]*weight[c][ky][kx];}out[n][c][y][z]=s;}}")

    flash = "aten/src/ATen/native/cpu/FlashAttentionKernel.cpp"
    fh = "#define B 1\n#define H 2\n#define Q 16\n#define K 16\n#define D 32\n"
    add("flash_attention_cpu", flash, "flash_attention_kernel_impl", fh + """extern float expf(float);extern float logf(float);extern float sqrtf(float);
void aten_flash_attention_cpu(float q[B][H][Q][D],float k[B][H][K][D],float v[B][H][K][D],float out[B][H][Q][D],float lse[B][H][Q]){float scale=1.0f/sqrtf((float)D);for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int i=0;i<Q;++i){float score[K],m=-3.402823466e38f;for(int j=0;j<K;++j){float s=0;for(int d=0;d<D;++d)s+=q[b][h][i][d]*k[b][h][j][d];score[j]=s*scale;m=score[j]>m?score[j]:m;}float z=0;for(int j=0;j<K;++j){score[j]=expf(score[j]-m);z+=score[j];}lse[b][h][i]=m+logf(z);for(int d=0;d<D;++d){float s=0;for(int j=0;j<K;++j)s+=score[j]/z*v[b][h][j][d];out[b][h][i][d]=s;}}}""")
    add("flash_attention_backward_cpu", flash, "flash_attention_backward_kernel_impl", fh + """extern float expf(float);extern float sqrtf(float);
void aten_flash_attention_backward_cpu(float q[B][H][Q][D],float k[B][H][K][D],float v[B][H][K][D],float grad[B][H][Q][D],float dq[B][H][Q][D],float dk[B][H][K][D],float dv[B][H][K][D]){for(int p=0;p<B*H*Q*D;++p)((float*)dq)[p]=0;for(int p=0;p<B*H*K*D;++p){((float*)dk)[p]=0;((float*)dv)[p]=0;}float scale=1.0f/sqrtf((float)D);for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int i=0;i<Q;++i){float p[K],dp[K],m=-3.402823466e38f,z=0,sump=0;for(int j=0;j<K;++j){float s=0;for(int d=0;d<D;++d)s+=q[b][h][i][d]*k[b][h][j][d];p[j]=s*scale;m=p[j]>m?p[j]:m;}for(int j=0;j<K;++j){p[j]=expf(p[j]-m);z+=p[j];}for(int j=0;j<K;++j){p[j]/=z;float s=0;for(int d=0;d<D;++d){s+=grad[b][h][i][d]*v[b][h][j][d];dv[b][h][j][d]+=p[j]*grad[b][h][i][d];}dp[j]=s;sump+=s*p[j];}for(int j=0;j<K;++j){float ds=p[j]*(dp[j]-sump)*scale;for(int d=0;d<D;++d){dq[b][h][i][d]+=ds*k[b][h][j][d];dk[b][h][j][d]+=ds*q[b][h][i][d];}}}}""")

    grid = "aten/src/ATen/native/cpu/GridSamplerKernel.cpp"
    grh = "#define B 1\n#define C 3\n#define IH 8\n#define IW 8\n#define OH 6\n#define OW 6\n"
    add("grid_sampler_2d_cpu", grid, "grid_sampler_2d_cpu_kernel_impl", grh + """void aten_grid_sampler_2d_cpu(float x[B][C][IH][IW],float grid[B][OH][OW][2],float out[B][C][OH][OW]){for(int n=0;n<B;++n)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z){float fx=(grid[n][y][z][0]+1)*0.5f*(IW-1),fy=(grid[n][y][z][1]+1)*0.5f*(IH-1);int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1;float wx=fx-x0,wy=fy-y0;for(int c=0;c<C;++c){float v=0;if(x0>=0&&x0<IW&&y0>=0&&y0<IH)v+=(1-wx)*(1-wy)*x[n][c][y0][x0];if(x1>=0&&x1<IW&&y0>=0&&y0<IH)v+=wx*(1-wy)*x[n][c][y0][x1];if(x0>=0&&x0<IW&&y1>=0&&y1<IH)v+=(1-wx)*wy*x[n][c][y1][x0];if(x1>=0&&x1<IW&&y1>=0&&y1<IH)v+=wx*wy*x[n][c][y1][x1];out[n][c][y][z]=v;}}}""")
    add("grid_sampler_2d_backward_cpu", grid, "grid_sampler_2d_backward_cpu_kernel_impl", grh + """void aten_grid_sampler_2d_backward_cpu(float x[B][C][IH][IW],float grid[B][OH][OW][2],float grad[B][C][OH][OW],float dx[B][C][IH][IW],float dgrid[B][OH][OW][2]){for(int p=0;p<B*C*IH*IW;++p)((float*)dx)[p]=0;for(int n=0;n<B;++n)for(int y=0;y<OH;++y)for(int z=0;z<OW;++z){float fx=(grid[n][y][z][0]+1)*0.5f*(IW-1),fy=(grid[n][y][z][1]+1)*0.5f*(IH-1);int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1;float wx=fx-x0,wy=fy-y0,gx=0,gy=0;for(int c=0;c<C;++c){float g=grad[n][c][y][z],v00=0,v01=0,v10=0,v11=0;if(x0>=0&&x0<IW&&y0>=0&&y0<IH){v00=x[n][c][y0][x0];dx[n][c][y0][x0]+=g*(1-wx)*(1-wy);}if(x1>=0&&x1<IW&&y0>=0&&y0<IH){v01=x[n][c][y0][x1];dx[n][c][y0][x1]+=g*wx*(1-wy);}if(x0>=0&&x0<IW&&y1>=0&&y1<IH){v10=x[n][c][y1][x0];dx[n][c][y1][x0]+=g*(1-wx)*wy;}if(x1>=0&&x1<IW&&y1>=0&&y1<IH){v11=x[n][c][y1][x1];dx[n][c][y1][x1]+=g*wx*wy;}gx+=g*((v01-v00)*(1-wy)+(v11-v10)*wy);gy+=g*((v10-v00)*(1-wx)+(v11-v01)*wx);}dgrid[n][y][z][0]=gx*0.5f*(IW-1);dgrid[n][y][z][1]=gy*0.5f*(IH-1);}}""")

    add("multinomial_with_replacement_cpu", "aten/src/ATen/native/cpu/MultinomialKernel.cpp", "multinomial_with_replacement_kernel_impl", "#define B 8\n#define C 32\n#define S 16\nvoid aten_multinomial_with_replacement_cpu(float probability[B][C],float uniform[B][S],int out[B][S]){for(int b=0;b<B;++b){float total=0,cdf[C];for(int c=0;c<C;++c){total+=probability[b][c];cdf[c]=total;}for(int s=0;s<S;++s){float u=uniform[b][s]*total;int c=0;while(c<C-1&&cdf[c]<u)++c;out[b][s]=c;}}}")
    add("transform_bias_rescale_qkv_cpu", "aten/src/ATen/native/cpu/NativeMultiheadAttnKernel.cpp", "transform_bias_rescale_qkv_kernel_impl", "#define B 2\n#define S 16\n#define H 4\n#define D 8\nvoid aten_transform_bias_rescale_qkv_cpu(float qkv[B][S][3][H][D],float bias[3][H][D],float scale,float q[B][H][S][D],float k[B][H][S][D],float v[B][H][S][D]){for(int b=0;b<B;++b)for(int s=0;s<S;++s)for(int h=0;h<H;++h)for(int d=0;d<D;++d){q[b][h][s][d]=(qkv[b][s][0][h][d]+bias[0][h][d])*scale;k[b][h][s][d]=qkv[b][s][1][h][d]+bias[1][h][d];v[b][h][s][d]=qkv[b][s][2][h][d]+bias[2][h][d];}}")
    add("sampled_addmm_sparse_csr_cpu", "aten/src/ATen/native/cpu/SampledAddmmKernel.cpp", "sampled_addmm_sparse_csr_kernel", "#define R 16\n#define K 32\n#define C 24\n#define NNZ 96\nvoid aten_sampled_addmm_sparse_csr_cpu(int crow[R+1],int col[NNZ],float self[NNZ],float a[R][K],float b[K][C],float alpha,float beta,float out[NNZ]){for(int r=0;r<R;++r)for(int p=crow[r];p<crow[r+1];++p){float s=0;for(int k=0;k<K;++k)s+=a[r][k]*b[k][col[p]];out[p]=beta*self[p]+alpha*s;}}")
    add("spdiags_cpu", "aten/src/ATen/native/cpu/SparseFactories.cpp", "_spdiags_kernel_cpu", "#define D 5\n#define N 16\nvoid aten_spdiags_cpu(float diagonals[D][N],int offsets[D],float out[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[i][j]=0;for(int d=0;d<D;++d)for(int i=0;i<N;++i){int j=i+offsets[d];if(j>=0&&j<N)out[i][j]=diagonals[d][offsets[d]>=0?i:i-offsets[d]];}}")

    norm = "#define B 4\n#define C 8\n#define S 32\n"
    add("weight_norm_cpu", "aten/src/ATen/native/cpu/WeightNormKernel.cpp", "weight_norm_kernel", norm + """extern float sqrtf(float);void aten_weight_norm_cpu(float v[C][S],float g[C],float out[C][S],float norms[C]){for(int c=0;c<C;++c){float s=0;for(int i=0;i<S;++i)s+=v[c][i]*v[c][i];norms[c]=sqrtf(s);for(int i=0;i<S;++i)out[c][i]=g[c]*v[c][i]/norms[c];}}""")
    add("weight_norm_backward_cpu", "aten/src/ATen/native/cpu/WeightNormKernel.cpp", "weight_norm_backward_kernel", norm + """void aten_weight_norm_backward_cpu(float grad[C][S],float v[C][S],float g[C],float norms[C],float dv[C][S],float dg[C]){for(int c=0;c<C;++c){float dot=0;for(int i=0;i<S;++i)dot+=grad[c][i]*v[c][i];dg[c]=dot/norms[c];for(int i=0;i<S;++i)dv[c][i]=g[c]/norms[c]*(grad[c][i]-v[c][i]*dot/(norms[c]*norms[c]));}}""")

    bn = "aten/src/ATen/native/cpu/batch_norm_kernel.cpp"
    add("batch_norm_collect_stats_cpu", bn, "batch_norm_cpu_collect_stats_kernel", norm + """extern float sqrtf(float);void aten_batch_norm_collect_stats_cpu(float x[B][C][S],float mean[C],float var[C]){for(int c=0;c<C;++c){float s=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i)s+=x[b][c][i];mean[c]=s/(B*S);float q=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i){float d=x[b][c][i]-mean[c];q+=d*d;}var[c]=q/(B*S);}}""")
    add("batch_norm_backward_cpu", bn, "batch_norm_cpu_backward_kernel", norm + """void aten_batch_norm_backward_cpu(float grad[B][C][S],float x[B][C][S],float mean[C],float invstd[C],float weight[C],float dx[B][C][S],float dweight[C],float dbias[C]){for(int c=0;c<C;++c){float sg=0,sgx=0;for(int b=0;b<B;++b)for(int i=0;i<S;++i){sg+=grad[b][c][i];sgx+=grad[b][c][i]*(x[b][c][i]-mean[c]);}dbias[c]=sg;dweight[c]=sgx*invstd[c];for(int b=0;b<B;++b)for(int i=0;i<S;++i)dx[b][c][i]=weight[c]*invstd[c]/(B*S)*((B*S)*grad[b][c][i]-sg-(x[b][c][i]-mean[c])*invstd[c]*invstd[c]*sgx);}}""")

    group = "aten/src/ATen/native/cpu/group_norm_kernel.cpp"
    layer = "aten/src/ATen/native/cpu/layer_norm_kernel.cpp"
    gh = "#define B 4\n#define G 4\n#define CPG 2\n#define S 16\n"
    add("group_norm_cpu", group, "GroupNormKernelImpl", gh + """extern float sqrtf(float);void aten_group_norm_cpu(float x[B][G][CPG][S],float weight[G][CPG],float bias[G][CPG],float eps,float out[B][G][CPG][S],float mean[B][G],float rstd[B][G]){for(int b=0;b<B;++b)for(int g=0;g<G;++g){float s=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i)s+=x[b][g][c][i];mean[b][g]=s/(CPG*S);float q=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float d=x[b][g][c][i]-mean[b][g];q+=d*d;}rstd[b][g]=1.0f/sqrtf(q/(CPG*S)+eps);for(int c=0;c<CPG;++c)for(int i=0;i<S;++i)out[b][g][c][i]=(x[b][g][c][i]-mean[b][g])*rstd[b][g]*weight[g][c]+bias[g][c];}}""")
    add("group_norm_backward_cpu", group, "GroupNormBackwardKernelImpl", gh + """void aten_group_norm_backward_cpu(float grad[B][G][CPG][S],float x[B][G][CPG][S],float mean[B][G],float rstd[B][G],float weight[G][CPG],float dx[B][G][CPG][S]){for(int b=0;b<B;++b)for(int g=0;g<G;++g){float s1=0,s2=0;for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float w=grad[b][g][c][i]*weight[g][c];s1+=w;s2+=w*(x[b][g][c][i]-mean[b][g]);}for(int c=0;c<CPG;++c)for(int i=0;i<S;++i){float w=grad[b][g][c][i]*weight[g][c];dx[b][g][c][i]=rstd[b][g]/(CPG*S)*((CPG*S)*w-s1-(x[b][g][c][i]-mean[b][g])*rstd[b][g]*rstd[b][g]*s2);}}}""")
    lh = "#define B 16\n#define D 64\n"
    add("layer_norm_cpu_backend", layer, "LayerNormKernelImpl", lh + """extern float sqrtf(float);void aten_layer_norm_cpu_backend(float x[B][D],float weight[D],float bias[D],float eps,float out[B][D],float mean[B],float rstd[B]){for(int b=0;b<B;++b){float s=0;for(int d=0;d<D;++d)s+=x[b][d];mean[b]=s/D;float q=0;for(int d=0;d<D;++d){float z=x[b][d]-mean[b];q+=z*z;}rstd[b]=1.0f/sqrtf(q/D+eps);for(int d=0;d<D;++d)out[b][d]=(x[b][d]-mean[b])*rstd[b]*weight[d]+bias[d];}}""")
    add("layer_norm_backward_cpu", layer, "LayerNormBackwardKernelImpl", lh + """void aten_layer_norm_backward_cpu(float grad[B][D],float x[B][D],float mean[B],float rstd[B],float weight[D],float dx[B][D],float dw[D],float db[D]){for(int d=0;d<D;++d){dw[d]=0;db[d]=0;}for(int b=0;b<B;++b){float s1=0,s2=0;for(int d=0;d<D;++d){float w=grad[b][d]*weight[d];s1+=w;s2+=w*(x[b][d]-mean[b]);dw[d]+=grad[b][d]*(x[b][d]-mean[b])*rstd[b];db[d]+=grad[b][d];}for(int d=0;d<D;++d){float w=grad[b][d]*weight[d];dx[b][d]=rstd[b]/D*(D*w-s1-(x[b][d]-mean[b])*rstd[b]*rstd[b]*s2);}}}""")
    return specs


def main() -> None:
    rows = []
    entries = []
    for rank in (1, 2, 3):
        for exact in (False, True):
            for backward in (False, True):
                entries.append(nearest(rank, exact, backward))
        for backward in (False, True):
            entries.append(linear(rank, backward))
    entries.append(bicubic_2d())
    for kind in ("bilinear", "bicubic", "lanczos"):
        for backward in (False, True):
            entries.append(filtered_2d(kind, backward))
    for rank in (1, 2, 3):
        for mode in ("reflection", "replication"):
            for backward in (False, True):
                entries.append(padding(rank, mode, backward))
    entries.extend(reduction_entries())
    entries.extend(distribution_entries())
    entries.extend(index_entries())
    entries.extend(pooling_entries())
    entries.extend(misc_entries())
    entries.extend(specialized_entries())
    for name, source, token, code in entries:
        (OUT / f"{name}.c").write_text(code)
        rows.append({"kernel": name, "source": source, "token": token})
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kernel", "source", "token"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"generated {len(rows)} structured C fixtures and {MANIFEST}")


if __name__ == "__main__":
    main()
