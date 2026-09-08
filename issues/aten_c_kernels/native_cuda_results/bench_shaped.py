#!/usr/bin/env python3
"""Shape-matched ATen benchmark for CUDA or x86 CPU.

Reads resident_shape_specs.json and measures the corresponding real torch op
at the same shape and dtype as the raised resident harness.  CUDA timing uses
a synchronized wall-clock scope by default so it is directly comparable to
the raised harness.  Prints one machine-readable kernel line per spec.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

d = os.environ.get("ATEN_BENCH_DEVICE", "cuda")
timing = os.environ.get("ATEN_BENCH_TIMING", "sync_wall")
SPECS = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "shape_specs.json"))
validate_only = os.environ.get("ATEN_BENCH_VALIDATE_ONLY", "0") not in {
    "", "0", "false", "FALSE"
}
if validate_only:
    for spec in SPECS:
        spec["dims"] = {
            key: ([min(int(item), 4) for item in value]
                  if isinstance(value, list) else min(int(value), 8))
            for key, value in spec["dims"].items()
        }
        spec["n"] = min(int(spec.get("n", 128)), 128)
        spec["shape"] = "VALIDATE_ONLY_" + spec["shape"]
only_kernels = {
    item for item in os.environ.get("ATEN_BENCH_KERNELS", "").split(",") if item
}
if only_kernels:
    SPECS = [spec for spec in SPECS if spec["kernel"] in only_kernels]
SPEC_BY_KERNEL = {spec["kernel"]: spec for spec in SPECS}
log_path = os.environ.get("ATEN_BENCH_LOG")
log_stream = open(log_path, "w") if log_path else None


def emit(message):
    print(message, flush=True)
    if log_stream:
        print(message, file=log_stream, flush=True)


emit(f"META device={d} timing={timing} torch={torch.__version__} "
     f"threads={torch.get_num_threads()} "
     f"interop_threads={torch.get_num_interop_threads()}")

UNARY_FN = {
 "abs":torch.abs,"acos":torch.acos,"acosh":lambda t:torch.acosh(t+1),"asin":torch.asin,
 "asinh":torch.asinh,"atan":torch.atan,"atanh":torch.atanh,"ceil":torch.ceil,"cos":torch.cos,
 "cosh":torch.cosh,"exp":torch.exp,"exp2":torch.exp2,"floor":torch.floor,"frac":torch.frac,
 "log":torch.log,"neg":torch.neg,"reciprocal":torch.reciprocal,"rsqrt":torch.rsqrt,
 "sigmoid":torch.sigmoid,"sin":torch.sin,"sinc":torch.sinc,"sinh":torch.sinh,"sqrt":torch.sqrt,
 "square":torch.square,"tan":torch.tan,"tanh":torch.tanh,"erf":torch.erf,"erfc":torch.erfc,
 "conj":lambda t:torch.conj(t).resolve_conj(),"abs_complex":torch.abs,
 "relu":F.relu,"silu":F.silu,"mish":F.mish,"softplus":F.softplus,"elu":F.elu,
 "gelu":F.gelu,"hardswish":F.hardswish,"leaky_relu":F.leaky_relu,
}
BINARY_FN = {"add":torch.add,"mul":torch.mul,"div":torch.div,"pow":torch.pow,
 "hypot":torch.hypot,"logaddexp":torch.logaddexp,"logaddexp2":torch.logaddexp2}


def scalar_arg(scalar_args, name, default):
    value = scalar_args.get(name)
    return float(value) if value is not None else float(default)


def recipe_suffix(kernel):
    fingerprint = SPEC_BY_KERNEL.get(kernel, {}).get("recipe_fingerprint",
                                                     "UNVERSIONED")
    return f"recipe={fingerprint}"


def bench_native_fixture(k, op, dims, n, shape, scalar_args):
    """Benchmark native ATen equivalents omitted by the legacy CAT registry.

    These recipes make omission explicit and reproducible.  Internal-stage and
    materialized-composition cases are still adjudicated by the provenance
    ledger before a paper ratio is emitted.
    """
    unary = {
        "expm1": torch.expm1, "log10": torch.log10, "log1p": torch.log1p,
        "log2": torch.log2, "round": torch.round, "trunc": torch.trunc,
        "hardshrink": F.hardshrink, "hardsigmoid": F.hardsigmoid,
        "hardtanh": F.hardtanh, "softshrink": F.softshrink,
        "log_sigmoid": F.logsigmoid, "logit": torch.logit,
    }
    binary = {
        "atan2": torch.atan2, "fmax": torch.fmax, "fmin": torch.fmin,
        "fmod": torch.fmod, "maximum": torch.maximum,
        "minimum": torch.minimum,
    }
    if k == "aten_div_trunc":
        x = torch.rand(n, device=d)
        y = torch.rand(n, device=d) + 0.5
        bench(k, lambda: torch.div(x, y, rounding_mode="trunc"), shape)
        return
    if (op in unary and "backward" not in k and
            k not in {"aten_round_decimals", "aten_log_sigmoid_cpu"}):
        x = torch.rand(n, device=d) + (0.5 if op.startswith("log") else 0)
        if op == "hardshrink":
            lambd = scalar_arg(scalar_args, "lambd", 0.5)
            fn = lambda: F.hardshrink(x, lambd)
        elif op == "hardtanh":
            lo = scalar_arg(scalar_args, "lo", -1.0)
            hi = scalar_arg(scalar_args, "hi", 1.0)
            fn = lambda: torch.clamp(x, lo, hi)
        elif op == "softshrink":
            lambd = scalar_arg(scalar_args, "lambd", 0.5)
            fn = lambda: F.softshrink(x, lambd)
        elif op == "logit":
            eps = scalar_arg(scalar_args, "eps", 1e-6)
            fn = lambda: torch.logit(x, eps=eps)
        elif op == "round":
            # The extracted C specialization calls roundf (ties away from
            # zero), whereas torch.round uses ties-to-even.
            fn = lambda: torch.where(x >= 0, torch.floor(x + 0.5),
                                     torch.ceil(x - 0.5))
        else:
            fn = lambda x=x, native_fn=unary[op]: native_fn(x)
        bench(k, fn, shape); return
    if op in binary:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d) + 0.5
        bench(k, lambda x=x, y=y, fn=binary[op]: fn(x, y), shape); return
    if op in {"clamp", "clamp_scalar", "clamp_max_scalar", "clamp_min_scalar"}:
        x = torch.rand(n, device=d) * 4 - 2
        lo = scalar_arg(scalar_args, "minval",
                        scalar_arg(scalar_args, "lo", -0.5))
        hi = scalar_arg(scalar_args, "maxval",
                        scalar_arg(scalar_args, "hi", 0.75))
        if k == "aten_clamp_cpu":
            minval = torch.rand(n, device=d) - 1
            maxval = minval + torch.rand(n, device=d) + 0.25
            fn = lambda: torch.minimum(torch.maximum(x, minval), maxval)
        elif op == "clamp_max_scalar": fn = lambda: torch.clamp_max(x, hi)
        elif op == "clamp_min_scalar": fn = lambda: torch.clamp_min(x, lo)
        else: fn = lambda: torch.clamp(x, lo, hi)
        bench(k, fn, shape); return
    if op in {"addcdiv", "addcmul"}:
        x = torch.rand(n, device=d); a = torch.rand(n, device=d)
        b = torch.rand(n, device=d) + 0.5
        fn = torch.addcdiv if op == "addcdiv" else torch.addcmul
        bench(k, lambda x=x, a=a, b=b, fn=fn: fn(x, a, b, value=0.75), shape); return
    if op and op.startswith("lerp"):
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        weight = (torch.rand(n, device=d)
                  if op in {"lerp", "lerp_tensor"}
                  else scalar_arg(scalar_args, "weight", 0.375))
        bench(k, lambda x=x, y=y, weight=weight: torch.lerp(x, y, weight), shape); return
    if op == "div_trunc":
        x = torch.rand(n, device=d); y = torch.rand(n, device=d) + 0.5
        bench(k, lambda: torch.div(x, y, rounding_mode="trunc"), shape); return
    if op in {"gelu_backward_cpu_exact", "gelu_backward_cpu_tanh"}:
        x = torch.rand(n, device=d); grad = torch.rand(n, device=d)
        approx = "tanh" if op.endswith("tanh") else "none"
        bench(k, lambda: torch.ops.aten.gelu_backward.default(grad, x, approximate=approx), shape); return
    if op == "gelu_cpu_tanh":
        x = torch.rand(n, device=d)
        bench(k, lambda: F.gelu(x, approximate="tanh"), shape); return
    if op in {"hardsigmoid", "glu"} and "backward" in k:
        x = torch.rand(n, device=d); grad = torch.rand(n, device=d)
        if op == "hardsigmoid": fn = lambda: torch.ops.aten.hardsigmoid_backward.default(grad, x)
        else:
            sigmoid_b = torch.rand(n, device=d)
            a = torch.rand(n, device=d)
            fn = lambda: (1.0 - sigmoid_b) * sigmoid_b * grad * a
        bench(k, fn, shape); return
    if op == "log_sigmoid" and "backward" in k:
        x = torch.rand(n, device=d); grad = torch.rand(n, device=d)
        _, buffer = torch.ops.aten.log_sigmoid_forward.default(x)
        bench(k, lambda: torch.ops.aten.log_sigmoid_backward.default(
            grad, x, buffer), shape); return
    if k == "aten_log_sigmoid_cpu":
        x = torch.rand(n, device=d)
        bench(k, lambda: torch.ops.aten.log_sigmoid_forward.default(x),
              shape); return
    if op == "glu":
        x = torch.rand(n, 2, device=d)
        bench(k, lambda: F.glu(x, -1), shape); return
    if op in {"hardshrink", "shrink", "threshold"} and "backward" in k:
        x = torch.rand(n, device=d) * 2 - 1; grad = torch.rand(n, device=d)
        if op == "threshold":
            threshold = scalar_arg(scalar_args, "threshold", 0.0)
            fn = lambda: torch.ops.aten.threshold_backward.default(
                grad, x, threshold)
        else:
            lambd = scalar_arg(scalar_args, "lambd", 0.5)
            fn = lambda: grad * (torch.abs(x) > lambd)
        bench(k, fn, shape); return
    if op == "hardtanh" and "backward" in k:
        x = torch.rand(n, device=d) * 4 - 2; grad = torch.rand(n, device=d)
        lo = scalar_arg(scalar_args, "minval", -1.0)
        hi = scalar_arg(scalar_args, "maxval", 1.0)
        bench(k, lambda: torch.ops.aten.hardtanh_backward.default(
            grad, x, lo, hi), shape); return
    if op in {"huber", "huber_elementwise", "smooth_l1", "smooth_l1_elementwise", "mse", "mse_elementwise"}:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        delta = scalar_arg(scalar_args, "delta", 1.0)
        beta = scalar_arg(scalar_args, "beta", 1.0)
        if "huber" in op: fn = lambda: F.huber_loss(x, y, reduction="none", delta=delta)
        elif "smooth" in op: fn = lambda: F.smooth_l1_loss(x, y, reduction="none", beta=beta)
        else: fn = lambda: F.mse_loss(x, y, reduction="none")
        if "backward" in k:
            grad = torch.rand(n, device=d)
            if "huber" in op:
                norm = scalar_arg(scalar_args, "norm", 1.0)
                fn = lambda: torch.where(x - y < -delta, -norm * delta,
                    torch.where(x - y > delta, norm * delta, norm * (x - y)))
            elif "smooth" in op:
                norm = scalar_arg(scalar_args, "norm", 1.0)
                fn = lambda: torch.where(x - y <= -beta, -norm,
                    torch.where(x - y >= beta, norm, norm * (x - y) / beta))
            elif "mse" in op:
                value = scalar_arg(scalar_args, "value", 1.0)
                scalar_grad = torch.full_like(x, value * 0.5)
                fn = lambda: torch.ops.aten.mse_loss_backward.default(
                    scalar_grad, x, y, 0)
        bench(k, fn, shape); return
    if op == "logit" and "backward" in k:
        x = torch.rand(n, device=d) * 0.8 + 0.1; grad = torch.rand(n, device=d)
        eps = scalar_arg(scalar_args, "eps", 1e-6)
        bench(k, lambda: torch.ops.aten.logit_backward.default(
            grad, x, eps), shape); return
    if op in {"masked_scale", "addr_elementwise"}:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        if op == "masked_scale":
            scale = scalar_arg(scalar_args, "inv_scale", 1.0)
            fn = lambda: x * scale
        else:
            self_value = torch.rand(n, device=d)
            beta = scalar_arg(scalar_args, "beta", 1.0)
            alpha = scalar_arg(scalar_args, "alpha", 1.0)
            fn = (lambda: alpha * x * y if beta == 0.0
                  else beta * self_value + alpha * x * y)
        bench(k, fn, shape); return

    if op in {"aminmax", "aminmax_allreduce", "max_all", "max_reduce",
              "max_values", "min_all", "min_reduce", "min_values", "prod",
              "blas_sum", "sum_cpu_backend", "nansum", "or_reduce", "xor_sum"}:
        rows = geti(dims, "R", default=0); cols = geti(dims, "K", default=64)
        x = torch.rand(rows, cols, device=d) if rows else torch.rand(n, device=d)
        dim = 1 if rows else None
        if op.startswith("aminmax"): fn = lambda: torch.aminmax(x, dim=dim) if dim is not None else torch.aminmax(x)
        elif op.startswith("max"): fn = lambda: torch.max(x, dim=dim).values if dim is not None else torch.max(x)
        elif op.startswith("min"): fn = lambda: torch.min(x, dim=dim).values if dim is not None else torch.min(x)
        elif op == "prod": fn = lambda: torch.prod(x)
        elif op == "nansum": fn = lambda: torch.nansum(x, dim=dim) if dim is not None else torch.nansum(x)
        elif op == "or_reduce":
            xi = (x > 0.5).to(torch.int32)
            fn = lambda: torch.any(xi != 0, dim=dim).to(torch.int32)
        elif op == "xor_sum":
            xi = torch.randint(0, 8, x.shape, device=d, dtype=torch.int32)
            # XOR has no reduction overload in ATen. Parity of each bit is an
            # exact bounded-width composition for the int32 fixture.
            fn = lambda: sum(
                ((torch.sum((xi >> bit) & 1, dim=dim) & 1) << bit)
                for bit in range(3))
        else: fn = lambda: torch.sum(x, dim=dim) if dim is not None else torch.sum(x)
        bench(k, fn, shape); return
    if op == "equal":
        x = torch.rand(n, device=d); y = x.clone()
        bench(k, lambda: torch.equal(x, y), shape); return
    if op == "trace":
        size = geti(dims, "N", default=2048); x = torch.rand(size, size, device=d)
        bench(k, lambda: torch.trace(x), shape); return

    if op in {"copy", "copy_tensor_array", "nested_clone", "nested_squeeze"}:
        x = torch.rand(n, device=d)
        bench(k, lambda: x.clone(), shape); return
    if op == "zeros":
        bench(k, lambda: torch.zeros(n, device=d), shape); return
    if op in {"transpose_copy", "narrow_copy_dense"}:
        rows = geti(dims, "M", "R", default=2048); cols = geti(dims, "N", "C", default=2048)
        x = torch.rand(rows, cols, device=d)
        if op == "transpose_copy":
            fn = lambda: x.t().contiguous()
        else:
            length = geti(dims, "L", default=max(1, cols // 2))
            start = min(8, max(0, cols - 1))
            length = min(length, cols - start)
            fn = lambda: torch.narrow_copy(x, 1, start, length)
        bench(k, fn, shape); return
    if op in {"pixel_shuffle", "pixel_shuffle_cpu_backend", "pixel_unshuffle_cpu_backend"}:
        B = geti(dims, "B", default=4); C = geti(dims, "C", default=16)
        H = geti(dims, "H", default=32); W = geti(dims, "W", default=32)
        r = geti(dims, "RATIO", "R", default=2)
        if "unshuffle" in op:
            x = torch.rand(B, C, H * r, W * r, device=d); fn = lambda: F.pixel_unshuffle(x, r)
        else:
            x = torch.rand(B, C * r * r, H, W, device=d); fn = lambda: F.pixel_shuffle(x, r)
        bench(k, fn, shape); return
    if op in {"repeat_compute", "repeat_tensor_shape"}:
        N = geti(dims, "N", default=8192); R = geti(dims, "R", default=512)
        x = torch.rand(N, device=d); bench(k, lambda: x.repeat(R), shape); return
    if op in {"stack_serial", "unbind_copy", "block_diag"}:
        B = geti(dims, "B", "T", default=16); N = geti(dims, "N", "K", default=256)
        tensors = [torch.rand(N, device=d) for _ in range(B)]
        if op == "stack_serial":
            R = geti(dims, "R", default=1)
            tensors = [torch.rand(R, N, device=d) for _ in range(B)]
            fn = lambda: torch.stack(tensors, dim=1)
        elif op == "unbind_copy":
            # The extracted fixture has one contiguous BxN output.
            x = torch.stack(tensors); fn = lambda: x.clone()
        else:
            matrices = [torch.rand(N,N,device=d) for _ in range(B)]
            fn = lambda: torch.block_diag(*matrices)
        bench(k, fn, shape); return

    if op in {"angle", "angle_complex", "angle_real", "as_complex", "complex", "conj_complex", "polar"}:
        if op == "angle_complex" or (op == "angle" and "complex" in k):
            re = torch.rand(n, device=d); im = torch.rand(n, device=d)
            fn = lambda: torch.atan2(im, re)
        elif op == "angle_real":
            x = torch.rand(n, device=d) * 2 - 1; fn = lambda: torch.angle(x)
        elif op == "as_complex" or k == "aten_as_complex_cpu":
            x = torch.rand(n, 2, device=d)
            fn = lambda: (x[:, 0].clone(), x[:, 1].clone())
        elif op == "complex" and k == "aten_complex_scalarized":
            x = torch.rand(n, device=d); y = torch.rand(n, device=d)
            fn = lambda: (x.clone(), y.clone())
        elif op == "conj_complex":
            re = torch.rand(n, device=d); im = torch.rand(n, device=d)
            fn = lambda: (re.clone(), torch.neg(im))
        elif op == "complex":
            x = torch.rand(n, device=d); y = torch.rand(n, device=d)
            fn = lambda: torch.complex(x, y)
        else:
            magnitude = torch.rand(n, device=d); angle = torch.rand(n, device=d)
            fn = lambda: (magnitude * torch.cos(angle),
                          magnitude * torch.sin(angle))
        bench(k, fn, shape); return
    if op in {"cross", "cross_cpu_backend"}:
        count = geti(dims, "V", "N", default=max(1, n // 3))
        x = torch.rand(count, 3, device=d); y = torch.rand(count, 3, device=d)
        bench(k, lambda: torch.linalg.cross(x, y), shape); return

    if op == "conv1d":
        B=geti(dims,"B",default=32); IC=geti(dims,"IC",default=64); OC=geti(dims,"OC",default=128)
        W=geti(dims,"W",default=4096); K=geti(dims,"K",default=3)
        x=torch.rand(B,IC,W,device=d); w=torch.rand(OC,IC,K,device=d); bias=torch.rand(OC,device=d)
        bench(k, lambda: F.conv1d(x,w,bias), shape); return
    if op in {"depthwise_conv3x3", "dilated_convolution"}:
        B=geti(dims,"B",default=2); C=geti(dims,"C",default=64); O=geti(dims,"O",default=C)
        H=geti(dims,"H",default=128); W=geti(dims,"W",default=128); K=geti(dims,"K",default=3)
        dilation=geti(dims,"D",default=1); groups=C if op=="depthwise_conv3x3" else 1
        x=torch.rand(B,C,H,W,device=d); w=torch.rand(O,C//groups,K,K,device=d)
        bias = torch.rand(O, device=d) if op == "depthwise_conv3x3" else None
        padding = dilation if op == "depthwise_conv3x3" else 0
        bench(k, lambda: F.conv2d(x, w, bias, padding=padding,
                                  dilation=dilation, groups=groups), shape); return
    if op and op.startswith("conv_transpose3d"):
        C=geti(dims,"C",default=8); O=geti(dims,"O",default=16); D=geti(dims,"D",default=32)
        H=geti(dims,"H",default=32); W=geti(dims,"W",default=32); K=geti(dims,"K",default=3)
        if "grad_weight" in k:
            x=torch.rand(1,C,D,H,W,device=d)
            g=torch.rand(1,O,D+K-1,H+K-1,W+K-1,device=d)
            fn=lambda: torch.nn.grad.conv3d_weight(
                g, (C,O,K,K,K), x, stride=1, padding=0)
        elif "backward" in k:
            g=torch.rand(1,O,D+K-1,H+K-1,W+K-1,device=d)
            w=torch.rand(C,O,K,K,K,device=d)
            fn=lambda: F.conv3d(g,w)
        else:
            x=torch.rand(1,C,D,H,W,device=d); w=torch.rand(C,O,K,K,K,device=d)
            fn=lambda: F.conv_transpose3d(x,w)
        bench(k, fn, shape); return
    if op == "int_mm":
        M=max(17,geti(dims,"M",default=512))
        N=max(24,geti(dims,"N",default=512)); N=((N+7)//8)*8
        K=max(24,geti(dims,"K",default=1024)); K=((K+7)//8)*8
        x=torch.randint(-8,8,(M,K),device=d,dtype=torch.int8); y=torch.randint(-8,8,(K,N),device=d,dtype=torch.int8)
        bench(k, lambda: torch._int_mm(x,y), shape); return
    if op in {"bilinear", "trilinear", "nested_matmul_broadcast"}:
        B=geti(dims,"B",default=64); I=geti(dims,"I","M",default=128)
        J=geti(dims,"J","K",default=128); O=geti(dims,"O","N",default=128)
        if op == "nested_matmul_broadcast":
            x=torch.rand(B,I,J,device=d); w=torch.rand(J,O,device=d)
            fn=lambda: torch.matmul(x,w)
        elif op == "trilinear":
            K=geti(dims,"K",default=128)
            x=torch.rand(B,I,device=d); y=torch.rand(B,J,device=d)
            w=torch.rand(I,J,K,device=d)
            fn=lambda: torch.einsum("bi,ijk,bj->bk",x,w,y)
        else:
            x=torch.rand(B,I,device=d); y=torch.rand(B,J,device=d); w=torch.rand(O,I,J,device=d)
            fn=lambda: F.bilinear(x,y,w)
        bench(k, fn, shape); return

    if op in {"batch_norm_cpu_entry", "blas_scale", "linear_combination",
              "renorm_scale_factor"}:
        x = torch.rand(n, device=d)
        if op == "batch_norm_cpu_entry":
            scale = scalar_arg(scalar_args, "scale", 1.0)
            bias = scalar_arg(scalar_args, "bias", 0.0)
            fn = lambda: x * scale + bias
        elif op == "blas_scale":
            scale = scalar_arg(scalar_args, "a", 1.0)
            fn = lambda: x.mul_(scale)
        elif op == "linear_combination":
            inputs = torch.rand(4, n, device=d)
            coefficients = torch.tensor([0.25, 0.5, 0.75, 1.1], device=d)
            fn = lambda: torch.sum(inputs * coefficients[:, None], dim=0)
        else:
            maxnorm = scalar_arg(scalar_args, "maxnorm", 1.0)
            fn = lambda: torch.where(x > maxnorm,
                maxnorm / (x + 1.0e-7), torch.ones_like(x))
        bench(k, fn, shape); return
    if op in {"diag", "block_diag"}:
        B = geti(dims, "B", default=16); N = geti(dims, "N", default=128)
        xs = [torch.rand(N, N, device=d) for _ in range(B)]
        bench(k, lambda: torch.block_diag(*xs), shape); return
    if op == "conv_tbc":
        T=geti(dims,"T",default=1024); B=geti(dims,"B",default=8)
        I=geti(dims,"I",default=32); O=geti(dims,"O",default=64); K=geti(dims,"K",default=3)
        if "backward" in k:
            g=torch.rand(T,B,O,device=d); w=torch.rand(K,I,O,device=d)
            # The extracted component is the input-gradient convolution with
            # full temporal padding.
            fn=lambda: torch.conv_transpose1d(
                g.permute(1,2,0), w.permute(2,1,0), padding=0
            ).permute(2,0,1)
        else:
            x=torch.rand(T,B,I,device=d); w=torch.rand(K,I,O,device=d)
            # conv_tbc requires a bias tensor; an untimed zero bias is exactly
            # equivalent to the extracted no-bias specialization.
            bias=torch.zeros(O,device=d)
            fn=lambda: torch.conv_tbc(x,w,bias,0)
        bench(k, fn, shape); return
    if op in {"dirichlet_grad", "standard_gamma_grad", "gamma"}:
        x=torch.rand(n,device=d)+0.5; alpha=torch.rand(n,device=d)+0.5
        if op == "dirichlet_grad":
            total=torch.rand(n,device=d)+0.5
            fn=lambda:x*(torch.log(x+0.001)-alpha/total)
        elif op == "gamma":
            normal=torch.rand(n,device=d); dgamma=alpha-1.0/3.0
            fn=lambda:dgamma*(1+normal/torch.sqrt(9*dgamma))**3
        else:
            fn=lambda:(x-alpha)/(alpha+0.001)+torch.log(x+0.001)
        bench(k, fn, shape); return
    if op == "embedding_bag_counts":
        E=geti(dims,"E",default=65536); N=geti(dims,"N",default=n)
        indices=torch.randint(E,(N,),device=d)
        bench(k, lambda: torch.bincount(indices,minlength=E).to(torch.int32), shape); return
    if op == "glu_jvp":
        x=torch.rand(n,2,device=d); dx=torch.rand_like(x)
        a,b=x.unbind(-1); da,db=dx.unbind(-1)
        sigmoid_b=torch.sigmoid(b); result=a*sigmoid_b
        bench(k, lambda: da*sigmoid_b+result*(db-sigmoid_b*db), shape); return
    if op == "gradient":
        x=torch.rand(n,device=d); spacing=scalar_arg(scalar_args,"h",1.0)
        bench(k, lambda: torch.gradient(x, spacing=spacing), shape); return
    if op in {"hspmm", "sparse_addmv_csr", "sparse_addmv_bsr", "sspaddmm"}:
        R=geti(dims,"R",default=4096); C=geti(dims,"C",default=4096)
        nnz=min(geti(dims,"N",default=R*16),R*C); per=max(1,nnz//R); nnz=per*R
        if op == "sparse_addmv_bsr":
            BR=geti(dims,"BR",default=4); BC=geti(dims,"BC",default=4)
            block_rows=R; block_cols=C
            crow=torch.arange(0,nnz+1,per,device=d,dtype=torch.int64)
            col=torch.arange(nnz,device=d,dtype=torch.int64)%block_cols
            values=torch.rand(nnz,BR,BC,device=d)
            sparse=torch.sparse_bsr_tensor(crow,col,values,
                size=(block_rows*BR,block_cols*BC),device=d)
            rhs=torch.rand(block_cols*BC,1,device=d)
        elif op in {"hspmm", "sspaddmm"}:
            K=geti(dims,"K",default=R)
            linear=torch.arange(nnz,device=d,dtype=torch.int64)
            row=linear//per; col=linear%K; values=torch.rand(nnz,device=d)
            sparse=torch.sparse_coo_tensor(torch.stack((row,col)),values,
                                            (R,K),device=d).coalesce()
            rhs=torch.rand(K,C,device=d)
        else:
            crow=torch.arange(0,nnz+1,per,device=d,dtype=torch.int64)
            col=torch.arange(nnz,device=d,dtype=torch.int64)%C
            values=torch.rand(nnz,device=d)
            sparse=torch.sparse_csr_tensor(crow,col,values,size=(R,C),device=d)
            rhs=torch.rand(C,1,device=d)
        bench(k, lambda: torch.sparse.mm(sparse,rhs), shape); return
    if op in {"sparse_intersection_apply", "sparse_intersection_launch"}:
        x=torch.rand(n,device=d); y=torch.rand(n,device=d)
        bench(k, lambda: torch.mul(x,y), shape); return
    if op == "sparse_norm":
        x=torch.rand(n,device=d); bench(k, lambda: torch.linalg.vector_norm(x), shape); return
    if op == "unfolded2d_copy":
        C=geti(dims,"C",default=3); H=geti(dims,"H",default=32); W=geti(dims,"W",default=32)
        KH=geti(dims,"KH",default=3); KW=geti(dims,"KW",default=3)
        x=torch.rand(1,C,H,W,device=d); bench(k, lambda: F.unfold(x,(KH,KW)), shape); return
    if op == "upsample_bilinear2d":
        B=geti(dims,"B",default=16); C=geti(dims,"C",default=3)
        H=geti(dims,"H",default=32); W=geti(dims,"W",default=32)
        x=torch.rand(B,C,H,W,device=d)
        # The fixed fixture uses floor(oh/2) with 0/0.5 weights rather than
        # PyTorch's half-pixel convention. Keep this as an explicit exact
        # composition, not a claim about interpolate semantics.
        y0=torch.arange(2*H,device=d)//2; y1=torch.clamp(y0+1,max=H-1)
        x0=torch.arange(2*W,device=d)//2; x1=torch.clamp(x0+1,max=W-1)
        fy=(torch.arange(2*H,device=d)%2).to(x.dtype)*0.5
        fx=(torch.arange(2*W,device=d)%2).to(x.dtype)*0.5
        fn=lambda: ((1-fy)[None,None,:,None]*(
            (1-fx)[None,None,None,:]*x[:,:,y0[:,None],x0[None,:]]+
            fx[None,None,None,:]*x[:,:,y0[:,None],x1[None,:]])+
            fy[None,None,:,None]*(
            (1-fx)[None,None,None,:]*x[:,:,y1[:,None],x0[None,:]]+
            fx[None,None,None,:]*x[:,:,y1[:,None],x1[None,:]]))
        bench(k, fn, shape); return

    if op in {"normal", "log_normal", "uniform", "exponential", "geometric", "cauchy"}:
        x=torch.empty(n,device=d)
        # gen_shape_specs historically groups both fixtures under log_normal;
        # the kernel name disambiguates the non-exponentiating normal case.
        if op == "normal" or k == "aten_normal_cpu":
            z=torch.rand(n,device=d); mean=scalar_arg(scalar_args,"mean",0.0)
            std=scalar_arg(scalar_args,"std",1.0); fn=lambda: mean+std*z
        elif op == "log_normal":
            z=torch.rand(n,device=d); mean=scalar_arg(scalar_args,"mean",0.0)
            std=scalar_arg(scalar_args,"std",1.0); fn=lambda: torch.exp(mean+std*z)
        elif op == "uniform":
            z=torch.rand(n,device=d); lo=scalar_arg(scalar_args,"from",0.0)
            hi=scalar_arg(scalar_args,"to",1.0); fn=lambda: lo+(hi-lo)*z
        elif op == "exponential":
            z=torch.rand(n,device=d); rate=scalar_arg(scalar_args,"lambda",1.0)
            fn=lambda: -torch.log1p(-z)/rate
        elif op == "geometric":
            z=torch.rand(n,device=d); probability=scalar_arg(scalar_args,"probability",0.5)
            fn=lambda: torch.ceil(torch.log1p(-z)/torch.log(torch.tensor(
                1.0-probability,device=d)))
        else:
            z=torch.rand(n,device=d); median=scalar_arg(scalar_args,"median",0.0)
            sigma=scalar_arg(scalar_args,"sigma",1.0)
            fn=lambda: median+sigma*torch.tan(torch.pi*(z-0.5))
        bench(k, fn, shape); return

    if k == "aten_round_decimals":
        x=torch.rand(n,device=d); scale=scalar_arg(scalar_args,"scale",1.0)
        bench(k, lambda: torch.where(x*scale >= 0,
            torch.floor(x*scale+0.5), torch.ceil(x*scale-0.5))/scale, shape); return

    # Complete fixture-specific adapters for kernels whose names were absent
    # from the legacy category registry. These reproduce the extracted
    # operation at its explicit shape; compositions are labelled separately
    # by the provenance audit and are not silently presented as one ATen op.
    if k in {"aten_allany_dims_cpu", "aten_and_reduce_cpu"}:
        R=geti(dims,"R",default=32); C=geti(dims,"C","K",default=64)
        x=torch.randint(0,2,(R,C),device=d,dtype=torch.int32)
        use_all = (k == "aten_and_reduce_cpu" or
                   int(scalar_arg(scalar_args,"all",1)) != 0)
        fn=(lambda: torch.all(x != 0,dim=1).to(torch.int32)) if use_all else \
           (lambda: torch.any(x != 0,dim=1).to(torch.int32))
        bench(k,fn,shape); return
    if k in {"aten_bf16_dot_cpu", "aten_fp16_dot_cpu",
             "aten_blas_dot_naive_cpu"}:
        K=geti(dims,"K","N",default=n)
        x=torch.rand(K,device=d); y=torch.rand(K,device=d)
        bench(k,lambda:torch.dot(x,y),shape); return
    if ("gemv" in k and k.startswith(("aten_bf16_", "aten_fp16_",
                                       "aten_blas_"))):
        M=geti(dims,"M",default=64); K=geti(dims,"K",default=96)
        a=torch.rand(M,K,device=d)
        if "trans" in k and "notrans" not in k:
            x=torch.rand(M,device=d); fn=lambda:torch.mv(a.t(),x)
        else:
            x=torch.rand(K,device=d); fn=lambda:torch.mv(a,x)
        bench(k,fn,shape); return
    if k == "aten_blas_axpy_cpu":
        x=torch.rand(n,device=d); y=torch.rand(n,device=d)
        alpha=scalar_arg(scalar_args,"a",0.5)
        bench(k,lambda:y.add_(x,alpha=alpha),shape); return
    if k == "aten_blas_copy_cpu":
        x=torch.rand(n,device=d); bench(k,lambda:x.clone(),shape); return
    if k == "aten_cartesian_prod_cpu":
        A=geti(dims,"A",default=16); B=geti(dims,"B",default=12)
        a=torch.rand(A,device=d); b=torch.rand(B,device=d)
        bench(k,lambda:torch.cartesian_prod(a,b),shape); return
    if k in {"aten_channel_shuffle", "aten_channel_shuffle_cpu"}:
        B=geti(dims,"B",default=2); G=geti(dims,"G",default=2)
        CPG=geti(dims,"CPG",default=4)
        if "cpu" in k:
            S=geti(dims,"S",default=32); x=torch.rand(B,G*CPG,S,device=d)
        else:
            H=geti(dims,"H",default=4); W=geti(dims,"W",default=4)
            x=torch.rand(B,G*CPG,H,W,device=d)
        bench(k,lambda:torch.channel_shuffle(x,G),shape); return
    if k == "aten_compressed_block_convert_cpu":
        R=geti(dims,"R",default=64); C=geti(dims,"C",default=64)
        BR=geti(dims,"BR",default=4); BC=geti(dims,"BC",default=4)
        x=torch.rand(R,C,device=d)
        bench(k,lambda:x.reshape(R//BR,BR,C//BC,BC).permute(0,2,1,3).contiguous(),shape); return
    if k in {"aten_convert_coo_to_csr_cpu", "aten_sparse_coo_to_csr_cpu"}:
        N=geti(dims,"N",default=512); R=geti(dims,"R",default=64)
        row=torch.arange(N,device=d,dtype=torch.int64)*R//N
        bench(k,lambda:torch._convert_indices_from_coo_to_csr(
            row,R,out_int32=True),shape); return
    if k in {"aten_convert_csr_to_coo_cpu",
             "aten_sparse_matmul_csr_to_coo_cpu"}:
        N=geti(dims,"N",default=512); R=geti(dims,"R",default=64)
        counts=torch.full((R,),N//R,device=d,dtype=torch.int32)
        counts[:N%R]+=1; ptr=torch.cat((torch.zeros(1,device=d,dtype=torch.int32),counts.cumsum(0)))
        bench(k,lambda:torch.repeat_interleave(
            torch.arange(R,device=d,dtype=torch.int32),ptr[1:]-ptr[:-1]),shape); return
    if k in {"aten_cpu_blas_gemm_cpu"} or k.startswith("aten_gemm_"):
        M=geti(dims,"M",default=32); N=geti(dims,"N",default=48); K=geti(dims,"K",default=40)
        a_shape=(K,M) if "transa" in k else (M,K)
        b_shape=(N,K) if "transb" in k or "transab" in k else (K,N)
        a=torch.rand(*a_shape,device=d); b=torch.rand(*b_shape,device=d)
        aa=a.t() if "transa" in k else a; bb=b.t() if "transb" in k or "transab" in k else b
        if k.startswith("aten_gemm_"):
            c=torch.rand(M,N,device=d); fn=lambda:c+aa@bb
        else: fn=lambda:aa@bb
        bench(k,fn,shape); return
    if k in {"aten_cpu_blas_gemm_batched_cpu",
             "aten_cpu_blas_gemm_strided_batched_cpu",
             "aten_nested_bmm_cpu", "aten_sparse_bmm_cpu",
             "aten_sumproduct_pair_cpu"}:
        B=geti(dims,"B",default=4); M=geti(dims,"M",default=16)
        N=geti(dims,"N",default=20); K=geti(dims,"K",default=24)
        a=torch.rand(B,M,K,device=d); b=torch.rand(B,K,N,device=d)
        bench(k,lambda:torch.bmm(a,b),shape); return
    if k == "aten_diff_cpu":
        x=torch.rand(geti(dims,"N",default=n),device=d)
        bench(k,lambda:torch.diff(x),shape); return
    if k == "aten_dropout_feature_noise_cpu":
        B=geti(dims,"B",default=8); C=geti(dims,"C",default=16)
        H=geti(dims,"H",default=8); W=geti(dims,"W",default=8)
        x=torch.rand(B,C,H,W,device=d); mask=torch.rand(B,C,device=d)
        scale=scalar_arg(scalar_args,"scale",1.0)
        bench(k,lambda:x*mask[:,:,None,None]*scale,shape); return
    if k == "aten_fast_cat_dim0_cpu":
        B=geti(dims,"B",default=4); N=geti(dims,"N",default=256)
        x=torch.rand(B,N,device=d); bench(k,lambda:x.reshape(-1).clone(),shape); return
    if k == "aten_flatten_nd_linear_cpu":
        B=geti(dims,"B",default=16); M=geti(dims,"M",default=32)
        K=geti(dims,"K",default=64); N=geti(dims,"N",default=48)
        x=torch.rand(B,M,K,device=d); w=torch.rand(K,N,device=d)
        bench(k,lambda:torch.matmul(x,w),shape); return
    if k == "aten_flip_tensor_transform_cpu":
        R=geti(dims,"R",default=32); C=geti(dims,"C",default=64)
        x=torch.rand(R,C,device=d); bench(k,lambda:torch.flip(x,(0,1)),shape); return
    if k == "aten_gradient_float_cpu":
        N=geti(dims,"N",default=n); x=torch.rand(N,device=d)
        coord=torch.arange(N,device=d,dtype=x.dtype)*0.125
        bench(k,lambda:torch.gradient(x,spacing=(coord,)),shape); return
    if k == "aten_histogram_select_outer_bin_edges_cpu":
        x=torch.rand(n,device=d); bench(k,lambda:torch.aminmax(x),shape); return
    if k == "aten_joint_scaling_cpu":
        x=torch.rand(n,device=d)*2-1; y=torch.rand(n,device=d)*2-1
        bench(k,lambda:torch.max(torch.abs(x))*torch.max(torch.abs(y)),shape); return
    if k in {"aten_kron_impl_cpu", "aten_kron_out_cpu"}:
        A=geti(dims,"A",default=16); B=geti(dims,"B",default=12)
        C=geti(dims,"C",default=8); D=geti(dims,"D",default=10)
        x=torch.rand(A,B,device=d); y=torch.rand(C,D,device=d)
        bench(k,lambda:torch.kron(x,y),shape); return
    if k in {"aten_nested_all_cpu", "aten_nested_sum_dim_cpu"}:
        B=geti(dims,"B",default=8); N=geti(dims,"N",default=64)
        lengths=torch.arange(B,device=d,dtype=torch.int64)%N+1
        mask=torch.arange(N,device=d)[None,:]<lengths[:,None]
        if k == "aten_nested_all_cpu":
            x=torch.randint(0,2,(B,N),device=d,dtype=torch.int32)
            fn=lambda:torch.all((x!=0)|~mask,dim=1).to(torch.int32)
        else:
            x=torch.rand(B,N,device=d); fn=lambda:torch.sum(x*mask,dim=1)
        bench(k,fn,shape); return
    if k == "aten_nested_batch_offsets_cpu":
        B=geti(dims,"B",default=64); sizes=torch.randint(1,9,(B,),device=d,dtype=torch.int32)
        bench(k,lambda:torch.cat((torch.zeros(1,device=d,dtype=torch.int32),sizes.cumsum(0))),shape); return
    if k == "aten_quant_col_offsets_cpu":
        K=geti(dims,"K",default=64); N=geti(dims,"N",default=48)
        w=torch.randint(-8,8,(K,N),device=d,dtype=torch.int8)
        zero=int(scalar_arg(scalar_args,"zero",0))
        bench(k,lambda:w.to(torch.int32).sum(0)-zero*K,shape); return
    if k.startswith("aten_slow_conv3d_"):
        C=geti(dims,"C",default=2); O=geti(dims,"O",default=3)
        D=geti(dims,"D",default=8); H=geti(dims,"H",default=9)
        W=geti(dims,"W",default=10); K=geti(dims,"K",default=3)
        if "backward_input" in k:
            g=torch.rand(1,O,D,H,W,device=d); w=torch.rand(O,C,K,K,K,device=d)
            fn=lambda:F.conv_transpose3d(g,w)
        elif "backward_weight" in k:
            x=torch.rand(1,C,D+K-1,H+K-1,W+K-1,device=d)
            g=torch.rand(1,O,D,H,W,device=d)
            fn=lambda:torch.nn.grad.conv3d_weight(x,(O,C,K,K,K),g)
        else:
            x=torch.rand(1,C,D,H,W,device=d); w=torch.rand(O,C,K,K,K,device=d)
            fn=lambda:F.conv3d(x,w)
        bench(k,fn,shape); return
    if k == "aten_sparse_matmul_cpu":
        R=geti(dims,"R",default=64); C=geti(dims,"C",default=64)
        N=geti(dims,"N",default=512); per=max(1,N//R); N=per*R
        ptr=torch.arange(0,N+1,per,device=d,dtype=torch.int64)
        col=torch.arange(N,device=d,dtype=torch.int64)%R
        values=torch.rand(N,device=d)
        a=torch.sparse_csr_tensor(ptr,col,values,size=(R,R),device=d)
        b=torch.rand(R,C,device=d); bench(k,lambda:torch.sparse.mm(a,b),shape); return
    if k == "aten_transform_bias_rescale_qkv_cpu":
        B=geti(dims,"B",default=2); S=geti(dims,"S",default=16)
        H=geti(dims,"H",default=4); D=geti(dims,"D",default=8)
        qkv=torch.rand(B,S,3,H,D,device=d); bias=torch.rand(3,H,D,device=d)
        scale=scalar_arg(scalar_args,"scale",1.0)
        fn=lambda:tuple((qkv[:,:,i]+bias[i]).permute(0,2,1,3)*(
            scale if i==0 else 1.0) for i in range(3))
        bench(k,fn,shape); return

    metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
    emit(f"kernel={k} {metric}=SKIP err=missing_native_fixture_adapter:{op} "
         f"{recipe_suffix(k)} shape='{shape}'")


def bench(name, fn, shape, warm=5, it=20):
    if validate_only:
        warm, it = 0, 1
    try:
        for _ in range(warm):
            fn()
        if d == "cuda":
            torch.cuda.synchronize()
        best = 1e30
        for _ in range(it):
            if d == "cuda" and timing == "cuda_event":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record(); torch.cuda.synchronize()
                elapsed_us = s.elapsed_time(e) * 1000.0
            else:
                start = time.perf_counter_ns(); fn()
                if d == "cuda":
                    torch.cuda.synchronize()
                elapsed_us = (time.perf_counter_ns() - start) / 1000.0
            best = min(best, elapsed_us)
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={name} {metric}={best:.3f} timing={timing} "
             f"{recipe_suffix(name)} shape='{shape}'")
    except Exception as ex:
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={name} {metric}=SKIP err={type(ex).__name__}:{ex} "
             f"{recipe_suffix(name)} shape='{shape}'")


def geti(dims, *keys, default=1):
    for k in keys:
        v = dims.get(k)
        if isinstance(v, int):
            return v
    return default


def getlist(dims, key, n, default):
    v = dims.get(key)
    if isinstance(v, list) and len(v) >= n:
        return v[:n]
    if isinstance(v, int):
        return [v] * n
    return default


for sp in SPECS:
    k, op, cat, dims, n, shape = sp["kernel"], sp["op"], sp["cat"], sp["dims"], sp["n"], sp["shape"]
    try:
        torch.set_default_dtype(torch.float64 if sp.get("dtype") == "f64"
                                else torch.float32)
        if cat == "unary":
            a = torch.rand(n, device=d)
            if op in ("acosh","sqrt","rsqrt","log"):  # positive domain
                a = a + 1.0
            fn = UNARY_FN.get(op, torch.abs)
            if op in ("conj","abs_complex"):
                a = torch.rand(n, dtype=torch.complex64, device=d)
            bench(k, (lambda a=a, fn=fn: fn(a)), shape)
        elif cat == "binary":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d) + 0.5
            fn = BINARY_FN[op]
            bench(k, (lambda a=a, b=b, fn=fn: fn(a, b)), shape)
        elif cat == "add_alpha":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d)
            alpha = scalar_arg(sp.get("scalar_args", {}), "alpha", 0.75)
            bench(k, (lambda a=a, b=b, alpha=alpha:
                      torch.add(a, b, alpha=alpha)), shape)
        elif cat == "add_clamp":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d)
            bench(k, (lambda a=a, b=b:
                      torch.clamp(torch.add(a, b, alpha=0.75), -0.5, 0.75)), shape)
        elif cat == "div_floor":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d) + 0.5
            bench(k, (lambda a=a, b=b:
                      torch.div(a, b, rounding_mode="floor")), shape)
        elif cat == "pow_scalar":
            a = torch.rand(n, device=d)
            bench(k, (lambda a=a: torch.pow(a, 1.75)), shape)
        elif cat == "loss":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d)
            fn = F.mse_loss if op == "mse_loss" else F.smooth_l1_loss
            bench(k, (lambda a=a, b=b, fn=fn: fn(a, b)), shape)
        elif cat == "loss_bce":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d)
            bench(k, (lambda a=a, b=b: F.binary_cross_entropy(a, b)), shape)
        elif cat == "dot":
            a = torch.rand(n, device=d); b = torch.rand(n, device=d)
            bench(k, (lambda a=a, b=b: torch.dot(a, b)), shape)
        elif cat == "reduce":
            R = geti(dims, "R", "M", "rows", default=0)
            if R:
                C = geti(dims, "K", "C", "N", "cols", default=64)
                x = torch.rand(R, C, device=d)
                fnmap = {"sum": lambda x: torch.sum(x, 1),
                         "count_nonzero": lambda x: torch.count_nonzero(x, 1),
                         "all": lambda x: torch.all(x > 0.5, 1)}
            else:
                x = torch.rand(n, device=d)
                fnmap = {"sum": torch.sum, "count_nonzero": torch.count_nonzero,
                         "all": lambda x: torch.all(x > 0.5)}
            bench(k, (lambda x=x, f=fnmap[op]: f(x)), shape)
        elif cat == "mean":
            x = torch.rand(n, device=d)
            bench(k, (lambda x=x: torch.mean(x)), shape)
        elif cat == "reduce_arg":
            R = geti(dims, "R", "rows", default=131072); C = geti(dims, "K", "cols", default=64)
            x = torch.rand(R, C, device=d)
            f = torch.argmax if op == "argmax" else torch.argmin
            bench(k, (lambda x=x, f=f: f(x, dim=1)), shape)
        elif cat == "norm":
            x = torch.rand(n, device=d)
            bench(k, (lambda x=x: torch.norm(x)), shape)
        elif cat == "cum":
            R = geti(dims, "R", "rows", default=0)
            if R:
                C = geti(dims, "K", "cols", default=64); x = torch.rand(R, C, device=d); dim = 1
            else:
                x = torch.rand(n, device=d); dim = 0
            f = torch.cumsum if op == "cumsum" else torch.cumprod
            bench(k, (lambda x=x, f=f, dim=dim: f(x, dim)), shape)
        elif cat == "cumprod_backward":
            x = torch.rand(n, device=d) + 0.5
            grad = torch.rand(n, device=d)
            product = torch.cumprod(x, 0)
            bench(k, (lambda grad=grad, x=x, product=product:
                      torch.ops.aten.cumprod_backward.default(
                          grad, x, 0, product)), shape)
        elif cat == "sort":
            R = geti(dims, "rows", "R", default=32768); C = geti(dims, "cols", "C", default=256)
            x = torch.rand(R, C, device=d)
            bench(k, (lambda x=x: torch.sort(x, dim=1)), shape)
        elif cat == "topk":
            R = geti(dims, "rows", "R", default=32768); C = geti(dims, "cols", "C", default=256)
            top = geti(dims, "top", default=16); x = torch.rand(R, C, device=d)
            bench(k, (lambda x=x, top=top: torch.topk(x, top, dim=1)), shape)
        elif cat in ("mm", "addmm"):
            M = geti(dims, "M", default=512); N = geti(dims, "N", default=512); K = geti(dims, "K", default=512)
            A = torch.rand(M, K, device=d); B = torch.rand(K, N, device=d)
            if cat == "mm":
                bench(k, (lambda A=A, B=B: torch.mm(A, B)), shape)
            else:
                C = torch.rand(M, N, device=d)
                bench(k, (lambda A=A, B=B, C=C: torch.addmm(C, A, B)), shape)
        elif cat == "gemv":
            M = geti(dims, "M", default=4096); K = geti(dims, "K", default=4096)
            A = torch.rand(M, K, device=d); x = torch.rand(K, device=d)
            bench(k, (lambda A=A, x=x: torch.mv(A, x)), shape)
        elif cat == "outer":
            M = geti(dims, "M", default=2048); N = geti(dims, "N", default=2048)
            x = torch.rand(M, device=d); y = torch.rand(N, device=d)
            bench(k, (lambda x=x, y=y: torch.outer(x, y)), shape)
        elif cat == "bmm":
            B = geti(dims, "B", "BATCH", default=64); M = geti(dims, "M", default=128)
            N = geti(dims, "N", default=128); K = geti(dims, "K", default=128)
            X = torch.rand(B, M, K, device=d); Y = torch.rand(B, K, N, device=d)
            bench(k, (lambda X=X, Y=Y: torch.bmm(X, Y)), shape)
        elif cat == "conv2d":
            B = geti(dims, "B", default=8); IC = geti(dims, "IC", default=32); OC = geti(dims, "OC", default=64)
            H = geti(dims, "H", default=64); W = geti(dims, "W", default=64); K = geti(dims, "KH", "K", default=3)
            x = torch.rand(B, IC, H, W, device=d); w = torch.rand(OC, IC, K, K, device=d)
            bench(k, (lambda x=x, w=w: F.conv2d(x, w, padding=0)), shape)
        elif cat == "unfold2d":
            C = geti(dims, "C", default=3); H = geti(dims, "H", default=16)
            W = geti(dims, "W", default=16); K = geti(dims, "K", default=3)
            x = torch.rand(1, C, H, W, device=d)
            bench(k, (lambda x=x, K=K: F.unfold(x, (K, K))), shape)
        elif cat == "conv3d":
            B = geti(dims, "B", default=1); IC = geti(dims, "IC", "C", default=8); OC = geti(dims, "OC", "O", default=16)
            D = geti(dims, "D", default=48); H = geti(dims, "H", default=48); W = geti(dims, "W", default=48)
            K = geti(dims, "K", default=3)
            x = torch.rand(B, IC, D, H, W, device=d); w = torch.rand(OC, IC, K, K, K, device=d)
            bench(k, (lambda x=x, w=w: F.conv3d(x, w, padding=0)), shape)
        elif cat == "convT2d":
            B = geti(dims, "B", default=2); IC = geti(dims, "IC", default=16); OC = geti(dims, "OC", default=32)
            H = geti(dims, "H", default=128); W = geti(dims, "W", default=128); K = geti(dims, "K", default=3)
            x = torch.rand(B, IC, H, W, device=d); w = torch.rand(IC, OC, K, K, device=d)
            bench(k, (lambda x=x, w=w: F.conv_transpose2d(x, w, stride=2)), shape)
        elif cat in ("maxpool2d", "avgpool2d", "adaptavg2d", "adaptmax2d"):
            B = geti(dims, "B", default=32); C = geti(dims, "C", default=64)
            HW = getlist(dims, "I", 2, [geti(dims, "H", default=64), geti(dims, "W", default=64)])
            x = torch.rand(B, C, HW[0], HW[1], device=d)
            if cat == "maxpool2d":
                bench(k, (lambda x=x: F.max_pool2d(x, 2)), shape)
            elif cat == "avgpool2d":
                bench(k, (lambda x=x: F.avg_pool2d(x, 2)), shape)
            elif cat == "adaptavg2d":
                O = getlist(dims, "O", 2, [geti(dims, "OH", default=4), geti(dims, "OW", default=4)])
                bench(k, (lambda x=x, O=O: F.adaptive_avg_pool2d(x, (O[0], O[1]))), shape)
            else:
                O = getlist(dims, "O", 2, [geti(dims, "OH", default=4), geti(dims, "OW", default=4)])
                bench(k, (lambda x=x, O=O: F.adaptive_max_pool2d(x, (O[0], O[1]))), shape)
        elif cat in ("avgpool2d_backward", "adaptavg2d_backward",
                     "adaptmax2d_backward"):
            B = geti(dims, "B", default=1); C = geti(dims, "C", default=2)
            I0 = geti(dims, "I0", default=6); I1 = geti(dims, "I1", default=7)
            O0 = geti(dims, "O0", default=I0 // 2)
            O1 = geti(dims, "O1", default=I1 // 2)
            x = torch.rand(B, C, I0, I1, device=d)
            grad = torch.rand(B, C, O0, O1, device=d)
            if cat == "avgpool2d_backward":
                bench(k, (lambda grad=grad, x=x:
                          torch.ops.aten.avg_pool2d_backward.default(
                              grad, x, [2, 2], [2, 2], [0, 0], False,
                              True, None)), shape)
            elif cat == "adaptavg2d_backward":
                bench(k, (lambda grad=grad, x=x:
                          torch.ops.aten._adaptive_avg_pool2d_backward.default(
                              grad, x)), shape)
            else:
                _, indices = F.adaptive_max_pool2d(
                    x, (O0, O1), return_indices=True)
                bench(k, (lambda grad=grad, x=x, indices=indices:
                          torch.ops.aten.adaptive_max_pool2d_backward.default(
                              grad, x, indices)), shape)
        elif cat in ("maxpool3d", "avgpool3d", "adaptavg3d"):
            B = geti(dims, "B", default=2); C = geti(dims, "C", default=3)
            DHW = getlist(dims, "I", 3, [8, 8, 8])
            x = torch.rand(B, C, DHW[0], DHW[1], DHW[2], device=d)
            if cat == "maxpool3d":
                bench(k, (lambda x=x: F.max_pool3d(x, 2)), shape)
            elif cat == "avgpool3d":
                bench(k, (lambda x=x: F.avg_pool3d(x, 2)), shape)
            else:
                O = getlist(dims, "O", 3, [4, 4, 4])
                bench(k, (lambda x=x, O=O: F.adaptive_avg_pool3d(x, (O[0], O[1], O[2]))), shape)
        elif cat in ("avgpool3d_backward", "adaptavg3d_backward"):
            B = geti(dims, "B", default=1); C = geti(dims, "C", default=2)
            I0 = geti(dims, "I0", default=6); I1 = geti(dims, "I1", default=7)
            I2 = geti(dims, "I2", default=8)
            O0 = geti(dims, "O0", default=I0 // 2)
            O1 = geti(dims, "O1", default=I1 // 2)
            O2 = geti(dims, "O2", default=I2 // 2)
            x = torch.rand(B, C, I0, I1, I2, device=d)
            grad = torch.rand(B, C, O0, O1, O2, device=d)
            if cat == "avgpool3d_backward":
                bench(k, (lambda grad=grad, x=x:
                          torch.ops.aten.avg_pool3d_backward.default(
                              grad, x, [2, 2, 2], [2, 2, 2], [0, 0, 0],
                              False, True, None)), shape)
            else:
                bench(k, (lambda grad=grad, x=x:
                          torch.ops.aten._adaptive_avg_pool3d_backward.default(
                              grad, x)), shape)
        elif cat in ("fractional_maxpool2d", "fractional_maxpool3d"):
            B = geti(dims, "B", default=1); C = geti(dims, "C", default=2)
            if cat == "fractional_maxpool2d":
                IH = geti(dims, "IH", default=9); IW = geti(dims, "IW", default=10)
                OH = geti(dims, "OH", default=4); OW = geti(dims, "OW", default=5)
                OH = min(OH, IH - 3 + 1); OW = min(OW, IW - 3 + 1)
                x = torch.rand(B, C, IH, IW, device=d)
                samples = torch.rand(B, C, 2, device=d)
                bench(k, (lambda x=x, samples=samples, OH=OH, OW=OW:
                          F.fractional_max_pool2d(
                              x, (3, 3), output_size=(OH, OW),
                              return_indices=True, _random_samples=samples)), shape)
            else:
                ID = geti(dims, "ID", default=8); IH = geti(dims, "IH", default=9)
                IW = geti(dims, "IW", default=10); OD = geti(dims, "OD", default=3)
                OH = geti(dims, "OH", default=4); OW = geti(dims, "OW", default=5)
                # PyTorch's 3-D implementation requires strict temporal room
                # beyond the pooling window in this validation configuration.
                OD = min(OD, ID - 2); OH = min(OH, IH - 3)
                OW = min(OW, IW - 3)
                x = torch.rand(B, C, ID, IH, IW, device=d)
                samples = torch.rand(B, C, 3, device=d)
                bench(k, (lambda x=x, samples=samples, OD=OD, OH=OH, OW=OW:
                          F.fractional_max_pool3d(
                              x, (2, 3, 3), output_size=(OD, OH, OW),
                              return_indices=True, _random_samples=samples)), shape)
        elif cat == "batchnorm":
            B = geti(dims, "B", "N", default=32); C = geti(dims, "C", default=64)
            sp2 = geti(dims, "spatial", default=0)
            H = sp2 or geti(dims, "H", default=64); W = sp2 or geti(dims, "W", default=64)
            x = torch.rand(B, C, H, W, device=d)
            rm = torch.zeros(C, device=d); rv = torch.ones(C, device=d)
            bench(k, (lambda x=x, rm=rm, rv=rv: F.batch_norm(x, rm, rv, training=False)), shape)
        elif cat in ("layernorm", "rmsnorm", "softmax"):
            rowsn = geti(dims, "R", "B", default=1)
            cols = geti(dims, "K", "D", "N", default=n)
            x = torch.rand(rowsn, cols, device=d)
            if cat == "layernorm":
                weight = torch.rand(cols, device=d)
                bias = torch.rand(cols, device=d)
                bench(k, (lambda x=x, weight=weight, bias=bias, cols=cols:
                          F.layer_norm(x, (cols,), weight, bias, 1e-5)), shape)
            elif cat == "rmsnorm":
                fn = getattr(F, "rms_norm", None)
                weight = torch.rand(cols, device=d)
                bench(k, (lambda x=x, weight=weight, fn=fn, cols=cols:
                          fn(x, (cols,), weight, 1e-5)
                          if fn else torch.sum(x)), shape)
            else:
                bench(k, (lambda x=x: torch.softmax(x, -1)), shape)
        elif cat == "layernorm_native":
            B = geti(dims, "B", default=1024); D = geti(dims, "D", default=4095)
            x = torch.rand(B, D, device=d); weight = torch.rand(D, device=d)
            bias = torch.rand(D, device=d)
            bench(k, (lambda x=x, weight=weight, bias=bias, D=D:
                      torch.ops.aten.native_layer_norm.default(
                          x, [D], weight, bias, 1e-5)), shape)
        elif cat == "layernorm_backward":
            B = geti(dims, "B", default=1024); D = geti(dims, "D", default=4095)
            x = torch.rand(B, D, device=d); grad = torch.rand(B, D, device=d)
            weight = torch.rand(D, device=d); bias = torch.rand(D, device=d)
            _, mean, rstd = torch.ops.aten.native_layer_norm.default(
                x, [D], weight, bias, 1e-5)
            bench(k, (lambda grad=grad, x=x, mean=mean, rstd=rstd,
                      weight=weight, bias=bias, D=D:
                      torch.ops.aten.native_layer_norm_backward.default(
                          grad, x, [D], mean, rstd, weight, bias,
                          [True, True, True])), shape)
        elif cat == "softmax_backward":
            R = geti(dims, "R", "B", default=65536)
            K = geti(dims, "K", "N", default=64)
            output = torch.softmax(torch.rand(R, K, device=d), -1)
            grad = torch.rand(R, K, device=d)
            bench(k, (lambda grad=grad, output=output:
                      torch.ops.aten._softmax_backward_data.default(
                          grad, output, -1, output.dtype)), shape)
        elif cat == "sparse_softmax_backward":
            R = geti(dims, "R", default=64); K = geti(dims, "K", default=8)
            row = torch.arange(R, device=d, dtype=torch.int64).repeat_interleave(K)
            col = torch.arange(K, device=d, dtype=torch.int64).repeat(R)
            indices = torch.stack((row, col))
            source = torch.sparse_coo_tensor(
                indices, torch.rand(R * K, device=d), (R, K), device=d).coalesce()
            output = torch.sparse.softmax(source, -1)
            grad = torch.sparse_coo_tensor(
                indices, torch.rand(R * K, device=d), (R, K), device=d).coalesce()
            bench(k, (lambda grad=grad, output=output, source=source:
                      torch.ops.aten._sparse_softmax_backward_data.default(
                          grad, output, -1, source)), shape)
        elif cat == "sum_backward":
            B = geti(dims, "B", default=724); N = geti(dims, "N", default=5793)
            grad = torch.rand(B, device=d)
            # The C fixture materializes the broadcast into dense output;
            # clone prevents PyTorch's zero-stride view from timing as a no-op.
            bench(k, (lambda grad=grad, N=N:
                      grad[:, None].expand(-1, N).clone()), shape)
        elif cat == "cat":
            B = geti(dims, "B", "R", default=4096); N = geti(dims, "N", "M", default=4096)
            a = torch.rand(B, N, device=d); b = torch.rand(B, N, device=d)
            bench(k, (lambda a=a, b=b: torch.cat([a, b], 0)), shape)
        elif cat == "cat_serial":
            R = geti(dims, "R", default=16); M = geti(dims, "M", default=12)
            K = geti(dims, "K", default=64)
            a = torch.rand(R, K, device=d); b = torch.rand(M, K, device=d)
            bench(k, (lambda a=a, b=b: torch.cat((a, b), 0)), shape)
        elif cat == "cat_sparse_values":
            B = geti(dims, "B", default=4); N = geti(dims, "N", default=256)
            chunks = [torch.rand(N, device=d) for _ in range(B)]
            bench(k, (lambda chunks=chunks: torch.cat(chunks, 0)), shape)
        elif cat == "split_copy":
            N = geti(dims, "N", default=n); S = geti(dims, "S", default=4)
            x = torch.rand(N, device=d)
            split_copy = getattr(torch, "split_copy", None)
            if split_copy:
                bench(k, (lambda x=x, S=S, N=N, split_copy=split_copy:
                          split_copy(x, N // S)), shape)
            else:
                bench(k, (lambda x=x, S=S, N=N:
                          tuple(part.clone() for part in torch.split(
                              x, N // S))), shape)
        elif cat == "dense_sparse_add":
            R = geti(dims, "R", default=2047)
            C = geti(dims, "C", default=2047)
            N = geti(dims, "N", default=16380)
            dense = torch.rand(R, C, device=d)
            linear = torch.arange(N, device=d, dtype=torch.int64)
            row = torch.div(linear, C, rounding_mode="floor") % R
            col = linear % C
            values = torch.rand(N, device=d)
            sparse = torch.sparse_coo_tensor(
                torch.stack((row, col)), values, (R, C), device=d).coalesce()
            bench(k, (lambda dense=dense, sparse=sparse: dense + sparse), shape)
        elif cat == "sparse_csr_mm":
            R = geti(dims, "R", default=65536)
            K = geti(dims, "K", default=65536)
            C = geti(dims, "C", default=64)
            N = geti(dims, "N", default=4194304)
            per_row = N // R
            crow = torch.arange(0, N + 1, per_row, device=d,
                                dtype=torch.int64)
            cols = torch.arange(N, device=d, dtype=torch.int64) % K
            values = torch.rand(N, device=d)
            sparse = torch.sparse_csr_tensor(crow, cols, values,
                                             size=(R, K), device=d)
            rhs = torch.rand(K, C, device=d)
            bench(k, (lambda sparse=sparse, rhs=rhs:
                      torch.sparse.mm(sparse, rhs)), shape)
        elif cat == "sparse_coo_mm":
            R = geti(dims, "R", default=64); C = geti(dims, "C", default=48)
            N = geti(dims, "N", default=512); K = R
            linear = torch.arange(N, device=d, dtype=torch.int64)
            row = torch.div(linear, max(1, N // R), rounding_mode="floor") % R
            col = linear % K
            values = torch.rand(N, device=d)
            sparse = torch.sparse_coo_tensor(
                torch.stack((row, col)), values, (R, K), device=d).coalesce()
            rhs = torch.rand(K, C, device=d)
            bench(k, (lambda sparse=sparse, rhs=rhs:
                      torch.sparse.mm(sparse, rhs)), shape)
        elif cat == "sparse_sampled_addmm":
            R = geti(dims, "R", default=4096); K = geti(dims, "K", default=512)
            C = geti(dims, "C", default=4096)
            NNZ = geti(dims, "NNZ", "N", default=262144)
            per_row = NNZ // R
            crow = torch.arange(0, NNZ + 1, per_row, device=d,
                                dtype=torch.int64)
            cols = torch.arange(NNZ, device=d, dtype=torch.int64) % C
            values = torch.rand(NNZ, device=d)
            mask = torch.sparse_csr_tensor(crow, cols, values,
                                           size=(R, C), device=d)
            lhs = torch.rand(R, K, device=d); rhs = torch.rand(K, C, device=d)
            bench(k, (lambda mask=mask, lhs=lhs, rhs=rhs:
                      torch.sparse.sampled_addmm(
                          mask, lhs, rhs, beta=-0.25, alpha=0.75)), shape)
        elif cat == "weight_norm_backward":
            C = geti(dims, "C", default=1024); S = geti(dims, "S", default=4095)
            grad = torch.rand(C, S, device=d); v = torch.rand(C, S, device=d)
            g = torch.rand(C, 1, device=d)
            norms = torch.linalg.vector_norm(v, dim=1, keepdim=True)
            bench(k, (lambda grad=grad, v=v, g=g, norms=norms:
                      torch.ops.aten._weight_norm_interface_backward.default(
                          grad, v, g, norms, 0)), shape)
        elif cat == "elu_backward":
            grad = torch.rand(n, device=d); output = F.elu(torch.rand(n, device=d))
            bench(k, (lambda grad=grad, output=output:
                      torch.ops.aten.elu_backward.default(
                          grad, 1.25, 0.75, 1.0, True, output)), shape)
        elif cat in ("hardswish_backward", "mish_backward",
                     "sigmoid_backward", "silu_backward", "tanh_backward"):
            grad = torch.rand(n, device=d); x = torch.rand(n, device=d)
            if cat == "hardswish_backward":
                fn = lambda: torch.ops.aten.hardswish_backward.default(grad, x)
            elif cat == "mish_backward":
                fn = lambda: torch.ops.aten.mish_backward.default(grad, x)
            elif cat == "sigmoid_backward":
                output = torch.sigmoid(x)
                fn = lambda: torch.ops.aten.sigmoid_backward.default(grad, output)
            elif cat == "silu_backward":
                fn = lambda: torch.ops.aten.silu_backward.default(grad, x)
            else:
                output = torch.tanh(x)
                fn = lambda: torch.ops.aten.tanh_backward.default(grad, output)
            bench(k, fn, shape)
        elif cat == "native_fixture":
            bench_native_fixture(k, op, dims, n, shape,
                                 sp.get("scalar_args", {}))
        else:
            metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
            emit(f"kernel={k} {metric}=SKIP err=unknown_cat:{cat} "
                 f"{recipe_suffix(k)} shape='{shape}'")
    except Exception as ex:
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={k} {metric}=SKIP err=build:{type(ex).__name__}:{ex} "
             f"{recipe_suffix(k)} shape='{shape}'")
emit("DONE")
if log_stream:
    log_stream.close()
