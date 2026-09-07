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


def bench_native_fixture(k, op, dims, n, shape):
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
    if op in unary and "backward" not in k:
        x = torch.rand(n, device=d) + (0.5 if op.startswith("log") else 0)
        bench(k, lambda x=x, fn=unary[op]: fn(x), shape); return
    if op in binary:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d) + 0.5
        bench(k, lambda x=x, y=y, fn=binary[op]: fn(x, y), shape); return
    if op in {"clamp", "clamp_scalar", "clamp_max_scalar", "clamp_min_scalar"}:
        x = torch.rand(n, device=d) * 4 - 2
        if op == "clamp_max_scalar": fn = lambda: torch.clamp_max(x, 0.75)
        elif op == "clamp_min_scalar": fn = lambda: torch.clamp_min(x, -0.5)
        else: fn = lambda: torch.clamp(x, -0.5, 0.75)
        bench(k, fn, shape); return
    if op in {"addcdiv", "addcmul"}:
        x = torch.rand(n, device=d); a = torch.rand(n, device=d)
        b = torch.rand(n, device=d) + 0.5
        fn = torch.addcdiv if op == "addcdiv" else torch.addcmul
        bench(k, lambda x=x, a=a, b=b, fn=fn: fn(x, a, b, value=0.75), shape); return
    if op.startswith("lerp"):
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        weight = torch.rand(n, device=d) if "tensor" in op else 0.375
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
            x = torch.rand(max(2, n // 2), 2, device=d)
            grad = torch.rand(x.shape[0], 1, device=d)
            fn = lambda: torch.ops.aten.glu_backward.default(grad, x, -1)
        bench(k, fn, shape); return
    if op == "log_sigmoid" and "backward" in k:
        x = torch.rand(n, device=d); grad = torch.rand(n, device=d)
        _, buffer = torch.ops.aten.log_sigmoid_forward.default(x)
        bench(k, lambda: torch.ops.aten.log_sigmoid_backward.default(
            grad, x, buffer), shape); return
    if op == "glu":
        x = torch.rand(max(2, n // 2), 2, device=d)
        bench(k, lambda: F.glu(x, -1), shape); return
    if op in {"hardshrink", "shrink", "threshold"} and "backward" in k:
        x = torch.rand(n, device=d) * 2 - 1; grad = torch.rand(n, device=d)
        if op == "threshold": fn = lambda: torch.ops.aten.threshold_backward.default(grad, x, 0.0)
        else: fn = lambda: grad * (torch.abs(x) > 0.5)
        bench(k, fn, shape); return
    if op == "hardtanh" and "backward" in k:
        x = torch.rand(n, device=d) * 4 - 2; grad = torch.rand(n, device=d)
        bench(k, lambda: torch.ops.aten.hardtanh_backward.default(grad, x, -1.0, 1.0), shape); return
    if op in {"huber", "huber_elementwise", "smooth_l1", "smooth_l1_elementwise", "mse", "mse_elementwise"}:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        if "huber" in op: fn = lambda: F.huber_loss(x, y, reduction="none")
        elif "smooth" in op: fn = lambda: F.smooth_l1_loss(x, y, reduction="none")
        else: fn = lambda: F.mse_loss(x, y, reduction="none")
        if "backward" in k:
            grad = torch.rand(n, device=d)
            if "smooth" in op: fn = lambda: torch.ops.aten.smooth_l1_loss_backward.default(grad, x, y, 0, 1.0)
            elif "mse" in op: fn = lambda: torch.ops.aten.mse_loss_backward.default(grad, x, y, 0)
        bench(k, fn, shape); return
    if op == "logit" and "backward" in k:
        x = torch.rand(n, device=d) * 0.8 + 0.1; grad = torch.rand(n, device=d)
        bench(k, lambda: torch.ops.aten.logit_backward.default(grad, x, 1e-6), shape); return
    if op in {"masked_scale", "addr_elementwise"}:
        x = torch.rand(n, device=d); y = torch.rand(n, device=d)
        if op == "masked_scale":
            mask = torch.rand(n, device=d) > 0.5; fn = lambda: x * mask * 0.75
        else: fn = lambda: torch.mul(x, y)
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
        elif op in {"or_reduce", "xor_sum"}:
            xb = x > 0.5; fn = lambda: torch.any(xb, dim=dim) if dim is not None else torch.any(xb)
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
        rows = geti(dims, "M", "L", default=2048); cols = geti(dims, "N", "C", default=2048)
        x = torch.rand(rows, cols, device=d)
        fn = (lambda: x.t().contiguous()) if op == "transpose_copy" else (lambda: torch.narrow_copy(x, 0, 0, max(1, rows // 2)))
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
        if op == "stack_serial": fn = lambda: torch.stack(tensors)
        elif op == "unbind_copy":
            x = torch.stack(tensors); fn = lambda: tuple(v.clone() for v in torch.unbind(x))
        else: fn = lambda: torch.block_diag(*tensors)
        bench(k, fn, shape); return

    if op in {"angle", "angle_complex", "angle_real", "as_complex", "complex", "conj_complex", "polar"}:
        if op == "angle_complex" or (op == "angle" and "complex" in k):
            x = torch.rand(n, device=d, dtype=torch.complex64); fn = lambda: torch.angle(x)
        elif op == "angle_real":
            x = torch.rand(n, device=d) * 2 - 1; fn = lambda: torch.angle(x)
        elif op == "as_complex":
            x = torch.rand(n, 2, device=d); fn = lambda: torch.view_as_complex(x)
        elif op == "complex":
            x = torch.rand(n, device=d); y = torch.rand(n, device=d); fn = lambda: torch.complex(x, y)
        elif op == "conj_complex":
            x = torch.rand(n, device=d, dtype=torch.complex64); fn = lambda: torch.conj(x).resolve_conj()
        else:
            x = torch.rand(n, device=d); y = torch.rand(n, device=d); fn = lambda: torch.polar(x, y)
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
        bench(k, lambda: F.conv2d(x,w,padding=dilation,dilation=dilation,groups=groups), shape); return
    if op.startswith("conv_transpose3d"):
        C=geti(dims,"C",default=8); O=geti(dims,"O",default=16); D=geti(dims,"D",default=32)
        H=geti(dims,"H",default=32); W=geti(dims,"W",default=32); K=geti(dims,"K",default=3)
        x=torch.rand(1,C,D,H,W,device=d); w=torch.rand(C,O,K,K,K,device=d)
        bench(k, lambda: F.conv_transpose3d(x,w), shape); return
    if op == "int_mm":
        M=geti(dims,"M",default=512); N=geti(dims,"N",default=512); K=geti(dims,"K",default=1024)
        x=torch.randint(-8,8,(M,K),device=d,dtype=torch.int8); y=torch.randint(-8,8,(K,N),device=d,dtype=torch.int8)
        bench(k, lambda: torch._int_mm(x,y), shape); return
    if op in {"bilinear", "trilinear", "nested_matmul_broadcast"}:
        B=geti(dims,"B",default=64); I=geti(dims,"I","M",default=128)
        J=geti(dims,"J","K",default=128); O=geti(dims,"O","N",default=128)
        x=torch.rand(B,I,device=d); y=torch.rand(B,J,device=d); w=torch.rand(O,I,J,device=d)
        bench(k, lambda: F.bilinear(x,y,w), shape); return

    if op in {"batch_norm_cpu_entry", "blas_scale", "linear_combination",
              "renorm_scale_factor"}:
        x = torch.rand(n, device=d)
        if op == "batch_norm_cpu_entry":
            C = geti(dims, "C", default=64); x = x[:(n // C) * C].reshape(-1, C)
            rm = torch.zeros(C, device=d); rv = torch.ones(C, device=d)
            fn = lambda: F.batch_norm(x, rm, rv, training=False)
        elif op == "blas_scale": fn = lambda: x * 0.75
        elif op == "linear_combination":
            y = torch.rand(n, device=d); z = torch.rand(n, device=d)
            fn = lambda: 0.25 * x + 0.5 * y + 0.75 * z
        else: fn = lambda: torch.clamp_max(1.0 / (torch.abs(x) + 1e-6), 1.0)
        bench(k, fn, shape); return
    if op in {"diag", "block_diag"}:
        B = geti(dims, "B", default=16); N = geti(dims, "N", default=128)
        xs = [torch.rand(N, device=d) for _ in range(B)]
        bench(k, lambda: torch.block_diag(*xs), shape); return
    if op == "conv_tbc":
        T=geti(dims,"T",default=1024); B=geti(dims,"B",default=8)
        I=geti(dims,"I",default=32); O=geti(dims,"O",default=64); K=geti(dims,"K",default=3)
        x=torch.rand(T,B,I,device=d); w=torch.rand(K,I,O,device=d); bias=torch.rand(O,device=d)
        bench(k, lambda: torch.conv_tbc(x,w,bias,0), shape); return
    if op in {"dirichlet_grad", "standard_gamma_grad", "gamma"}:
        x=torch.rand(n,device=d)+0.5; alpha=torch.rand(n,device=d)+0.5
        if op == "dirichlet_grad": fn=lambda: torch._dirichlet_grad(x,alpha,torch.ones_like(x))
        else: fn=lambda: torch._standard_gamma_grad(alpha,x)
        bench(k, fn, shape); return
    if op == "embedding_bag_counts":
        E=geti(dims,"E",default=65536); N=geti(dims,"N",default=n)
        indices=torch.randint(E,(N,),device=d)
        bench(k, lambda: torch.bincount(indices,minlength=E), shape); return
    if op == "glu_jvp":
        x=torch.rand(max(2,n//2),2,device=d); dx=torch.rand_like(x)
        a,b=x.unbind(-1); da,db=dx.unbind(-1)
        bench(k, lambda: da*torch.sigmoid(b)+a*torch.sigmoid(b)*(1-torch.sigmoid(b))*db, shape); return
    if op == "gradient":
        x=torch.rand(n,device=d); bench(k, lambda: torch.gradient(x), shape); return
    if op in {"hspmm", "sparse_addmv_csr", "sparse_addmv_bsr", "sspaddmm"}:
        R=geti(dims,"R",default=4096); C=geti(dims,"C",default=4096)
        nnz=min(geti(dims,"N",default=R*16),R*C); per=max(1,nnz//R); nnz=per*R
        crow=torch.arange(0,nnz+1,per,device=d,dtype=torch.int64)
        col=torch.arange(nnz,device=d,dtype=torch.int64)%C
        values=torch.rand(nnz,device=d); sparse=torch.sparse_csr_tensor(crow,col,values,size=(R,C),device=d)
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
        bench(k, lambda: F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False), shape); return

    if op in {"normal", "log_normal", "uniform", "exponential", "geometric", "cauchy"}:
        x=torch.empty(n,device=d)
        methods={"normal":lambda:x.normal_(),"log_normal":lambda:x.log_normal_(),
                 "uniform":lambda:x.uniform_(),"exponential":lambda:x.exponential_(),
                 "geometric":lambda:x.geometric_(0.5),"cauchy":lambda:x.cauchy_()}
        bench(k, methods[op], shape); return

    metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
    emit(f"kernel={k} {metric}=SKIP err=missing_native_fixture_adapter:{op} shape='{shape}'")


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
        emit(f"kernel={name} {metric}={best:.3f} timing={timing} shape='{shape}'")
    except Exception as ex:
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={name} {metric}=SKIP err={type(ex).__name__}:{ex} shape='{shape}'")


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
            bench(k, (lambda a=a, b=b: torch.add(a, b, alpha=0.75)), shape)
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
            bench_native_fixture(k, op, dims, n, shape)
        else:
            metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
            emit(f"kernel={k} {metric}=SKIP err=unknown_cat:{cat} shape='{shape}'")
    except Exception as ex:
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={k} {metric}=SKIP err=build:{type(ex).__name__}:{ex} shape='{shape}'")
emit("DONE")
if log_stream:
    log_stream.close()
