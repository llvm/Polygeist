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


def bench(name, fn, shape, warm=5, it=20):
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
        else:
            metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
            emit(f"kernel={k} {metric}=SKIP err=unknown_cat:{cat} shape='{shape}'")
    except Exception as ex:
        metric = "torch_resident_us" if d == "cuda" else "torch_cpu_us"
        emit(f"kernel={k} {metric}=SKIP err=build:{type(ex).__name__}:{ex} shape='{shape}'")
emit("DONE")
if log_stream:
    log_stream.close()
