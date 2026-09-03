#!/usr/bin/env python3
"""Shape-matched ATen-native benchmark. Reads shape_specs.json and measures
each kernel's real torch op at the SAME size as its raised 'large problem'.
Runs on the Jetson via  PYTHONPATH=/tmp/tpy python3 bench_shaped.py
Prints:  kernel=<k> torch_device_us=<v> shape='<s>'   (or SKIP with reason)."""
import json, sys
import torch
import torch.nn.functional as F

d = "cuda"
SPECS = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "shape_specs.json"))

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
        torch.cuda.synchronize()
        best = 1e30
        for _ in range(it):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        print(f"kernel={name} torch_device_us={best*1000:.3f} shape='{shape}'", flush=True)
    except Exception as ex:
        print(f"kernel={name} torch_device_us=SKIP err={type(ex).__name__}:{ex} shape='{shape}'", flush=True)


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
            R = geti(dims, "R", "rows", default=0)
            if R:
                C = geti(dims, "K", "C", "cols", default=64)
                x = torch.rand(R, C, device=d)
                fnmap = {"sum": lambda x: torch.sum(x, 1),
                         "count_nonzero": lambda x: torch.count_nonzero(x, 1),
                         "all": lambda x: torch.all(x > 0.5, 1)}
            else:
                x = torch.rand(n, device=d)
                fnmap = {"sum": torch.sum, "count_nonzero": torch.count_nonzero,
                         "all": lambda x: torch.all(x > 0.5)}
            bench(k, (lambda x=x, f=fnmap[op]: f(x)), shape)
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
        elif cat == "bmm":
            B = geti(dims, "B", default=64); M = geti(dims, "M", default=128)
            N = geti(dims, "N", default=128); K = geti(dims, "K", default=128)
            X = torch.rand(B, M, K, device=d); Y = torch.rand(B, K, N, device=d)
            bench(k, (lambda X=X, Y=Y: torch.bmm(X, Y)), shape)
        elif cat == "conv2d":
            B = geti(dims, "B", default=8); IC = geti(dims, "IC", default=32); OC = geti(dims, "OC", default=64)
            H = geti(dims, "H", default=64); W = geti(dims, "W", default=64); K = geti(dims, "KH", "K", default=3)
            x = torch.rand(B, IC, H, W, device=d); w = torch.rand(OC, IC, K, K, device=d)
            bench(k, (lambda x=x, w=w: F.conv2d(x, w, padding=1)), shape)
        elif cat == "conv3d":
            B = geti(dims, "B", default=1); IC = geti(dims, "IC", "C", default=8); OC = geti(dims, "OC", "O", default=16)
            D = geti(dims, "D", default=48); H = geti(dims, "H", default=48); W = geti(dims, "W", default=48)
            K = geti(dims, "K", default=3)
            x = torch.rand(B, IC, D, H, W, device=d); w = torch.rand(OC, IC, K, K, K, device=d)
            bench(k, (lambda x=x, w=w: F.conv3d(x, w, padding=1)), shape)
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
        elif cat == "batchnorm":
            B = geti(dims, "B", "N", default=32); C = geti(dims, "C", default=64)
            sp2 = geti(dims, "spatial", default=0)
            H = sp2 or geti(dims, "H", default=64); W = sp2 or geti(dims, "W", default=64)
            x = torch.rand(B, C, H, W, device=d)
            rm = torch.zeros(C, device=d); rv = torch.ones(C, device=d)
            bench(k, (lambda x=x, rm=rm, rv=rv: F.batch_norm(x, rm, rv, training=False)), shape)
        elif cat in ("layernorm", "rmsnorm", "softmax"):
            cols = 128
            rowsn = max(1, n // cols)
            x = torch.rand(rowsn, cols, device=d)
            if cat == "layernorm":
                bench(k, (lambda x=x: F.layer_norm(x, (cols,))), shape)
            elif cat == "rmsnorm":
                fn = getattr(F, "rms_norm", None)
                bench(k, (lambda x=x, fn=fn: fn(x, (cols,)) if fn else torch.sum(x)), shape)
            else:
                bench(k, (lambda x=x: torch.softmax(x, -1)), shape)
        elif cat == "cat":
            B = geti(dims, "B", "R", default=4096); N = geti(dims, "N", "M", default=4096)
            a = torch.rand(B, N, device=d); b = torch.rand(B, N, device=d)
            bench(k, (lambda a=a, b=b: torch.cat([a, b], 0)), shape)
        else:
            print(f"kernel={k} torch_device_us=SKIP err=unknown_cat:{cat} shape='{shape}'", flush=True)
    except Exception as ex:
        print(f"kernel={k} torch_device_us=SKIP err=build:{type(ex).__name__}:{ex} shape='{shape}'", flush=True)
print("DONE", flush=True)
