#!/usr/bin/env python3
"""Numerically compare every normalized MFEM kernel with its faithful original."""
import ctypes as C
import random
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues" / "mfem_c_kernels"
random.seed(20260721)
P = C.POINTER(C.c_double)

def array(values): return (C.c_double * len(values))(*values)
def rand(n): return array([random.uniform(-1, 1) for _ in range(n)])
def transpose(a, rows, cols): return array([a[i*cols+j] for j in range(cols) for i in range(rows)])
def invoke(lib, name, args, seed):
    out = array(seed); f = getattr(lib, name); f.argtypes = [P] * (len(args)+1); f(*args, out); return out
def compare(lib, original, normalized, args, n, seed=None, tolerance=4e-14):
    initial = [0.0]*n if seed is None else seed
    a = invoke(lib, original, args, initial); b = invoke(lib, normalized, args, initial)
    error = max(abs(a[i]-b[i]) for i in range(n))
    print(f"{original:<40} max_error={error:.3e}")
    if error > tolerance: raise RuntimeError(f"{original} mismatch: {error}")

def main():
    with tempfile.TemporaryDirectory(prefix="mfem-c-kernels-") as tmp:
        library = Path(tmp) / "libmfem_c_kernels.so"
        sources = sorted((CORPUS/"original").glob("*.c")) + sorted((CORPUS/"normalized").glob("*.c"))
        subprocess.run(["clang", "-shared", "-fPIC", "-O0", *map(str,sources), "-o", str(library)], check=True)
        lib = C.CDLL(str(library)); D,Q,E,NE,V = 4,5,3,2,2
        B,G = rand(Q*D),rand(Q*D); Bt,Gt = transpose(B,Q,D),transpose(G,Q,D)
        for d in (2,3):
            compare(lib,f"mfem_interp_value_{d}d",f"mfem_interp_value_{d}d_scratch_sliced",(rand(V*D**d),B),V*Q**d)
            compare(lib,f"mfem_integrate_value_{d}d",f"mfem_integrate_value_{d}d_scratch_sliced",(rand(V*Q**d),B),V*D**d,[random.uniform(-1,1) for _ in range(V*D**d)])
            compare(lib,f"mfem_interp_grad_{d}d",f"mfem_interp_grad_{d}d_stage_sliced",(rand(V*D**d),B,G),V*d*Q**d)
            compare(lib,f"mfem_integrate_grad_{d}d",f"mfem_integrate_grad_{d}d_stage_sliced",(rand(V*d*Q**d),B,G),V*D**d,[random.uniform(-1,1) for _ in range(V*D**d)])
            n=NE*D**d; seed=[random.uniform(-1,1) for _ in range(n)]
            compare(lib,f"mfem_pa_mass_apply_{d}d",f"mfem_pa_mass_apply_{d}d_stage_sliced",(B,Bt,rand(NE*Q**d),rand(n)),n,seed)
            compare(lib,f"mfem_pa_diffusion_apply_{d}d",f"mfem_pa_diffusion_apply_{d}d_stage_sliced",(B,G,Bt,Gt,rand(NE*(3 if d==2 else 6)*Q**d),rand(n)),n,seed)
            compare(lib,f"mfem_pa_convection_apply_{d}d",f"mfem_pa_convection_apply_{d}d_stage_sliced",(B,G,Bt,rand(NE*d*Q**d),rand(n)),n,seed)
        for d,nq in ((2,Q**2),(3,Q**3)):
            n=NE*d*d*nq; qv=[random.uniform(-1,1) for _ in range(n)]; J=[0.0]*n
            for e in range(NE):
                for i in range(d):
                    for j in range(d):
                        for p in range(nq): J[p+nq*(j+d*(i+d*e))]=(1.5 if i==j else 0.0)+random.uniform(-.1,.1)
            la,mu,wt=rand(NE*nq),rand(NE*nq),array([random.uniform(.2,1.2) for _ in range(nq)])
            original=array(qv); f=getattr(lib,f"mfem_elasticity_qpoint_{d}d"); f.argtypes=[P]*5; f(la,mu,array(J),wt,original)
            output=array([0.0]*n); g=getattr(lib,f"mfem_elasticity_qpoint_{d}d_scalarized"); g.argtypes=[P]*6; g(la,mu,array(J),wt,array(qv),output)
            error=max(abs(original[i]-output[i]) for i in range(n)); print(f"mfem_elasticity_qpoint_{d}d{'':<13} max_error={error:.3e}"); assert error < 4e-14
        Bo=rand(Q*E); Bot=transpose(Bo,Q,E); op2=rand(NE*Q**2); x2=rand(NE*2*E*D); seed2=[random.uniform(-1,1) for _ in range(len(x2))]
        for kind in ("curlcurl","divdiv"):
            compare(lib,f"mfem_pa_{kind}_apply_2d",f"mfem_pa_{kind}_apply_2d_stage_sliced",(Bo,Bot,G,Gt,op2,x2),len(x2),seed2)
        xdiv=rand(NE*3*E*E*D); compare(lib,"mfem_pa_divdiv_apply_3d","mfem_pa_divdiv_apply_3d_stage_sliced",(Bo,Bot,G,Gt,rand(NE*Q**3),xdiv),len(xdiv),[random.uniform(-1,1) for _ in range(len(xdiv))])
        Bc=rand(Q*D); Bct=transpose(Bc,Q,D); xcurl=rand(NE*3*E*D*D)
        compare(lib,"mfem_pa_curlcurl_apply_3d","mfem_pa_curlcurl_apply_3d_stage_sliced",(Bo,Bc,Bot,Bct,G,Gt,rand(NE*6*Q**3),xcurl),len(xcurl),[random.uniform(-1,1) for _ in range(len(xcurl))])

if __name__ == "__main__": main()
