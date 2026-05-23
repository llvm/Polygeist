#!/home/arjaiswal/slacker/.venv/bin/python3
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
import re
import subprocess
import urllib.parse
from pathlib import Path

POLYBENCH_TEST_DIR = Path("/home/arjaiswal/Polygeist/tools/cgeist/Test/polybench")
POLYBENCH_UTILS = POLYBENCH_TEST_DIR / "utilities"
MLIR_DIR = Path("/tmp/polybench_new")
MACHSUITE_ROOT = Path("/home/arjaiswal/Polygeist/third_party/MachSuite")
MACHSUITE_MLIR_DIR = Path("/tmp/machsuite_mlir")
NPB_ROOT = Path("/home/arjaiswal/Polygeist/third_party/NPB-polybenchified")
NPB_MLIR_DIR = Path("/tmp/npb_mlir")
POLYBENCHGPU_ROOT = Path("/home/arjaiswal/Polygeist/third_party/polybenchGpu/OpenMP")
POLYBENCHGPU_MLIR_DIR = Path("/tmp/pbgpu_mlir")
LLAMA2C_ROOT = Path("/home/arjaiswal/Polygeist/third_party/llama2.c")
LLAMA2C_MLIR_DIR = Path("/tmp/llama2c_mlir")
LLMC_ROOT = Path("/home/arjaiswal/Polygeist/third_party/llm.c")
LLMC_MLIR_DIR = Path("/tmp/llmc_mlir")
OUTPUT_DIR = Path("/tmp/ir_viewer")
REWRITER = Path("/home/arjaiswal/Polygeist/scripts/correctness/kernel_match_rewrite.py")
PYTHON = "/home/arjaiswal/slacker/.venv/bin/python3"

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

# polybenchGpu OpenMP variant — each kernel is a single .c file holding both
# kernel_<name>() AND main(). cgeist inlines the kernel into main and DCEs the
# standalone definition, so the bake uses --function=* and skips --select-func.
# See bake_polybenchgpu_mlir.sh and the project-polybenchgpu-cgeist-inlining
# memory note.
POLYBENCHGPU_KERNELS: dict[str, tuple[str, str]] = {
    "correlation":     ("datamining/correlation/correlation.c",          "kernel_correlation"),
    "covariance":      ("datamining/covariance/covariance.c",            "kernel_covariance"),
    "2mm":             ("linear-algebra/kernels/2mm/2mm.c",              "kernel_2mm"),
    "3mm":             ("linear-algebra/kernels/3mm/3mm.c",              "kernel_3mm"),
    "atax":            ("linear-algebra/kernels/atax/atax.c",            "kernel_atax"),
    "bicg":            ("linear-algebra/kernels/bicg/bicg.c",            "kernel_bicg"),
    "cholesky":        ("linear-algebra/kernels/cholesky/cholesky.c",    "kernel_cholesky"),
    "doitgen":         ("linear-algebra/kernels/doitgen/doitgen.c",      "kernel_doitgen"),
    "gemm":            ("linear-algebra/kernels/gemm/gemm.c",            "kernel_gemm"),
    "gemver":          ("linear-algebra/kernels/gemver/gemver.c",        "kernel_gemver"),
    "gesummv":         ("linear-algebra/kernels/gesummv/gesummv.c",      "kernel_gesummv"),
    "mvt":             ("linear-algebra/kernels/mvt/mvt.c",              "kernel_mvt"),
    "symm":            ("linear-algebra/kernels/symm/symm.c",            "kernel_symm"),
    "syr2k":           ("linear-algebra/kernels/syr2k/syr2k.c",          "kernel_syr2k"),
    "syrk":            ("linear-algebra/kernels/syrk/syrk.c",            "kernel_syrk"),
    "trisolv":         ("linear-algebra/kernels/trisolv/trisolv.c",      "kernel_trisolv"),
    "trmm":            ("linear-algebra/kernels/trmm/trmm.c",            "kernel_trmm"),
    "durbin":          ("linear-algebra/solvers/durbin/durbin.c",        "kernel_durbin"),
    "dynprog":         ("linear-algebra/solvers/dynprog/dynprog.c",      "kernel_dynprog"),
    "gramschmidt":     ("linear-algebra/solvers/gramschmidt/gramschmidt.c", "kernel_gramschmidt"),
    "lu":              ("linear-algebra/solvers/lu/lu.c",                "kernel_lu"),
    "ludcmp":          ("linear-algebra/solvers/ludcmp/ludcmp.c",        "kernel_ludcmp"),
    "floyd-warshall":  ("medley/floyd-warshall/floyd-warshall.c",        "kernel_floyd_warshall"),
    "reg_detect":      ("medley/reg_detect/reg_detect.c",                "kernel_reg_detect"),
    "adi":             ("stencils/adi/adi.c",                            "kernel_adi"),
    "convolution-2d":  ("stencils/convolution-2d/convolution-2d.c",      "kernel_conv2d"),
    "convolution-3d":  ("stencils/convolution-3d/convolution-3d.c",      "kernel_conv2d"),
    "fdtd-2d":         ("stencils/fdtd-2d/fdtd-2d.c",                    "kernel_fdtd_2d"),
    "fdtd-apml":       ("stencils/fdtd-apml/fdtd-apml.c",                "kernel_fdtd_apml"),
    "jacobi-1d-imper": ("stencils/jacobi-1d-imper/jacobi-1d-imper.c",    "kernel_jacobi_1d_imper"),
    "jacobi-2d-imper": ("stencils/jacobi-2d-imper/jacobi-2d-imper.c",    "kernel_jacobi_2d_imper"),
    "seidel-2d":       ("stencils/seidel-2d/seidel-2d.c",                "kernel_seidel_2d"),
}

# llama2.c hot numeric functions in run.c. All three live in the same file.
LLAMA2C_KERNELS: dict[str, tuple[str, str]] = {
    "rmsnorm":  ("run.c", "rmsnorm"),
    "softmax":  ("run.c", "softmax"),
    "matmul":   ("run.c", "matmul"),
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

# Per-polybenchGpu-kernel parallelism + characterisation notes. Many overlap
# with the PolyBench shapes (same algorithm in a slightly different harness),
# but the polybenchGpu suite adds 3D conv / fdtd-apml / reg_detect / dynprog.
POLYBENCHGPU_NOTES: dict[str, tuple[str, str]] = {
    "correlation":     ("partial parallel",  "mean + stddev reductions parallel; symmetric output, diagonal/off-diagonal phases"),
    "covariance":      ("partial parallel",  "mean-centred outer product; mostly parallel with reduction phases"),
    "2mm":             ("highly parallel",   "two chained gemms, parallel"),
    "3mm":             ("highly parallel",   "three chained gemms, parallel"),
    "atax":            ("highly parallel",   "y = A·x then t = Aᵀ·y, parallel"),
    "bicg":            ("highly parallel",   "s = Aᵀ·p and q = A·r, parallel"),
    "cholesky":        ("serial",            "L·Lᵀ factorization — column-sequential"),
    "doitgen":         ("partial parallel",  "inner contraction parallel; outer r-update has loop-carried scratch"),
    "gemm":            ("highly parallel",   "dense gemm, 3-loop parallel + reduction"),
    "gemver":          ("highly parallel",   "rank-2 update + gemv stages, all parallel"),
    "gesummv":         ("highly parallel",   "two gemvs + axpby, all parallel"),
    "mvt":             ("highly parallel",   "x1 += A·y1; x2 += Aᵀ·y2, parallel"),
    "symm":            ("highly parallel",   "symmetric gemm (lower triangle), parallel"),
    "syr2k":           ("highly parallel",   "symmetric rank-2k update (lower triangle)"),
    "syrk":            ("highly parallel",   "symmetric rank-k update (lower triangle)"),
    "trisolv":         ("serial",            "triangular solve — y[i] depends on y[0..i-1]"),
    "trmm":            ("highly parallel",   "triangular gemm — (i,j) parallel, k reduction"),
    "durbin":          ("serial",            "Levinson-Durbin recurrence — O(N²) scalar carry"),
    "dynprog":         ("serial",            "knapsack-style DP — outer time step + inner table fill have carry"),
    "gramschmidt":     ("serial",            "modified Gram-Schmidt — column k+1 reads column k just written"),
    "lu":              ("serial",            "LU factorization — column-sequential pattern as cholesky"),
    "ludcmp":          ("serial",            "LU + triangular solve — both phases row-by-row carry"),
    "floyd-warshall":  ("partial parallel",  "all-pairs shortest path: (i,j) parallel per k, k loop sequential"),
    "reg_detect":      ("partial parallel",  "regression detection — convolution-style inner loops, sequential outer phases"),
    "adi":             ("parallel + T loop", "alternating direction implicit; T+sweep loops sequential"),
    "convolution-2d":  ("highly parallel",   "single 3x3 stencil pass over a 2D field — fully parallel, no T loop"),
    "convolution-3d":  ("highly parallel",   "single 3x3x3 stencil pass over a 3D field — fully parallel"),
    "fdtd-2d":         ("parallel + T loop", "E/H field cross-updates; T steps sequential, inner parallel"),
    "fdtd-apml":       ("parallel + T loop", "FDTD with anisotropic PML boundary; T steps sequential, inner parallel"),
    "jacobi-1d-imper": ("parallel + T loop", "3-point 1D smoother; T steps sequential, inner parallel"),
    "jacobi-2d-imper": ("parallel + T loop", "5-point 2D stencil; T steps sequential, inner parallel"),
    "seidel-2d":       ("serial",            "Gauss-Seidel — in-place writes within a sweep, current cell reads recently-updated values"),
}

# llama2.c numeric kernels — the building blocks of LLM forward pass.
LLAMA2C_NOTES: dict[str, tuple[str, str]] = {
    "matmul":   ("highly parallel",   "dense gemv (W·x = xout); single linalg.generic after raise"),
    "rmsnorm":  ("highly parallel",   "ss = mean(x²) + eps then o = weight·x/√ss; reduction + parallel scale"),
    "softmax":  ("partial parallel",  "max-shift then exp + sum then divide; three reduction/parallel phases"),
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

# polybenchGpu blockers — most algorithms overlap with PolyBench, but the bake
# pipeline is different (whole-program raise; main scaffolding is intermixed
# with linalg ops), which makes v2 debuf consistently crash. The multi-root
# debuf variant succeeds and is what the IR explorer surfaces.
POLYBENCHGPU_BLOCKERS: dict[str, tuple[str, str]] = {
    "correlation":     ("scratch-carry",     "row-mean + variance accumulation; cross-pass scratch in cov-style outer loops"),
    "covariance":      ("scratch-carry",     "mean-centred outer product; cross-pass scratch state"),
    "2mm":             ("none",              ""),
    "3mm":             ("none",              ""),
    "atax":            ("none",              ""),
    "bicg":            ("none",              ""),
    "cholesky":        ("serial-recurrence", "lower-triangular factorization — column k modifies columns 0..k-1, k+1..N-1 depends on them"),
    "doitgen":         ("matcher-gap",       "per-iter scratch-copy body not in matcher library"),
    "gemm":            ("none",              ""),
    "gemver":          ("none",              ""),
    "gesummv":         ("none",              ""),
    "mvt":             ("none",              ""),
    "symm":            ("matcher-gap",       "lifts; one residual symm-edge body unmatched"),
    "syr2k":           ("none",              ""),
    "syrk":            ("none",              ""),
    "trisolv":         ("serial-recurrence", "triangular solve — y[i] depends on y[0..i-1]"),
    "trmm":            ("matcher-gap",       "lifts cleanly; triangular-edge body unmatched"),
    "durbin":          ("serial-recurrence", "Levinson-Durbin recurrence — alpha/beta scalars carried across outer k"),
    "dynprog":         ("serial-recurrence", "knapsack-style DP — outer time step + table-fill row dependencies"),
    "gramschmidt":     ("serial-recurrence", "column-by-column modified Gram-Schmidt — column k+1 reads what column k wrote"),
    "lu":              ("serial-recurrence", "LU factorization — pivot row k modifies later rows"),
    "ludcmp":          ("serial-recurrence", "LU + triangular solve — both phases have row-by-row carry"),
    "floyd-warshall":  ("cgeist-frontend",   "upstream syntax error (extraneous } at floyd-warshall.c:75) — cgeist fails"),
    "reg_detect":      ("raise-crash",       "polygeist-opt segfaults inside the raise pipeline"),
    "adi":             ("t-loop",            "ADI (alternating direction implicit) — T-step outer, direction sweeps inside"),
    "convolution-2d":  ("matcher-gap",       "single 3x3 conv2d pass; lifts cleanly but matcher has no conv2d-3x3 template"),
    "convolution-3d":  ("matcher-gap",       "single 3x3x3 conv3d pass; lifts cleanly but matcher has no conv3d template"),
    "fdtd-2d":         ("t-loop",            "Yee FDTD E/H field update; T steps serial, per-step body parallel"),
    "fdtd-apml":       ("t-loop",            "FDTD with PML boundary; T steps serial, inner parallel"),
    "jacobi-1d-imper": ("t-loop",            "3-point 1D smoother; T steps serial, inner 1D parallel"),
    "jacobi-2d-imper": ("t-loop",            "5-point 2D smoother; T steps serial, inner 2D parallel"),
    "seidel-2d":       ("serial-recurrence", "Gauss-Seidel — in-place writes within a sweep"),
}

# llama2.c blockers — all three lift to linalg.generic cleanly; the gaps are
# matcher-library entries for LLM-shaped bodies (rmsnorm, softmax) and a
# v2-debufferize limitation on softmax's fused exp+sum tuple yield.
LLAMA2C_BLOCKERS: dict[str, tuple[str, str]] = {
    "matmul":   ("none",           ""),
    "rmsnorm":  ("matcher-gap",    "ss-reduction + parallel weighted-scale; rmsnorm body not in matcher library"),
    "softmax":  ("matcher-gap",    "max-shift / exp+sum / divide pipeline; softmax body not in library, and v2 debuf can't handle the fused tuple-yield generic (multi-root debuf succeeds)"),
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
    "softmax-fwd":              ("matcher-gap",       "per-row softmax with max-shift; same library gap as llama2 softmax; v2 debuf fails on fused exp+sum tuple yield, multi-root succeeds"),
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
    if kset == "polybenchgpu":
        info = POLYBENCHGPU_KERNELS.get(name)
        if not info:
            return None
        relsrc, _fn = info
        p = POLYBENCHGPU_ROOT / relsrc
        return p if p.exists() else None
    if kset == "llama2c":
        info = LLAMA2C_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = LLAMA2C_ROOT / srcname
        return p if p.exists() else None
    if kset == "llmc":
        info = LLMC_KERNELS.get(name)
        if not info:
            return None
        srcname, _fn = info
        p = LLMC_ROOT / srcname
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

    if raised.exists():
        html, css = syntax_highlight(raised.read_text())
        pages["raised"] = html
    if debuf.exists():
        debuf_text = debuf.read_text()
        n_for = count_for_loops(debuf_text)
        html, css = syntax_highlight(debuf_text)
        pages["debuf"] = html
        rewritten, report = run_rewriter(debuf)
        html, css = syntax_highlight(rewritten)
        pages["matched"] = html
    else:
        report = [("launches", 0), ("residual_lg", 0)]
    if debuf_mr.exists():
        debuf_mr_text = debuf_mr.read_text()
        html, css = syntax_highlight(debuf_mr_text)
        pages["debuf_mr"] = html
        # Fallback: if v2 debuf failed but multi-root succeeded (the
        # common pattern for whole-program-raise suites like polybenchGpu),
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
}


def _render_section_rows(kernel_stats: dict[str, dict],
                          notes: dict[str, tuple[str, str]],
                          blockers: dict[str, tuple[str, str]]) -> str:
    rows = []
    for k, s in sorted(kernel_stats.items()):
        l = s["launches"]; r = s["residual"]; f = s["residual_for"]
        if l > 0 and r == 0 and f == 0:
            cls = "pass"; status = "FULL"
        elif l > 0:
            cls = "partial"; status = "PARTIAL"
        else:
            cls = "none"; status = "NONE"
        for_cls = "none" if f > 0 else "pass"

        if s["ce_url"]:
            kernel_link = f'<a class="kernel" href="{s["ce_url"]}" target="_blank">{k}</a>'
        else:
            kernel_link = f'<span class="nope">{k} (no source)</span>'

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
        rows.append(
            f'<tr>'
            f'<td>{kernel_link}'
            f'<a class="viewer" href="{page_file}" style="margin-left:12px">[IR preview]</a>'
            f'</td>'
            f'<td>{l}</td><td>{r}</td><td class="{for_cls}">{f}</td>'
            f'<td class="{cls}">{status}</td>'
            f'{note_cell}'
            f'{block_cell}'
            f'</tr>'
        )
    return "\n".join(rows)


def _build_section(title: str, anchor: str, blurb: str,
                    kernel_stats: dict[str, dict],
                    notes: dict[str, tuple[str, str]],
                    blockers: dict[str, tuple[str, str]]) -> str:
    """Render one benchmark-suite section: a section header, blurb, then table."""
    rows_html = _render_section_rows(kernel_stats, notes, blockers)
    return (
        f'<a name="{anchor}"></a>'
        f'<div class="section-header"><h2 class="section-title">{title}</h2></div>'
        f'<div class="intro">{blurb}</div>'
        '<table><thead><tr>'
        '<th>kernel</th><th>kernel.launches</th>'
        '<th>residual linalg.generic</th>'
        '<th>residual for-loops</th>'
        '<th>match status</th>'
        '<th>parallelism</th>'
        '<th>parallelism notes</th>'
        '<th>blocker</th>'
        '<th>blocker notes</th>'
        '</tr></thead><tbody>'
        + rows_html +
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


def build_index(polybench_stats: dict[str, dict],
                 machsuite_stats: dict[str, dict],
                 npb_stats: dict[str, dict],
                 polybenchgpu_stats: dict[str, dict],
                 llama2c_stats: dict[str, dict],
                 llmc_stats: dict[str, dict]) -> str:
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
    )
    machsuite_section = _build_section(
        title="MachSuite",
        anchor="machsuite",
        blurb=(
            "19 kernels from the MachSuite accelerator-research benchmark — "
            "wider coverage than PolyBench (AES, sorting, FFT bit-reversal, "
            "SpMV, BFS, KMP, MD, Viterbi) at the cost of more kernels that "
            "fall outside the pipeline's affine sweet spot. Kernels marked "
            "<span class=\"nope\">(no source)</span> failed at the cgeist "
            "front-end (typically due to pointer- or bit-heavy C that cgeist "
            "doesn't model)."
        ),
        kernel_stats=machsuite_stats,
        notes=MACHSUITE_NOTES,
        blockers=MACHSUITE_BLOCKERS,
    )
    npb_section = _build_section(
        title="NPB (polybenchified)",
        anchor="npb",
        blurb=(
            "Selected kernels from NPB3.0-omp-C extracted into PolyBench-"
            "style single-file form (third_party/NPB-polybenchified/). The "
            "original NPB is one giant .c per benchmark with module-level "
            "static globals — cgeist can't isolate a single function from "
            "that layout. Each kernel here had its array dependencies "
            "rewritten as parameters so the pipeline can lift it. The "
            "results surface gaps that whole-file NPB didn't expose: "
            "indirect indexing (ft-evolve), scratch-row carries (MG "
            "stencils), and mixed sum+max reductions (norm2u3)."
        ),
        kernel_stats=npb_stats,
        notes=NPB_NOTES,
        blockers=NPB_BLOCKERS,
    )
    polybenchgpu_section = _build_section(
        title="polybenchGpu (OpenMP variant)",
        anchor="polybenchgpu",
        blurb=(
            "32 kernels from sgrauerg/polybenchGpu, OpenMP variant — the "
            "same numerical bodies as PolyBench but in single-file harness "
            "form (kernel + init + main + print_array per .c). cgeist "
            "inlines kernel_<name>() into main() and DCEs the standalone "
            "definition, so the bake uses <code>--function=*</code> and "
            "skips <code>--select-func</code>. The raise pass still finds "
            "the inlined affine loops; the v2 debufferize gets confused by "
            "the main-scaffolding ops (addressof / strcmp / print_array) "
            "intermixed with linalg, so the multi-root debuf is what "
            "appears in the IR preview."
        ),
        kernel_stats=polybenchgpu_stats,
        notes=POLYBENCHGPU_NOTES,
        blockers=POLYBENCHGPU_BLOCKERS,
    )
    llama2c_section = _build_section(
        title="llama2.c (karpathy/llama2.c)",
        anchor="llama2c",
        blurb=(
            "Hot numeric functions from run.c — the building blocks of "
            "the LLM forward pass: matmul (W·x), rmsnorm (mean-square "
            "normalize + scale), softmax (max-shift / exp / sum-normalize). "
            "All three lift to linalg.generic cleanly. The blockers are "
            "matcher-library gaps (no gemv / rmsnorm / softmax templates) "
            "and a v2-debufferize limitation on softmax's fused exp+sum "
            "tuple yield (multi-root debuf succeeds)."
        ),
        kernel_stats=llama2c_stats,
        notes=LLAMA2C_NOTES,
        blockers=LLAMA2C_BLOCKERS,
    )
    llmc_section = _build_section(
        title="llm.c (karpathy/llm.c — GPT-2 in C, forward + backward)",
        anchor="llmc",
        blurb=(
            "15 leaf kernels from train_gpt2.c — the full GPT-2 building "
            "blocks for both inference and training: encoder, layernorm, "
            "matmul, attention, gelu, residual, softmax, crossentropy "
            "(forward + backward where it applies). Direct continuation of "
            "llama2.c — same author, wider coverage. Stresses the pipeline "
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

    body = (
        '<div class="header"><h1>Polygeist IR explorer</h1>'
        '<div style="margin-top:6px; font-size:13px;">'
        '  Jump to: '
        '  <a href="#taxonomy">Algorithm taxonomy</a> &middot; '
        '  <a href="#polybench">PolyBench</a> &middot; '
        '  <a href="#machsuite">MachSuite</a> &middot; '
        '  <a href="#npb">NPB (polybenchified)</a> &middot; '
        '  <a href="#polybenchgpu">polybenchGpu</a> &middot; '
        '  <a href="#llama2c">llama2.c</a> &middot; '
        '  <a href="#llmc">llm.c</a>'
        '</div></div>'
        + _build_taxonomy_panel()
        + polybench_section
        + machsuite_section
        + npb_section
        + polybenchgpu_section
        + llama2c_section
        + llmc_section
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

    # PolyBench set.
    pb_kernels = discover_kernels(MLIR_DIR)
    print(f"Rendering {len(pb_kernels)} PolyBench kernels...", flush=True)
    pb_stats = {}
    for i, k in enumerate(pb_kernels, 1):
        print(f"  [PB {i:2d}/{len(pb_kernels)}] {k}", flush=True)
        pb_stats[k] = build_kernel_page(k, mlir_dir=MLIR_DIR,
                                         kset="polybench", file_prefix="")

    # MachSuite set.
    ms_kernels_from_files = discover_kernels(MACHSUITE_MLIR_DIR)
    # Also include kernels that have NO MLIR (cgeist failed) so they show as
    # "(no source)" entries with the explanatory parallelism note. We still
    # need them in the index to be honest about what the pipeline did/didn't
    # eat. They get an empty stats record below.
    ms_kernels = sorted(set(ms_kernels_from_files) | set(MACHSUITE_KERNELS.keys()))
    print(f"Rendering {len(ms_kernels)} MachSuite kernels...", flush=True)
    ms_stats = {}
    for i, k in enumerate(ms_kernels, 1):
        print(f"  [MS {i:2d}/{len(ms_kernels)}] {k}", flush=True)
        # If the kernel produced no MLIR files at all, fabricate a zero-stat
        # record so it still appears in the index (with no CE link).
        has_any = any((MACHSUITE_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            ms_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                            "ce_url": None, "page_filename": ""}
            continue
        ms_stats[k] = build_kernel_page(
            k, mlir_dir=MACHSUITE_MLIR_DIR, kset="machsuite",
            file_prefix="ms_",
        )

    # NPB-polybenchified set.
    npb_kernels_from_files = discover_kernels(NPB_MLIR_DIR)
    npb_kernels = sorted(set(npb_kernels_from_files) | set(NPB_KERNELS.keys()))
    print(f"Rendering {len(npb_kernels)} NPB kernels...", flush=True)
    npb_stats = {}
    for i, k in enumerate(npb_kernels, 1):
        print(f"  [NPB {i:2d}/{len(npb_kernels)}] {k}", flush=True)
        has_any = any((NPB_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            npb_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                             "ce_url": None, "page_filename": ""}
            continue
        npb_stats[k] = build_kernel_page(
            k, mlir_dir=NPB_MLIR_DIR, kset="npb",
            file_prefix="npb_",
        )

    # polybenchGpu OpenMP set.
    pbgpu_kernels_from_files = discover_kernels(POLYBENCHGPU_MLIR_DIR)
    pbgpu_kernels = sorted(set(pbgpu_kernels_from_files) | set(POLYBENCHGPU_KERNELS.keys()))
    print(f"Rendering {len(pbgpu_kernels)} polybenchGpu kernels...", flush=True)
    pbgpu_stats = {}
    for i, k in enumerate(pbgpu_kernels, 1):
        print(f"  [PBGPU {i:2d}/{len(pbgpu_kernels)}] {k}", flush=True)
        has_any = any((POLYBENCHGPU_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            pbgpu_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                               "ce_url": None, "page_filename": ""}
            continue
        pbgpu_stats[k] = build_kernel_page(
            k, mlir_dir=POLYBENCHGPU_MLIR_DIR, kset="polybenchgpu",
            file_prefix="pbgpu_",
        )

    # llama2.c set.
    llama_kernels_from_files = discover_kernels(LLAMA2C_MLIR_DIR)
    llama_kernels = sorted(set(llama_kernels_from_files) | set(LLAMA2C_KERNELS.keys()))
    print(f"Rendering {len(llama_kernels)} llama2.c kernels...", flush=True)
    llama_stats = {}
    for i, k in enumerate(llama_kernels, 1):
        print(f"  [LLAMA {i:2d}/{len(llama_kernels)}] {k}", flush=True)
        has_any = any((LLAMA2C_MLIR_DIR / f"{k}{suf}").exists()
                      for suf in (".mlir", "_linalg.mlir", "_debuf.mlir",
                                   "_debuf_mr.mlir"))
        if not has_any:
            llama_stats[k] = {"launches": 0, "residual": 0, "residual_for": 0,
                               "ce_url": None, "page_filename": ""}
            continue
        llama_stats[k] = build_kernel_page(
            k, mlir_dir=LLAMA2C_MLIR_DIR, kset="llama2c",
            file_prefix="llama_",
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

    OUTPUT_DIR.joinpath("index.html").write_text(
        build_index(pb_stats, ms_stats, npb_stats, pbgpu_stats, llama_stats, llmc_stats))
    print(f"\nDone. Open {OUTPUT_DIR}/index.html.")


if __name__ == "__main__":
    main()
