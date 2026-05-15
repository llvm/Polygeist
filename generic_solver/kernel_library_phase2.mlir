// Phase-2 kernel library — canonical linalg implementations for each library
// symbol the kernel matcher emits. The --lower-kernel-launch pass loads this
// file (via kernel-library-path=) and substitutes each kernel.defn's body
// in place of its matching kernel.launch op.
//
// Conventions:
//   - All bodies operate on `f64` tensors. The PolyBench corpus is double-only.
//   - Operand order matches what kernel_match_rewrite.py emits:
//     all tensor inputs (in source order) + first generic's outs + scalars.
//   - Each defn's linalg.generic uses *self-contained* indexing_maps and
//     iterator_types; it operates on whatever shape the launch's operands
//     have at the call site, without referring to any caller context.
//
// To add a new library entry: pick a unique kernel.launch signature observed
// in `kernel_match_rewrite.py` output and author a kernel.defn with that
// signature whose body computes the canonical semantics for that library op.

module {

  // GEMM: C = alpha*A*B + beta*C    (standard textbook gemm)
  // Operand order: A, B, C, beta, alpha.
  kernel.defn @cublasDgemm(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                            %C: tensor<?x?xf64>,
                            %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    // Step 1: C = beta * C
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %t = arith.mulf %out, %beta : f64
      linalg.yield %t : f64
    } -> tensor<?x?xf64>
    // Step 2: C = alpha * A * B + C
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMM-SIMPLE: C += A*B (alpha=1, beta=1, accumulate-into-C).
  kernel.defn @cublasDgemm_simple(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                                   %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMM-ALPHA-ONLY: C += alpha*A*B (beta=1, accumulate-into-C, custom alpha).
  kernel.defn @cublasDgemm_alpha_only(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                                       %C: tensor<?x?xf64>,
                                       %alpha: f64) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%C : tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %out: f64):
      %p = arith.mulf %a, %b : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEAM-SCALE-2D: C = alpha * C (elementwise scaling, 2D).
  kernel.defn @cublasDgeam_scale2D(%C: tensor<?x?xf64>, %alpha: f64)
                                  -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %t = arith.mulf %out, %alpha : f64
      linalg.yield %t : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // GEMV (2D matrix x 1D vector): y += A * x.
  // Operand order seen in atax, mvt, gesummv, 3mm.
  kernel.defn @cublasDgemv(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                            %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // GEMV-ALPHA: y += alpha * A * x (gemver pattern).
  kernel.defn @cublasDgemv_alpha(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                                  %y: tensor<?xf64>,
                                  %alpha: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%a: f64, %xv: f64, %out: f64):
      %p = arith.mulf %a, %xv : f64
      %ap = arith.mulf %alpha, %p : f64
      %s = arith.addf %out, %ap : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // GER-RANK2: A += u1*v1^T + u2*v2^T.
  // gemver-style fused rank-2 update.
  kernel.defn @cublasDger_rank2(%u1: tensor<?xf64>, %v1: tensor<?xf64>,
                                 %u2: tensor<?xf64>, %v2: tensor<?xf64>,
                                 %A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%u1, %v1, %u2, %v2
          : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
      outs(%A : tensor<?x?xf64>) {
    ^bb0(%u1v: f64, %v1v: f64, %u2v: f64, %v2v: f64, %out: f64):
      %p1 = arith.mulf %u1v, %v1v : f64
      %p2 = arith.mulf %u2v, %v2v : f64
      %s1 = arith.addf %out, %p1 : f64
      %s2 = arith.addf %s1, %p2 : f64
      linalg.yield %s2 : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // AXPBY: y = a*x + b*y (gesummv pattern).
  kernel.defn @cublasDaxpby(%x: tensor<?xf64>, %y: tensor<?xf64>,
                             %a: f64, %b: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%x : tensor<?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%xv: f64, %out: f64):
      %ax = arith.mulf %a, %xv : f64
      %by = arith.mulf %b, %out : f64
      %s = arith.addf %ax, %by : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // AXPY (alpha=1): y += x.
  kernel.defn @cublasDaxpy_unit(%x: tensor<?xf64>, %y: tensor<?xf64>)
                                -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%x : tensor<?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%xv: f64, %out: f64):
      %s = arith.addf %out, %xv : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // MEMSET-ZERO-1D: y[i] = 0 for all i.
  kernel.defn @memset_zero_1D(%y: tensor<?xf64>) -> tensor<?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // MEMSET-ZERO-2D: A[i,j] = 0 for all i,j.
  kernel.defn @memset_zero_2D(%A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %zero = arith.constant 0.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%A : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }
}
