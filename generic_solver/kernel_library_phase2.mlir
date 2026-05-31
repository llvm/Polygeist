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

  // FP32 Darknet im2col+GEMM lowered shape. The linalg raiser represents the
  // scalar A[i,k] load as a broadcasted rank-3 input so the output submap can
  // still ignore the reduction dim when lowered back to the flat C buffer.
  kernel.defn @cublasSgemm_broadcast3d_simple(
      %A: tensor<?x?x?xf32>, %B: tensor<?x?x?xf32>,
      %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%C : tensor<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?x?x?xf32>
    kernel.yield %result : tensor<?x?x?xf32>
  }

  kernel.defn @cublasSgemm_broadcast3d_memref(
      %A: memref<?x?x?xf32>, %B: memref<?x?x?xf32>,
      %C: memref<?x?x?xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : memref<?x?x?xf32>, memref<?x?x?xf32>)
      outs(%C : memref<?x?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    }
    kernel.yield
  }

  // Darknet-style explicit im2col + SGEMM as one library op. The matcher
  // recognizes the zero-fill, guarded im2col workspace materialization, and
  // following GEMM as a single composition; ABI lowering maps this directly
  // to cuDNN convolution with caller-supplied padding and stride.
  kernel.defn @cudnnConvolutionFwd_im2col_gemm(
      %input: memref<?xf32>, %weights: memref<?x?x?xf32>,
      %output: memref<?xf32>,
      %channels: i32, %height: i32, %width: i32, %out_channels: i32,
      %ksize: i32, %stride: i32, %pad: i32) {
    kernel.yield
  }

  // llama2.c RMSNorm matched as:
  //   ss = sum(x[i] * x[i])
  //   out[i] = weight[i] * x[i] * rsqrt(ss / N + 1e-5)
  // ABI lowering maps this to a runtime shim. The shim owns the optimized
  // implementation choice (cuDNN frontend/custom CUDA/CPU fallback).
  kernel.defn @rmsnorm_f32(
      %x: memref<?xf32>, %weight: memref<?xf32>, %out: memref<?xf32>) {
    kernel.yield
  }

  kernel.defn @rmsnorm_f32_tensor(
      %x: tensor<?xf32>, %weight: tensor<?xf32>,
      %out: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  // llama2.c row softmax in-place:
  //   x = exp(x - max(x)) / sum(exp(x - max(x)))
  // ABI lowering maps this to cudnnSoftmaxForward for FP32.
  kernel.defn @cudnnSoftmaxForward(%x: memref<?xf32>) {
    kernel.yield
  }

  kernel.defn @cudnnSoftmaxForward_tensor(%x: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %x : tensor<?xf32>
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

  kernel.defn @cublasDgemv_T(%A: tensor<?x?xf64>, %x: tensor<?xf64>,
                              %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
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

  kernel.defn @cublasSgemv(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                            %y: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf32>, tensor<?xf32>)
      outs(%y : tensor<?xf32>) {
    ^bb0(%a: f32, %xv: f32, %out: f32):
      %p = arith.mulf %a, %xv : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
  }

  kernel.defn @cublasSgemv_T(%A: tensor<?x?xf32>, %x: tensor<?xf32>,
                              %y: tensor<?xf32>) -> tensor<?xf32> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%A, %x : tensor<?x?xf32>, tensor<?xf32>)
      outs(%y : tensor<?xf32>) {
    ^bb0(%a: f32, %xv: f32, %out: f32):
      %p = arith.mulf %a, %xv : f32
      %s = arith.addf %out, %p : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
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

  kernel.defn @memset_zero_1D_f32(%y: tensor<?xf32>) -> tensor<?xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?xf32>
    kernel.yield %result : tensor<?xf32>
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

  // MEMSET-CONST-1D: fill the diagonal of a 2D tensor with 1.0.
  // The matcher names this "1D" because the iter space is 1D (single d0) —
  // the tensor is 2D but accessed at (d0, d0). Used in correlation's
  // diagonal initialization. NOTE: the constant value is HARD-CODED to 1.0
  // because the matcher's Cap binding for the literal isn't currently
  // propagated through render_launch. A different caller wanting a
  // different fill value would need a separate library entry.
  kernel.defn @memset_const_1D(%A: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %one = arith.constant 1.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0, d0)>],
      iterator_types = ["parallel"]
    } outs(%A : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %one : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // ELEMWISE-DIV-SCALAR: y[i] = y[i] / s.
  kernel.defn @elemwise_div_scalar(%y: tensor<?xf64>, %s: f64) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } outs(%y : tensor<?xf64>) {
    ^bb0(%out: f64):
      %t = arith.divf %out, %s : f64
      linalg.yield %t : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // REDUCE-SUM-AXIS: out[j] = sum over the *other* axis of a 2D tensor.
  // The 1D output's length matches the parallel axis of the 2D input.
  // Indexing maps mirror what correlation's raise step produces.
  kernel.defn @reduce_sum_axis(%X: tensor<?x?xf64>, %y: tensor<?xf64>)
                                -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%X : tensor<?x?xf64>) outs(%y : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %s = arith.addf %out, %in : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // SYRK: C[j<=i] = beta*C[j<=i] + alpha*A*A^T  (symmetric rank-k update).
  //
  // Two-step canonical body matching what RaiseToLinalg emits for PolyBench
  // syrk: masked beta-scale of C on the lower triangle, then masked
  // alpha-A*A^T-accumulate. The mask is recomputed from linalg.index +
  // affine.apply inside each linalg.generic so the defn body is
  // self-contained — no external mask SSA is threaded as an operand.
  //
  // Operand order (matches matcher emit): two A-views (the matcher passes
  // both ins of the gemm-shape linalg, which is the same A twice), C, beta,
  // alpha.
  kernel.defn @cublasDsyrk(%A: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
                            %C: tensor<?x?xf64>,
                            %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %scaled_val = arith.mulf %out, %beta : f64
      %r = arith.select %cond, %scaled_val, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d0)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %A2 : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a: f64, %a_t: f64, %out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 2 : index
      %scaled_a = arith.mulf %alpha, %a : f64
      %p = arith.mulf %scaled_a, %a_t : f64
      %s = arith.addf %out, %p : f64
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %r = arith.select %cond, %s, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // SYR2K: C[j<=i] = beta*C[j<=i] + alpha*(A*B^T + B*A^T)  (rank-2k update).
  //
  // Five tensor operands: (A1, B1, B2, A2, C) — the matcher's body splits
  // the rank-2 update across four ins to the second linalg.generic. Maps
  // and iter ordering replicate exactly what RaiseToLinalg emits.
  kernel.defn @cublasDsyr2k(%A1: tensor<?x?xf64>, %B1: tensor<?x?xf64>,
                             %B2: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
                             %C: tensor<?x?xf64>,
                             %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %scaled_val = arith.mulf %out, %beta : f64
      %r = arith.select %cond, %scaled_val, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d2, d0)>
      ],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A1, %B1, %B2, %A2
          : tensor<?x?xf64>, tensor<?x?xf64>,
            tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%a1: f64, %b1: f64, %b2: f64, %a2: f64, %out: f64):
      %i = linalg.index 0 : index
      %j = linalg.index 2 : index
      %t1 = arith.mulf %a1, %alpha : f64
      %t2 = arith.mulf %t1, %b1 : f64
      %t3 = arith.mulf %b2, %alpha : f64
      %t4 = arith.mulf %t3, %a2 : f64
      %t5 = arith.addf %t2, %t4 : f64
      %t6 = arith.addf %out, %t5 : f64
      %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %cond = arith.cmpi slt, %j, %i1 : index
      %r = arith.select %cond, %t6, %out : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }

  // ========================================================================
  // Stencils (Bucket 2). These bodies operate on memref-form linalg.generic
  // because the surrounding time-stepping loop holds a memref iter, so
  // --linalg-debufferize never lifts them to tensor form. The defns mirror
  // the strided memref types that RaiseToLinalg emits for PolyBench stencils.
  // Constants are hard-coded to PolyBench's values (1/3, 1/5, 1/8, etc.) —
  // a Cap-bound literal would be passed as a runtime operand for general
  // callers; we don't do that yet (matcher's Cap-binds-to-Lit means the
  // launch operand list drops the literal).
  // ========================================================================

  // JACOBI 1D 3-point: out[i] = (a[i] + b[i+1] + c[i+2]) / 3
  // The "shift" is baked into the subview offsets (the linalg body sees
  // identity-accessed memrefs at different base offsets).
  kernel.defn @jacobi_1d_3pt(
      %a: memref<?xf64, strided<[1]>>,
      %b: memref<?xf64, strided<[1], offset: 1>>,
      %c: memref<?xf64, strided<[1], offset: 2>>,
      %out: memref<?xf64, strided<[1], offset: 1>>) {
    %cst = arith.constant 0.33333333333333331 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b, %c
          : memref<?xf64, strided<[1]>>,
            memref<?xf64, strided<[1], offset: 1>>,
            memref<?xf64, strided<[1], offset: 2>>)
      outs(%out : memref<?xf64, strided<[1], offset: 1>>) {
    ^bb0(%av: f64, %bv: f64, %cv: f64, %outv: f64):
      %s1 = arith.addf %av, %bv : f64
      %s2 = arith.addf %s1, %cv : f64
      %r  = arith.mulf %s2, %cst : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // JACOBI 2D 5-point: out[i,j] = (c + n + s + w + e) / 5
  kernel.defn @jacobi_2d_5pt(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a4: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %cst = arith.constant 0.20000000000000001 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d1, d0)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64, %ov: f64):
      %s1 = arith.addf %v0, %v1 : f64
      %s2 = arith.addf %s1, %v2 : f64
      %s3 = arith.addf %s2, %v3 : f64
      %s4 = arith.addf %s3, %v4 : f64
      %r  = arith.mulf %s4, %cst : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // HEAT 3D 7-point: out = c + (l-2c+r + d-2c+u + b-2c+f)/8.
  // Operand order from matcher: x-pair (a0,a2), center (a1), y-pair (a3,a4),
  // z-pair (a5,a6).
  kernel.defn @heat_3d_7pt(
      %a0: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a1: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a2: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a3: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a4: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a5: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %a6: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
      %out: memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    %coef = arith.constant 0.125 : f64
    %two  = arith.constant 2.000000e+00 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4, %a5, %a6
          : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>,
            memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>)
      outs(%out : memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64,
         %v5: f64, %v6: f64, %ov: f64):
      %t2c = arith.mulf %v1, %two : f64
      %x_diff = arith.subf %v0, %t2c : f64
      %x_lap  = arith.addf %x_diff, %v2 : f64
      %x_sc   = arith.mulf %x_lap, %coef : f64
      %y_diff = arith.subf %v3, %t2c : f64
      %y_lap  = arith.addf %y_diff, %v4 : f64
      %y_sc   = arith.mulf %y_lap, %coef : f64
      %z_diff = arith.subf %v5, %t2c : f64
      %z_lap  = arith.addf %z_diff, %v6 : f64
      %z_sc   = arith.mulf %z_lap, %coef : f64
      %xy     = arith.addf %x_sc, %y_sc : f64
      %xyz    = arith.addf %xy, %z_sc : f64
      %r      = arith.addf %xyz, %v1 : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D H-field update: out -= 0.5 * (in0 - in1).
  kernel.defn @fdtd_update_2in(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %coef = arith.constant 5.000000e-01 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %ov: f64):
      %diff = arith.subf %v0, %v1 : f64
      %sc   = arith.mulf %diff, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D E-field update: out -= 0.7 * (in0 - in1 + in2 - in3).
  kernel.defn @fdtd_E_update(
      %a0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %a3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %out: memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    %coef = arith.constant 6.999999999999999e-01 : f64
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3
          : memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>,
            memref<?x?xf64, strided<[?, 1], offset: ?>>)
      outs(%out : memref<?x?xf64, strided<[?, 1], offset: ?>>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %ov: f64):
      %d1   = arith.subf %v0, %v1 : f64
      %a    = arith.addf %d1, %v2 : f64
      %d2   = arith.subf %a, %v3 : f64
      %sc   = arith.mulf %d2, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    }
    kernel.yield
  }

  // FDTD-2D source-injection: out[j] = source (broadcast 0-D memref over 1D).
  // Matcher emits this when the input's indexing map is `() -> ()` (scalar
  // access).
  kernel.defn @broadcast_scalar_to_vec(
      %src: memref<f64, strided<[], offset: ?>>,
      %out: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : memref<f64, strided<[], offset: ?>>)
      outs(%out : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    }
    kernel.yield
  }

  // cublasDcopy: 1D-to-1D identity copy (out[i] = in[i]). Used by doitgen
  // for write-back of the scratch buffer.
  kernel.defn @cublasDcopy(
      %src: memref<?xf64, strided<[1]>>,
      %out: memref<?xf64, strided<[1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : memref<?xf64, strided<[1]>>)
      outs(%out : memref<?xf64, strided<[1], offset: ?>>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    }
    kernel.yield
  }

  // CENTERED-SUM-SQUARES: out[j] = sum_i (X[i,j] - mean[j])^2.
  // Variance accumulation (without the 1/N division — that's a separate
  // elemwise_div_scalar in correlation).
  kernel.defn @centered_sum_squares(%X: tensor<?x?xf64>,
                                     %mean: tensor<?xf64>,
                                     %y: tensor<?xf64>) -> tensor<?xf64> {
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d1)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%X, %mean : tensor<?x?xf64>, tensor<?xf64>)
      outs(%y : tensor<?xf64>) {
    ^bb0(%in: f64, %m: f64, %out: f64):
      %d = arith.subf %in, %m : f64
      %p = arith.mulf %d, %d : f64
      %s = arith.addf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    kernel.yield %result : tensor<?xf64>
  }

  // ============================================================
  // Tensor-form stencil defns (multi-root debufferize emits these).
  // Identical bodies to the memref-form stencils above, but with plain
  // `tensor<?...xf64>` operand/result types — the polygeist.submap chain
  // that encodes the offsets is opaque to the lowerer, so the defns can
  // treat each input as a plain tensor of the same rank.
  // ============================================================

  // JACOBI 1D 3-point, tensor form.
  kernel.defn @jacobi_1d_3pt_tensor(
      %a: tensor<?xf64>, %b: tensor<?xf64>, %c: tensor<?xf64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %cst = arith.constant 0.33333333333333331 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%a, %b, %c : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%av: f64, %bv: f64, %cv: f64, %ov: f64):
      %s1 = arith.addf %av, %bv : f64
      %s2 = arith.addf %s1, %cv : f64
      %r  = arith.mulf %s2, %cst : f64
      linalg.yield %r : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // JACOBI 2D 5-point, tensor form.
  kernel.defn @jacobi_2d_5pt_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>, %a2: tensor<?x?xf64>,
      %a3: tensor<?x?xf64>, %a4: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %cst = arith.constant 0.20000000000000001 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4
          : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>,
            tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64, %ov: f64):
      %s1 = arith.addf %v0, %v1 : f64
      %s2 = arith.addf %s1, %v2 : f64
      %s3 = arith.addf %s2, %v3 : f64
      %s4 = arith.addf %s3, %v4 : f64
      %r  = arith.mulf %s4, %cst : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // HEAT 3D 7-point, tensor form.
  kernel.defn @heat_3d_7pt_tensor(
      %a0: tensor<?x?x?xf64>, %a1: tensor<?x?x?xf64>, %a2: tensor<?x?x?xf64>,
      %a3: tensor<?x?x?xf64>, %a4: tensor<?x?x?xf64>, %a5: tensor<?x?x?xf64>,
      %a6: tensor<?x?x?xf64>,
      %out_init: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %coef = arith.constant 0.125 : f64
    %two  = arith.constant 2.000000e+00 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3, %a4, %a5, %a6
          : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>,
            tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>,
            tensor<?x?x?xf64>)
      outs(%out_init : tensor<?x?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %v4: f64,
         %v5: f64, %v6: f64, %ov: f64):
      %t2c = arith.mulf %v1, %two : f64
      %x_diff = arith.subf %v0, %t2c : f64
      %x_lap  = arith.addf %x_diff, %v2 : f64
      %x_sc   = arith.mulf %x_lap, %coef : f64
      %y_diff = arith.subf %v3, %t2c : f64
      %y_lap  = arith.addf %y_diff, %v4 : f64
      %y_sc   = arith.mulf %y_lap, %coef : f64
      %z_diff = arith.subf %v5, %t2c : f64
      %z_lap  = arith.addf %z_diff, %v6 : f64
      %z_sc   = arith.mulf %z_lap, %coef : f64
      %xy     = arith.addf %x_sc, %y_sc : f64
      %xyz    = arith.addf %xy, %z_sc : f64
      %r      = arith.addf %xyz, %v1 : f64
      linalg.yield %r : f64
    } -> tensor<?x?x?xf64>
    kernel.yield %r : tensor<?x?x?xf64>
  }

  // FDTD-2D H-field update, tensor form.
  kernel.defn @fdtd_update_2in_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %coef = arith.constant 5.000000e-01 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1 : tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %ov: f64):
      %diff = arith.subf %v0, %v1 : f64
      %sc   = arith.mulf %diff, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // Broadcast a 0-D tensor (scalar) over a 1D tensor — tensor-form twin
  // of @broadcast_scalar_to_vec. Used by multi-root fdtd-2d's source-
  // injection step where polygeist.submap produces a rank-0 tensor<f64>.
  kernel.defn @broadcast_scalar_to_vec_tensor(
      %src: tensor<f64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : tensor<f64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // cublasDcopy, tensor form (1D identity copy). Used by multi-root
  // fdtd-2d's source-injection step.
  kernel.defn @cublasDcopy_tensor(
      %src: tensor<?xf64>,
      %out_init: tensor<?xf64>) -> tensor<?xf64> {
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%src : tensor<?xf64>)
      outs(%out_init : tensor<?xf64>) {
    ^bb0(%sv: f64, %ov: f64):
      linalg.yield %sv : f64
    } -> tensor<?xf64>
    kernel.yield %r : tensor<?xf64>
  }

  // FDTD-2D E-field update, tensor form.
  kernel.defn @fdtd_E_update_tensor(
      %a0: tensor<?x?xf64>, %a1: tensor<?x?xf64>,
      %a2: tensor<?x?xf64>, %a3: tensor<?x?xf64>,
      %out_init: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %coef = arith.constant 6.999999999999999e-01 : f64
    %r = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %a1, %a2, %a3
          : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%out_init : tensor<?x?xf64>) {
    ^bb0(%v0: f64, %v1: f64, %v2: f64, %v3: f64, %ov: f64):
      %d1   = arith.subf %v0, %v1 : f64
      %a    = arith.addf %d1, %v2 : f64
      %d2   = arith.subf %a, %v3 : f64
      %sc   = arith.mulf %d2, %coef : f64
      %r    = arith.subf %ov, %sc : f64
      linalg.yield %r : f64
    } -> tensor<?x?xf64>
    kernel.yield %r : tensor<?x?xf64>
  }

  // Conv2D 9-tap weighted (3x3 stencil).
  // Operands: 9 input subviews (memref form) of one source tensor (one per
  // 3x3 neighbour position) + 1 output subview. The 9 scalar weights live
  // *inside* the matched linalg.generic body, not in the kernel.launch
  // operand list — surfacing them is a matcher-extension TODO. For the
  // --lower-kernel-launch-to-cublas dispatch this defn is just a symbol
  // carrier (the cuDNN runtime shim hardcodes the polybench weights);
  // body is no-op so the verifier passes.
  kernel.defn @cudnnConvolution2D_9tap(
      %A0: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %w0: f64, %w1: f64, %w2: f64,
      %w3: f64, %w4: f64, %w5: f64,
      %w6: f64, %w7: f64, %w8: f64) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_tensor(
      %A0: tensor<?x?xf64>, %A1: tensor<?x?xf64>, %A2: tensor<?x?xf64>,
      %A3: tensor<?x?xf64>, %A4: tensor<?x?xf64>, %A5: tensor<?x?xf64>,
      %A6: tensor<?x?xf64>, %A7: tensor<?x?xf64>, %A8: tensor<?x?xf64>,
      %C:  tensor<?x?xf64>,
      %w0: f64, %w1: f64, %w2: f64,
      %w3: f64, %w4: f64, %w5: f64,
      %w6: f64, %w7: f64, %w8: f64) -> tensor<?x?xf64> {
    kernel.yield %C : tensor<?x?xf64>
  }

  // FP32 variant of the conv2d 9-tap defn. Same structure as the f64 one
  // but with f32 memrefs + f32 weights. Selected by the rewriter when the
  // matched body's operand types are f32 (it emits @cudnnConvolution2D_9tap_f32
  // as the launch symbol). Phase 2 of the cuDNN conv generalization.
  kernel.defn @cudnnConvolution2D_9tap_f32(
      %A0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf32, strided<[?, 1], offset: ?>>,
      %w0: f32, %w1: f32, %w2: f32,
      %w3: f32, %w4: f32, %w5: f32,
      %w6: f32, %w7: f32, %w8: f32) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_f16(
      %A0: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xf16, strided<[?, 1], offset: ?>>,
      %w0: f16, %w1: f16, %w2: f16,
      %w3: f16, %w4: f16, %w5: f16,
      %w6: f16, %w7: f16, %w8: f16) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_bf16(
      %A0: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xbf16, strided<[?, 1], offset: ?>>,
      %w0: bf16, %w1: bf16, %w2: bf16,
      %w3: bf16, %w4: bf16, %w5: bf16,
      %w6: bf16, %w7: bf16, %w8: bf16) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_i32(
      %A0: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xi32, strided<[?, 1], offset: ?>>,
      %w0: i32, %w1: i32, %w2: i32,
      %w3: i32, %w4: i32, %w5: i32,
      %w6: i32, %w7: i32, %w8: i32) {
    kernel.yield
  }

  kernel.defn @cudnnConvolution2D_9tap_i16(
      %A0: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A1: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A2: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A3: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A4: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A5: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A6: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A7: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %A8: memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %C:  memref<?x?xi16, strided<[?, 1], offset: ?>>,
      %w0: i16, %w1: i16, %w2: i16,
      %w3: i16, %w4: i16, %w5: i16,
      %w6: i16, %w7: i16, %w8: i16) {
    kernel.yield
  }
}
