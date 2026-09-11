// Kernel Library - Reusable kernel definitions
// This file contains a collection of kernel definitions that can be loaded
// by the linalg-to-kernel pass and applied to different MLIR modules.

module {
  // Collection of kernel operation definitions
  kernel.defn_collection {
    
    // Simple GEMM operation definition with linalg.generic representation
    kernel.defn @simple_gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
      // Simple matrix multiplication: C = A * B + C
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
        outs(%C : tensor<?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %result = arith.addf %product, %c : f32
          linalg.yield %result : f32
      } -> tensor<?x?xf32>
      kernel.yield %result : tensor<?x?xf32>
    }

    // Scaled GEMM operation definition with alpha and beta coefficients
    kernel.defn @gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>, %alpha: f32, %beta: f32) -> tensor<?x?xf32> {
      // GEMM with scaling: C = alpha * A * B + beta * C
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
        outs(%C : tensor<?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %scaled = arith.mulf %product, %alpha : f32
          %scaled_c = arith.mulf %c, %beta : f32
          %result = arith.addf %scaled, %scaled_c : f32
          linalg.yield %result : f32
      } -> tensor<?x?xf32>
      kernel.yield %result : tensor<?x?xf32>
    }

    // Alpha-scaled GEMM accumulation (matches the second operation in the user's pattern)
    kernel.defn @alpha_gemm_accumulate(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>, %alpha: f64) -> tensor<?x?xf64> {
      // Matrix multiplication with alpha scaling: C += alpha * A * B
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>) 
        outs(%C : tensor<?x?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %6 = arith.mulf %alpha, %in : f64
          %7 = arith.mulf %6, %in_0 : f64
          %8 = arith.addf %out, %7 : f64
          linalg.yield %8 : f64
      } -> tensor<?x?xf64>
      kernel.yield %result : tensor<?x?xf64>
    }

    // Element-wise beta scaling (matches the first operation in the user's pattern)
    kernel.defn @beta_scale(%C: tensor<?x?xf64>, %beta: f64) -> tensor<?x?xf64> {
      // Element-wise scaling: C = beta * C
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d1, d0)>
        ],
        iterator_types = ["parallel", "parallel"]
      } outs(%C : tensor<?x?xf64>) {
        ^bb0(%out: f64):
          %6 = arith.mulf %out, %beta : f64
          linalg.yield %6 : f64
      } -> tensor<?x?xf64>
      kernel.yield %result : tensor<?x?xf64>
    }

    // Matrix multiplication with alpha scaling (second operation standalone)
    kernel.defn @gemm_alpha_only(%A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>, %alpha: f64) -> tensor<?x?xf64> {
      // Matrix multiplication: C += alpha * A * B 
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d1, d0)>,
          affine_map<(d0, d1, d2) -> (d2, d0)>
        ],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>) 
        outs(%C : tensor<?x?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %out: f64):
          %6 = arith.mulf %alpha, %in : f64
          %7 = arith.mulf %6, %in_0 : f64
          %8 = arith.addf %out, %7 : f64
          linalg.yield %8 : f64
      } -> tensor<?x?xf64>
      kernel.yield %result : tensor<?x?xf64>
    }

    // Sum of absolute values operation (ASUM)
    kernel.defn @asum_linalg(%X: tensor<?xf32>, %init: tensor<f32>) -> tensor<f32> {
      // Sum of absolute values: result = sum_i |x_i|
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%init : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %abs_val = math.absf %in : f32
          %result = arith.addf %abs_val, %out : f32
          linalg.yield %result : f32
      } -> tensor<f32>
      kernel.yield %result : tensor<f32>
    }

    // Vector dot product
    kernel.defn @dot_linalg(%X: tensor<?xf32>, %Y: tensor<?xf32>, %init: tensor<f32>) -> tensor<f32> {
      // Dot product: result = sum_i x_i * y_i
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      } ins(%X, %Y : tensor<?xf32>, tensor<?xf32>) 
        outs(%init : tensor<f32>) {
        ^bb0(%x: f32, %y: f32, %out: f32):
          %product = arith.mulf %x, %y : f32
          %result = arith.addf %product, %out : f32
          linalg.yield %result : f32
      } -> tensor<f32>
      kernel.yield %result : tensor<f32>
    }

    // Index of maximum absolute value operation definition with linalg.generic representation
    kernel.defn @iamax_linalg(%X: tensor<?xf32>, %init: tensor<i32>) -> tensor<i32> {
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,      // Input vector
          affine_map<(i) -> ()>        // Result scalar (index)
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%init : tensor<i32>) {
        ^bb0(%in: f32, %out: i32):
          %idx = linalg.index 0 : index
          %abs_val = math.absf %in : f32
          %curr_max_idx = arith.index_cast %out : i32 to index
          %curr_max = tensor.extract %X[%curr_max_idx] : tensor<?xf32>
          %curr_max_abs = math.absf %curr_max : f32
          %cmp = arith.cmpf ogt, %abs_val, %curr_max_abs : f32
          %new_idx = arith.select %cmp, %idx, %curr_max_idx : index
          %result = arith.index_cast %new_idx : index to i32
          linalg.yield %result : i32
      } -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }

    // General Matrix-Vector Multiply (GEMV)
    kernel.defn @gemv_simple(%A: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>) -> tensor<?xf64> {
      // Simple matrix-vector multiplication: y += A * x
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d1, d0)>,  // Matrix A[d0, d1]
          affine_map<(d0, d1) -> (d0)>,      // Vector x[d1]
          affine_map<(d0, d1) -> (d1)>       // Vector y[d0]
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%A, %x : tensor<?x?xf64>, tensor<?xf64>) 
        outs(%y : tensor<?xf64>) {
        ^bb0(%a: f64, %x_val: f64, %y_val: f64):
          %product = arith.mulf %a, %x_val : f64
          %result = arith.addf %y_val, %product : f64
          linalg.yield %result : f64
      } -> tensor<?xf64>
      kernel.yield %result : tensor<?xf64>
    }

    // Index of minimum absolute value operation definition with linalg.generic representation
    kernel.defn @iamin_linalg(%X: tensor<?xf32>, %init: tensor<i32>) -> tensor<i32> {
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,      // Input vector
          affine_map<(i) -> ()>        // Result scalar (index)
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%init : tensor<i32>) {
        ^bb0(%in: f32, %out: i32):
          %idx = linalg.index 0 : index
          %abs_val = math.absf %in : f32
          %curr_min_idx = arith.index_cast %out : i32 to index
          %curr_min = tensor.extract %X[%curr_min_idx] : tensor<?xf32>
          %curr_min_abs = math.absf %curr_min : f32
          %cmp = arith.cmpf olt, %abs_val, %curr_min_abs : f32
          %new_idx = arith.select %cmp, %idx, %curr_min_idx : index
          %result = arith.index_cast %new_idx : index to i32
          linalg.yield %result : i32
      } -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }
  }
} 