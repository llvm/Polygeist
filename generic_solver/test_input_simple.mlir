// Test input file - contains linalg.generic operations to be matched
// This file does NOT contain kernel.defn_collection - those will be loaded externally

module {
  // Function that performs simple matrix multiplication
  func.func @simple_gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // This linalg.generic should match @simple_gemm_linalg from kernel_library.mlir
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
    return %result : tensor<?x?xf32>
  }

  // Function that computes sum of absolute values
  func.func @compute_asum(%X: tensor<?xf32>) -> tensor<f32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<f32>) -> tensor<f32>
    
    // This linalg.generic should match @asum_linalg from kernel_library.mlir
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%X : tensor<?xf32>) 
      outs(%fill : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %abs_val = math.absf %in : f32
        %result = arith.addf %abs_val, %out : f32
        linalg.yield %result : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }

  // Function that computes dot product
  func.func @compute_dot(%X: tensor<?xf32>, %Y: tensor<?xf32>) -> tensor<f32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<f32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<f32>) -> tensor<f32>
    
    // This linalg.generic should match @dot_linalg from kernel_library.mlir
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%X, %Y : tensor<?xf32>, tensor<?xf32>) 
      outs(%fill : tensor<f32>) {
      ^bb0(%x: f32, %y: f32, %out: f32):
        %product = arith.mulf %x, %y : f32
        %result = arith.addf %product, %out : f32
        linalg.yield %result : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }
} 