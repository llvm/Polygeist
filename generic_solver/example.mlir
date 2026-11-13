//RUN: polygeist-opt --linalg-to-kernel="kernel-library-path=%S/kernel_library.mlir" -allow-unregistered-dialect %s
// Example MLIR module demonstrating kernel operations and their linalg.generic representations
module {
    //Func that uses simple gemm
    func.func @simple_gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
            // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i, j, k) -> (i, k)>,  // A(i,k)
          affine_map<(i, j, k) -> (k, j)>,  // B(k,j)
          affine_map<(i, j, k) -> (i, j)>   // C(i,j)
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

    // Function that uses iamin (index of minimum absolute value)
    func.func @find_min_abs_index(%X: tensor<?xf32>, %init: tensor<i32>) -> tensor<i32> {
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
      return %result : tensor<i32>
    }

}
