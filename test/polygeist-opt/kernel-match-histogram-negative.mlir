// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // This is a real weighted histogram, but the available implementation uses
  // a project-authored binning functor.  It is therefore a GPU fallback, not
  // a complete reusable CUDA-library implementation.  In particular, do not
  // recover the old operation-count fingerprint as a library match.
  func.func @histogram2d(
      %points: memref<?x2xf32>, %weights: memref<?xf32>,
      %min0: f32, %max0: f32, %min1: f32, %max1: f32,
      %output: memref<?x12xf32>) {
    %zero_i32 = arith.constant 0 : i32
    %bins0_i32 = arith.constant 16 : i32
    %bins1_i32 = arith.constant 12 : i32
    %zero = arith.constant 0.0 : f32
    %bins0_f32 = arith.constant 16.0 : f32
    %bins1_f32 = arith.constant 12.0 : f32
    %bins0 = arith.constant 16 : index
    %bins1 = arith.constant 12 : index
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %output_tensor = bufferization.to_tensor %output : memref<?x12xf32>
    %weight_tensor = bufferization.to_tensor %weights : memref<?xf32>
    %point_tensor = bufferization.to_tensor %points : memref<?x2xf32>
    %slice = tensor.extract_slice %output_tensor[0, 0] [%bins0, %bins1]
        [1, 1] : tensor<?x12xf32> to tensor<?x?xf32>
    %cleared = linalg.generic {
        indexing_maps = [#map], iterator_types = ["parallel", "parallel"]}
        outs(%slice : tensor<?x?xf32>) {
    ^bb0(%old: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?xf32>
    %initialized = tensor.insert_slice %cleared into %output_tensor[0, 0]
        [%bins0, %bins1] [1, 1] : tensor<?x?xf32> into tensor<?x12xf32>
    %range0 = arith.subf %max0, %min0 : f32
    %range1 = arith.subf %max1, %min1 : f32
    %result = affine.for %i = 0 to 4096
        iter_args(%hist = %initialized) -> tensor<?x12xf32> {
      %x = tensor.extract %point_tensor[%i, %i0] : tensor<?x2xf32>
      %x0 = arith.subf %x, %min0 : f32
      %x1 = arith.mulf %x0, %bins0_f32 : f32
      %x2 = arith.divf %x1, %range0 : f32
      %bx = arith.fptosi %x2 : f32 to i32
      %y = tensor.extract %point_tensor[%i, %i1] : tensor<?x2xf32>
      %y0 = arith.subf %y, %min1 : f32
      %y1 = arith.mulf %y0, %bins1_f32 : f32
      %y2 = arith.divf %y1, %range1 : f32
      %by = arith.fptosi %y2 : f32 to i32
      %bx_lo = arith.cmpi sge, %bx, %zero_i32 : i32
      %bx_hi = arith.cmpi slt, %bx, %bins0_i32 : i32
      %by_lo = arith.cmpi sge, %by, %zero_i32 : i32
      %by_hi = arith.cmpi slt, %by, %bins1_i32 : i32
      %in_y = arith.andi %by_lo, %by_hi : i1
      %in_x_hi_y = arith.andi %bx_hi, %in_y : i1
      %inside = arith.andi %bx_lo, %in_x_hi_y : i1
      %next = scf.if %inside -> tensor<?x12xf32> {
        %bx_index = arith.index_cast %bx : i32 to index
        %by_index = arith.index_cast %by : i32 to index
        %weight = tensor.extract %weight_tensor[%i] : tensor<?xf32>
        %old = tensor.extract %hist[%bx_index, %by_index] : tensor<?x12xf32>
        %sum = arith.addf %old, %weight : f32
        %updated = tensor.insert %sum into %hist[%bx_index, %by_index]
            : tensor<?x12xf32>
        scf.yield %updated : tensor<?x12xf32>
      } else {
        scf.yield %hist : tensor<?x12xf32>
      }
      affine.yield %next : tensor<?x12xf32>
    }
    %result_memref = bufferization.to_memref %result : memref<?x12xf32>
    memref.copy %result_memref, %output : memref<?x12xf32> to memref<?x12xf32>
    return
  }
}

// CHECK-LABEL: func.func @histogram2d
// CHECK: affine.for
// CHECK-NOT: kernel.launch @thrustHistogram2D{{Weighted}}_f32_memref
