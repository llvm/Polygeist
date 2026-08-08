#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multinomial_with_replacement_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x16xf32>, %arg2: memref<?x16xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %false = arith.constant false
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x16xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x16xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %3 = tensor.empty() : tensor<32xf32>
    %4 = tensor.empty(%c8) : tensor<?xf32>
    %5:3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %0) -> (tensor<32xf32>, tensor<?xf32>, tensor<?x16xi32>) {
      %inserted = tensor.insert %cst into %arg5[%arg3] : tensor<?xf32>
      %extracted_slice = tensor.extract_slice %arg4[0] [%c32] [1] : tensor<32xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %inserted[%arg3] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %2[%arg3, 0] [1, %c32] [1, 1] : tensor<?x32xf32> to tensor<?xf32>
      %7:2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_1 : tensor<?xf32>) outs(%extracted_slice, %extracted_slice_0 : tensor<?xf32>, tensor<f32>) {
      ^bb0(%in: f32, %out: f32, %out_3: f32):
        %9 = arith.addf %out_3, %in : f32
        linalg.yield %9, %9 : f32, f32
      } -> (tensor<?xf32>, tensor<f32>)
      %inserted_slice = tensor.insert_slice %7#1 into %inserted[%arg3] [1] [1] : tensor<f32> into tensor<?xf32>
      %inserted_slice_2 = tensor.insert_slice %7#0 into %arg4[0] [%c32] [1] : tensor<?xf32> into tensor<32xf32>
      %extracted = tensor.extract %inserted_slice[%arg3] : tensor<?xf32>
      %8 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %arg6) -> (tensor<?x16xi32>) {
        %extracted_3 = tensor.extract %1[%arg3, %arg7] : tensor<?x16xf32>
        %9 = arith.mulf %extracted_3, %extracted : f32
        %10 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
          %11 = arith.cmpi slt, %arg9, %c31_i32 : i32
          %12 = arith.index_cast %arg9 : i32 to index
          %extracted_5 = tensor.extract %inserted_slice_2[%12] : tensor<32xf32>
          %13 = arith.cmpf olt, %extracted_5, %9 : f32
          %14 = arith.addi %arg9, %c1_i32 : i32
          %15 = arith.select %13, %14, %arg9 : i32
          %16 = arith.select %11, %13, %false : i1
          %17 = arith.select %11, %15, %arg9 : i32
          scf.condition(%16) %17 : i32
        } do {
        ^bb0(%arg9: i32):
          scf.yield %arg9 : i32
        }
        %inserted_4 = tensor.insert %10 into %arg8[%arg3, %arg7] : tensor<?x16xi32>
        affine.yield %inserted_4 : tensor<?x16xi32>
      }
      affine.yield %inserted_slice_2, %inserted_slice, %8 : tensor<32xf32>, tensor<?xf32>, tensor<?x16xi32>
    }
    %6 = bufferization.to_memref %5#2 : memref<?x16xi32>
    memref.copy %6, %arg2 : memref<?x16xi32> to memref<?x16xi32>
    return
  }
}

