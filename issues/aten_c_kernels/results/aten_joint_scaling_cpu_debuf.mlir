#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_joint_scaling_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %cst into %3[] : tensor<f32>
    %4 = tensor.empty() : tensor<f32>
    %inserted_0 = tensor.insert %cst into %4[] : tensor<f32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%1 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf olt, %in, %cst : f32
      %10 = arith.negf %in : f32
      %11 = arith.select %9, %10, %in : f32
      %12 = arith.cmpf ogt, %11, %out : f32
      %13 = arith.select %12, %11, %out : f32
      linalg.yield %13 : f32
    } -> tensor<f32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%0 : tensor<?xf32>) outs(%inserted_0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf olt, %in, %cst : f32
      %10 = arith.negf %in : f32
      %11 = arith.select %9, %10, %in : f32
      %12 = arith.cmpf ogt, %11, %out : f32
      %13 = arith.select %12, %11, %out : f32
      linalg.yield %13 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %5[] : tensor<f32>
    %extracted_1 = tensor.extract %6[] : tensor<f32>
    %7 = arith.mulf %extracted_1, %extracted : f32
    %inserted_2 = tensor.insert %7 into %2[%c0] : tensor<?xf32>
    %8 = bufferization.to_memref %inserted_2 : memref<?xf32>
    memref.copy %8, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

