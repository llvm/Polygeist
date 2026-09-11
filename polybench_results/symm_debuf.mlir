#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0)[s0] -> (d0, s0)>
#map2 = affine_map<(d0)[s0] -> (s0, d0)>
#map3 = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>, %arg6: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x30xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?x20xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x30xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = tensor.empty() : tensor<f64>
    %5 = llvm.mlir.undef : f64
    %inserted = tensor.insert %5 into %4[] : tensor<f64>
    %6 = arith.index_cast %arg0 : i32 to index
    %7:2 = affine.for %arg7 = 0 to %6 iter_args(%arg8 = %inserted, %arg9 = %2) -> (tensor<f64>, tensor<?x30xf64>) {
      %9:2 = affine.for %arg10 = 0 to %3 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<f64>, tensor<?x30xf64>) {
        %inserted_0 = tensor.insert %cst into %arg11[] : tensor<f64>
        %10 = arith.subi %6, %c1 : index
        %11 = polygeist.submap(%inserted_0, %10) {map = #map} : (tensor<f64>, index) -> tensor<?xf64>
        %12 = polygeist.submap(%arg12, %arg10, %10) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?xf64>
        %13 = polygeist.submap(%1, %arg7, %10) {map = #map2} : (tensor<?x20xf64>, index, index) -> tensor<?xf64>
        %14 = polygeist.submap(%1, %arg7, %10) {map = #map2} : (tensor<?x20xf64>, index, index) -> tensor<?xf64>
        %15 = polygeist.submap(%0, %arg7, %arg10, %10) {map = #map3} : (tensor<?x30xf64>, index, index, index) -> tensor<?xf64>
        %16 = polygeist.submap(%0, %arg10, %10) {map = #map1} : (tensor<?x30xf64>, index, index) -> tensor<?xf64>
        %17:2 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["reduction"], library_call = ""} ins(%15, %13, %16, %14 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%12, %11 : tensor<?xf64>, tensor<?xf64>) {
        ^bb0(%in: f64, %in_5: f64, %in_6: f64, %in_7: f64, %out: f64, %out_8: f64):
          %26 = arith.mulf %arg2, %in : f64
          %27 = arith.mulf %26, %in_5 : f64
          %28 = arith.addf %out, %27 : f64
          %29 = arith.mulf %in_6, %in_7 : f64
          %30 = arith.addf %out_8, %29 : f64
          %31 = linalg.index 0 : index
          %32 = arith.cmpi slt, %31, %arg7 : index
          %33 = arith.select %32, %28, %out : f64
          %34 = arith.select %32, %30, %out_8 : f64
          linalg.yield %33, %34 : f64, f64
        } -> (tensor<?xf64>, tensor<?xf64>)
        %18 = polygeist.submapInverse(%arg12, %17#0, %arg10, %10) {map = #map1} : (tensor<?x30xf64>, tensor<?xf64>, index, index) -> tensor<?x30xf64>
        %19 = polygeist.submapInverse(%inserted_0, %17#1, %10) {map = #map} : (tensor<f64>, tensor<?xf64>, index) -> tensor<f64>
        %extracted = tensor.extract %18[%arg7, %arg10] : tensor<?x30xf64>
        %20 = arith.mulf %arg3, %extracted : f64
        %extracted_1 = tensor.extract %0[%arg7, %arg10] : tensor<?x30xf64>
        %21 = arith.mulf %arg2, %extracted_1 : f64
        %extracted_2 = tensor.extract %1[%arg7, %arg7] : tensor<?x20xf64>
        %22 = arith.mulf %21, %extracted_2 : f64
        %23 = arith.addf %20, %22 : f64
        %extracted_3 = tensor.extract %19[] : tensor<f64>
        %24 = arith.mulf %arg2, %extracted_3 : f64
        %25 = arith.addf %23, %24 : f64
        %inserted_4 = tensor.insert %25 into %18[%arg7, %arg10] : tensor<?x30xf64>
        affine.yield %19, %inserted_4 : tensor<f64>, tensor<?x30xf64>
      }
      affine.yield %9#0, %9#1 : tensor<f64>, tensor<?x30xf64>
    }
    %8 = bufferization.to_memref %7#1 : memref<?x30xf64>
    memref.copy %8, %arg4 : memref<?x30xf64> to memref<?x30xf64>
    return
  }
}

