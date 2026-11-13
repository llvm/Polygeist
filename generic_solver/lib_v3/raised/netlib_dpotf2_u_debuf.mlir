#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0, d1)[s0] -> (d0 + s0 + 1, d1)>
#map5 = affine_map<(d0, d1)[s0] -> (s0, d1)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 + s0 + 1, s0)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0)[s0] -> (d0 + s0 + 1, s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dpotf2_u(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant -1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf64>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %1, %c1 : index
    %3 = arith.subi %1, %c1 : index
    %4 = affine.apply #map(%c0)[%1]
    %5 = affine.apply #map(%c0)[%1]
    %6 = affine.for %arg3 = 0 to %1 iter_args(%arg4 = %0) -> (tensor<?x?xf64>) {
      %alloca = memref.alloca() : memref<f64>
      %8 = bufferization.to_tensor %alloca : memref<f64>
      %inserted = tensor.insert %cst_1 into %8[] : tensor<f64>
      %9 = polygeist.submap(%arg4, %arg3, %2) {map = #map1} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %10 = linalg.generic {doc = "", indexing_maps = [#map2, #map3], iterator_types = ["reduction"], library_call = ""} ins(%9 : tensor<?xf64>) outs(%inserted : tensor<f64>) {
      ^bb0(%in: f64, %out: f64):
        %22 = arith.mulf %in, %in : f64
        %23 = arith.addf %out, %22 : f64
        %24 = linalg.index 0 : index
        %25 = arith.cmpi slt, %24, %arg3 : index
        %26 = arith.select %25, %23, %out : f64
        linalg.yield %26 : f64
      } -> tensor<f64>
      %extracted = tensor.extract %10[] : tensor<f64>
      %extracted_2 = tensor.extract %arg4[%arg3, %arg3] : tensor<?x?xf64>
      %11 = arith.subf %extracted_2, %extracted : f64
      %12 = math.sqrt %11 : f64
      %inserted_3 = tensor.insert %12 into %arg4[%arg3, %arg3] : tensor<?x?xf64>
      %13 = polygeist.submap(%inserted_3, %arg3, %4, %3) {map = #map4} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %14 = polygeist.submap(%inserted_3, %arg3, %4, %3) {map = #map5} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %15 = polygeist.submap(%inserted_3, %arg3, %4, %3) {map = #map6} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %16 = linalg.generic {doc = "", indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%13, %14 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%15 : tensor<?x?xf64>) {
      ^bb0(%in: f64, %in_4: f64, %out: f64):
        %22 = arith.mulf %in, %in_4 : f64
        %23 = arith.mulf %22, %cst_0 : f64
        %24 = arith.addf %out, %23 : f64
        %25 = linalg.index 1 : index
        %26 = arith.cmpi slt, %25, %arg3 : index
        %27 = arith.select %26, %24, %out : f64
        %28 = linalg.index 0 : index
        %29 = affine.apply #map(%arg3)[%1]
        %30 = arith.cmpi slt, %28, %29 : index
        %31 = arith.select %30, %27, %out : f64
        linalg.yield %31 : f64
      } -> tensor<?x?xf64>
      %17 = polygeist.submapInverse(%inserted_3, %16, %arg3, %4, %3) {map = #map6} : (tensor<?x?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?x?xf64>
      %18 = arith.divf %cst, %12 : f64
      %19 = polygeist.submap(%17, %arg3, %5) {map = #map8} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
      %20 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel"], library_call = ""} outs(%19 : tensor<?xf64>) {
      ^bb0(%out: f64):
        %22 = arith.mulf %18, %out : f64
        %23 = linalg.index 0 : index
        %24 = affine.apply #map(%arg3)[%1]
        %25 = arith.cmpi slt, %23, %24 : index
        %26 = arith.select %25, %22, %out : f64
        linalg.yield %26 : f64
      } -> tensor<?xf64>
      %21 = polygeist.submapInverse(%17, %20, %arg3, %5) {map = #map8} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
      affine.yield %21 : tensor<?x?xf64>
    }
    %7 = bufferization.to_memref %6 : memref<?x?xf64>
    memref.copy %7, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

