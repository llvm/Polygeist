#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 20 + d0 * 5)>
#map4 = affine_map<(d0, d1)[s0, s1, s2] -> (s0 + s1 * 56 + s2 * 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d_aa_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 6.250000e-01 : f32
    %cst_3 = arith.constant 0.571428597 : f32
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = kernel.launch @memset_zero_1D_f32(%1) : (tensor<?xf32>) -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %6 = arith.index_cast %arg4 : index to i32
        %7 = arith.sitofp %6 : i32 to f32
        %8 = arith.addf %7, %cst_1 : f32
        %9 = arith.mulf %8, %cst_3 : f32
        %10 = arith.subf %9, %cst_1 : f32
        %11 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %12 = arith.index_cast %arg6 : index to i32
          %13 = arith.sitofp %12 : i32 to f32
          %14 = arith.addf %13, %cst_1 : f32
          %15 = arith.mulf %14, %cst_2 : f32
          %16 = arith.subf %15, %cst_1 : f32
          %alloca = memref.alloca() : memref<f32>
          %17 = bufferization.to_tensor %alloca : memref<f32>
          %inserted = tensor.insert %cst into %17[] : tensor<f32>
          %18 = polygeist.submap(%inserted, %c4, %c5) {map = #map1} : (tensor<f32>, index, index) -> tensor<?x?xf32>
          %19 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["reduction", "reduction"], library_call = ""} outs(%18 : tensor<?x?xf32>) {
          ^bb0(%out: f32):
            %25 = linalg.index 0 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %10, %27 : f32
            %29 = arith.cmpf olt, %28, %cst : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_0 : f32
            %33 = arith.subf %cst_0, %31 : f32
            %34 = arith.select %32, %33, %cst : f32
            %35 = linalg.index 1 : index
            %36 = arith.index_cast %35 : index to i32
            %37 = arith.sitofp %36 : i32 to f32
            %38 = arith.subf %16, %37 : f32
            %39 = arith.cmpf olt, %38, %cst : f32
            %40 = arith.negf %38 : f32
            %41 = arith.select %39, %40, %38 : f32
            %42 = arith.cmpf olt, %41, %cst_0 : f32
            %43 = arith.subf %cst_0, %41 : f32
            %44 = arith.select %42, %43, %cst : f32
            %45 = arith.mulf %34, %44 : f32
            %46 = arith.addf %out, %45 : f32
            linalg.yield %46 : f32
          } -> tensor<?x?xf32>
          %20 = polygeist.submapInverse(%inserted, %19, %c4, %c5) {map = #map1} : (tensor<f32>, tensor<?x?xf32>, index, index) -> tensor<f32>
          %extracted = tensor.extract %20[] : tensor<f32>
          %21 = polygeist.submap(%arg7, %arg2, %c4, %c5) {map = #map3} : (tensor<?xf32>, index, index, index) -> tensor<?x?xf32>
          %22 = polygeist.submap(%0, %arg6, %arg2, %arg4, %c4, %c5) {map = #map4} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?xf32>
          %23 = linalg.generic {doc = "", indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%22 : tensor<?x?xf32>) outs(%21 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %25 = linalg.index 0 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %10, %27 : f32
            %29 = arith.cmpf olt, %28, %cst : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_0 : f32
            %33 = arith.subf %cst_0, %31 : f32
            %34 = arith.select %32, %33, %cst : f32
            %35 = linalg.index 1 : index
            %36 = arith.index_cast %35 : index to i32
            %37 = arith.sitofp %36 : i32 to f32
            %38 = arith.subf %16, %37 : f32
            %39 = arith.cmpf olt, %38, %cst : f32
            %40 = arith.negf %38 : f32
            %41 = arith.select %39, %40, %38 : f32
            %42 = arith.cmpf olt, %41, %cst_0 : f32
            %43 = arith.subf %cst_0, %41 : f32
            %44 = arith.select %42, %43, %cst : f32
            %45 = arith.mulf %in, %34 : f32
            %46 = arith.mulf %45, %44 : f32
            %47 = arith.divf %46, %extracted : f32
            %48 = arith.addf %out, %47 : f32
            linalg.yield %48 : f32
          } -> tensor<?x?xf32>
          %24 = polygeist.submapInverse(%arg7, %23, %arg2, %c4, %c5) {map = #map3} : (tensor<?xf32>, tensor<?x?xf32>, index, index, index) -> tensor<?xf32>
          affine.yield %24 : tensor<?xf32>
        }
        affine.yield %11 : tensor<?xf32>
      }
      affine.yield %5 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

