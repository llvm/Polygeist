#map = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map3 = affine_map<(d0, d1) -> (d0 + 1, d1 + 2)>
#map4 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map5 = affine_map<(d0, d1) -> (d0, d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e-01 : f64
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %c-1 = arith.constant -1 : index
    %2 = arith.addi %0, %c-1 : index
    %c-1_0 = arith.constant -1 : index
    %3 = arith.addi %0, %c-1_0 : index
    %4 = arith.subi %3, %c1 : index
    %c-1_1 = arith.constant -1 : index
    %5 = arith.addi %0, %c-1_1 : index
    %6 = arith.subi %5, %c1 : index
    %7 = arith.subi %2, %c1 : index
    %c-1_2 = arith.constant -1 : index
    %8 = arith.addi %0, %c-1_2 : index
    %9 = arith.subi %8, %c1 : index
    %c0 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %dim = memref.dim %arg3, %c0_4 : memref<?x?xf64>
    %c1_5 = arith.constant 1 : index
    %dim_6 = memref.dim %arg3, %c1_5 : memref<?x?xf64>
    %alloc = memref.alloc(%dim, %dim_6) {alignment = 64 : i64} : memref<?x?xf64>
    memref.copy %arg3, %alloc : memref<?x?xf64> to memref<?x?xf64>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %arg2, %c0_7 : memref<?x?xf64>
    %c1_9 = arith.constant 1 : index
    %dim_10 = memref.dim %arg2, %c1_9 : memref<?x?xf64>
    %alloc_11 = memref.alloc(%dim_8, %dim_10) {alignment = 64 : i64} : memref<?x?xf64>
    memref.copy %arg2, %alloc_11 : memref<?x?xf64> to memref<?x?xf64>
    %10:2 = scf.for %arg4 = %c0 to %1 step %c1_3 iter_args(%arg5 = %alloc, %arg6 = %alloc_11) -> (memref<?x?xf64>, memref<?x?xf64>) {
      %alloc_12 = memref.alloc(%6, %4) {alignment = 64 : i64} : memref<?x?xf64>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg5 : memref<?x?xf64>) outs(%alloc_12 : memref<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      }
      linalg.generic {doc = "", indexing_maps = [#map, #map2, #map3, #map4, #map5, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%arg6, %arg6, %arg6, %arg6, %arg6 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%alloc_12 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
        %11 = arith.addf %in, %in_15 : f64
        %12 = arith.addf %11, %in_16 : f64
        %13 = arith.addf %12, %in_17 : f64
        %14 = arith.addf %13, %in_18 : f64
        %15 = arith.mulf %14, %cst : f64
        linalg.yield %15 : f64
      }
      %subview = memref.subview %arg5[1, 1] [%6, %4] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
      memref.copy %alloc_12, %subview : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
      %alloc_13 = memref.alloc(%9, %7) {alignment = 64 : i64} : memref<?x?xf64>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : memref<?x?xf64>) outs(%alloc_13 : memref<?x?xf64>) {
      ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
      }
      linalg.generic {doc = "", indexing_maps = [#map, #map2, #map3, #map4, #map5, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%arg5, %arg5, %arg5, %arg5, %arg5 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%alloc_13 : memref<?x?xf64>) {
      ^bb0(%in: f64, %in_15: f64, %in_16: f64, %in_17: f64, %in_18: f64, %out: f64):
        %11 = arith.addf %in, %in_15 : f64
        %12 = arith.addf %11, %in_16 : f64
        %13 = arith.addf %12, %in_17 : f64
        %14 = arith.addf %13, %in_18 : f64
        %15 = arith.mulf %14, %cst : f64
        linalg.yield %15 : f64
      }
      %subview_14 = memref.subview %arg6[1, 1] [%9, %7] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
      memref.copy %alloc_13, %subview_14 : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
      scf.yield %arg5, %arg6 : memref<?x?xf64>, memref<?x?xf64>
    }
    memref.copy %10#0, %arg3 : memref<?x?xf64> to memref<?x?xf64>
    memref.copy %10#1, %arg2 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

