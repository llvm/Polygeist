#map = affine_map<(d0)[s0, s1] -> (s0, s1)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0)[s0] -> (d0, s0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x30xf64>, %arg5: memref<?x20xf64>, %arg6: memref<?x30xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %alloca = memref.alloca() : memref<f64>
    %1 = llvm.mlir.undef : f64
    affine.store %1, %alloca[] : memref<f64>
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      affine.for %arg8 = 0 to %0 {
        affine.store %cst, %alloca[] : memref<f64>
        %3 = arith.subi %2, %c1 : index
        %4 = polygeist.submap(%arg6, %arg7, %arg8, %3) {map = #map} : (memref<?x30xf64>, index, index, index) -> memref<?xf64>
        %5 = polygeist.submap(%arg5, %arg7, %3) {map = #map1} : (memref<?x20xf64>, index, index) -> memref<?xf64>
        %6 = polygeist.submap(%arg6, %arg8, %3) {map = #map2} : (memref<?x30xf64>, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%arg5, %arg7, %3) {map = #map1} : (memref<?x20xf64>, index, index) -> memref<?xf64>
        %8 = polygeist.submap(%arg4, %arg8, %3) {map = #map2} : (memref<?x30xf64>, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%alloca, %3) {map = #map3} : (memref<f64>, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["reduction"]} ins(%4, %5, %6, %7 : memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%8, %9 : memref<?xf64>, memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64, %out_3: f64):
          %20 = arith.mulf %arg2, %in : f64
          %21 = arith.mulf %20, %in_0 : f64
          %22 = arith.addf %out, %21 : f64
          %23 = arith.mulf %in_1, %in_2 : f64
          %24 = arith.addf %out_3, %23 : f64
          %25 = linalg.index 0 : index
          %26 = arith.cmpi slt, %25, %arg7 : index
          %27 = arith.select %26, %22, %out : f64
          %28 = arith.select %26, %24, %out_3 : f64
          linalg.yield %27, %28 : f64, f64
        }
        %10 = affine.load %arg4[%arg7, %arg8] : memref<?x30xf64>
        %11 = arith.mulf %arg3, %10 : f64
        %12 = affine.load %arg6[%arg7, %arg8] : memref<?x30xf64>
        %13 = arith.mulf %arg2, %12 : f64
        %14 = affine.load %arg5[%arg7, %arg7] : memref<?x20xf64>
        %15 = arith.mulf %13, %14 : f64
        %16 = arith.addf %11, %15 : f64
        %17 = affine.load %alloca[] : memref<f64>
        %18 = arith.mulf %arg2, %17 : f64
        %19 = arith.addf %16, %18 : f64
        affine.store %19, %arg4[%arg7, %arg8] : memref<?x30xf64>
      }
    }
    return
  }
}

