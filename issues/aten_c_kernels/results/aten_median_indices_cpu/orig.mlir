#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_median_indices_cpu(%arg0: memref<?x63xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c31_i32 = arith.constant 31 : i32
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 32 {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = affine.for %arg5 = #map(%arg4) to 63 iter_args(%arg6 = %1) -> (i32) {
          %6 = arith.index_cast %arg5 : index to i32
          %7 = affine.load %arg0[%arg3, %arg5] : memref<?x63xf32>
          %8 = arith.index_cast %arg6 : i32 to index
          %9 = memref.load %arg0[%arg3, %8] : memref<?x63xf32>
          %10 = arith.cmpf olt, %7, %9 : f32
          %11 = arith.select %10, %6, %arg6 : i32
          affine.yield %11 : i32
        }
        %3 = affine.load %arg0[%arg3, %arg4] : memref<?x63xf32>
        %4 = arith.index_cast %2 : i32 to index
        %5 = memref.load %arg0[%arg3, %4] : memref<?x63xf32>
        affine.store %5, %arg0[%arg3, %arg4] : memref<?x63xf32>
        memref.store %3, %arg0[%arg3, %4] : memref<?x63xf32>
      }
      %0 = affine.load %arg0[%arg3, 31] : memref<?x63xf32>
      affine.store %0, %arg1[%arg3] : memref<?xf32>
      affine.store %c31_i32, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}
