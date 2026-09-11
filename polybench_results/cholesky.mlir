#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_cholesky(%arg0: i32, %arg1: memref<?x40xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        affine.for %arg4 = 0 to #map(%arg3) {
          %6 = affine.load %arg1[%arg2, %arg4] : memref<?x40xf64>
          %7 = affine.load %arg1[%arg3, %arg4] : memref<?x40xf64>
          %8 = arith.mulf %6, %7 : f64
          %9 = affine.load %arg1[%arg2, %arg3] : memref<?x40xf64>
          %10 = arith.subf %9, %8 : f64
          affine.store %10, %arg1[%arg2, %arg3] : memref<?x40xf64>
        }
        %3 = affine.load %arg1[%arg3, %arg3] : memref<?x40xf64>
        %4 = affine.load %arg1[%arg2, %arg3] : memref<?x40xf64>
        %5 = arith.divf %4, %3 : f64
        affine.store %5, %arg1[%arg2, %arg3] : memref<?x40xf64>
      }
      affine.for %arg3 = 0 to #map(%arg2) {
        %3 = affine.load %arg1[%arg2, %arg3] : memref<?x40xf64>
        %4 = arith.mulf %3, %3 : f64
        %5 = affine.load %arg1[%arg2, %arg2] : memref<?x40xf64>
        %6 = arith.subf %5, %4 : f64
        affine.store %6, %arg1[%arg2, %arg2] : memref<?x40xf64>
      }
      %1 = affine.load %arg1[%arg2, %arg2] : memref<?x40xf64>
      %2 = math.sqrt %1 : f64
      affine.store %2, %arg1[%arg2, %arg2] : memref<?x40xf64>
    }
    return
  }
}
