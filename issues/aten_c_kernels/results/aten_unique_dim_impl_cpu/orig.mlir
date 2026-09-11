#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_dim_impl_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.for %arg3 = 0 to #map(%arg2) iter_args(%arg4 = %c1_i32) -> (i32) {
        %1 = affine.for %arg5 = 0 to 16 iter_args(%arg6 = %c1_i32) -> (i32) {
          %5 = affine.load %arg0[%arg2, %arg5] : memref<?x16xf32>
          %6 = affine.load %arg0[%arg3, %arg5] : memref<?x16xf32>
          %7 = arith.cmpf oeq, %5, %6 : f32
          %8 = arith.extui %7 : i1 to i32
          %9 = arith.andi %arg6, %8 : i32
          affine.yield %9 : i32
        }
        %2 = arith.cmpi eq, %1, %c0_i32 : i32
        %3 = arith.extui %2 : i1 to i32
        %4 = arith.andi %arg4, %3 : i32
        affine.yield %4 : i32
      }
      affine.store %0, %arg1[%arg2] : memref<?xi32>
    }
    return
  }
}
