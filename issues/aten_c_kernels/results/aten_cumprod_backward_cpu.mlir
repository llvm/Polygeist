#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cumprod_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    affine.for %arg4 = 0 to 128 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = affine.for %arg5 = #map(%arg4) to 128 iter_args(%arg6 = %cst_0) -> (f32) {
        %2 = affine.for %arg7 = 0 to #map1(%arg5) iter_args(%arg8 = %cst) -> (f32) {
          %6 = arith.index_cast %arg7 : index to i32
          %7 = arith.cmpi ne, %6, %0 : i32
          %8 = scf.if %7 -> (f32) {
            %9 = affine.load %arg0[%arg7] : memref<?xf32>
            %10 = arith.mulf %arg8, %9 : f32
            scf.yield %10 : f32
          } else {
            scf.yield %arg8 : f32
          }
          affine.yield %8 : f32
        }
        %3 = affine.load %arg2[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %2 : f32
        %5 = arith.addf %arg6, %4 : f32
        affine.yield %5 : f32
      }
      affine.store %1, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
}
