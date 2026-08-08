module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sgn_complex_scalarized(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg4 = 0 to 4096 {
      %0 = affine.load %arg0[%arg4] : memref<?xf32>
      %1 = affine.load %arg1[%arg4] : memref<?xf32>
      %2 = func.call @hypotf(%0, %1) : (f32, f32) -> f32
      %3 = arith.cmpf oeq, %2, %cst : f32
      %4 = scf.if %3 -> (f32) {
        scf.yield %cst : f32
      } else {
        %6 = affine.load %arg0[%arg4] : memref<?xf32>
        %7 = arith.divf %6, %2 : f32
        scf.yield %7 : f32
      }
      affine.store %4, %arg2[%arg4] : memref<?xf32>
      %5 = scf.if %3 -> (f32) {
        scf.yield %cst : f32
      } else {
        %6 = affine.load %arg1[%arg4] : memref<?xf32>
        %7 = arith.divf %6, %2 : f32
        scf.yield %7 : f32
      }
      affine.store %5, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @hypotf(f32, f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>, polygeist.pure}
}
