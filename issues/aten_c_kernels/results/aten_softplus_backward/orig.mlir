module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_softplus_backward(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    affine.for %arg5 = 0 to 4096 {
      %0 = affine.load %arg1[%arg5] : memref<?xf32>
      %1 = arith.mulf %arg2, %0 : f32
      %2 = arith.cmpf ogt, %1, %arg3 : f32
      %3 = scf.if %2 -> (f32) {
        %4 = affine.load %arg0[%arg5] : memref<?xf32>
        scf.yield %4 : f32
      } else {
        %4 = affine.load %arg0[%arg5] : memref<?xf32>
        %5 = math.exp %1 : f32
        %6 = arith.addf %5, %cst : f32
        %7 = arith.divf %cst, %6 : f32
        %8 = arith.subf %cst, %7 : f32
        %9 = arith.mulf %4, %8 : f32
        scf.yield %9 : f32
      }
      affine.store %3, %arg4[%arg5] : memref<?xf32>
    }
    return
  }
}
