module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_huber_backward(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.negf %arg3 : f32
    %1 = arith.negf %arg2 : f32
    %2 = arith.mulf %1, %arg3 : f32
    %3 = arith.mulf %arg2, %arg3 : f32
    affine.for %arg5 = 0 to 4096 {
      %4 = affine.load %arg0[%arg5] : memref<?xf32>
      %5 = affine.load %arg1[%arg5] : memref<?xf32>
      %6 = arith.subf %4, %5 : f32
      %7 = arith.cmpf olt, %6, %0 : f32
      %8 = scf.if %7 -> (f32) {
        scf.yield %2 : f32
      } else {
        %9 = arith.cmpf ogt, %6, %arg3 : f32
        %10 = scf.if %9 -> (f32) {
          scf.yield %3 : f32
        } else {
          %11 = arith.mulf %arg2, %6 : f32
          scf.yield %11 : f32
        }
        scf.yield %10 : f32
      }
      affine.store %8, %arg4[%arg5] : memref<?xf32>
    }
    return
  }
}
