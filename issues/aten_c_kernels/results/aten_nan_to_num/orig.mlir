module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nan_to_num(%arg0: memref<?xf32>, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.negf %arg4 : f32
    affine.for %arg6 = 0 to 4096 {
      %1 = affine.load %arg0[%arg6] : memref<?xf32>
      %2 = arith.cmpf une, %1, %1 : f32
      %3 = scf.if %2 -> (f32) {
        scf.yield %arg1 : f32
      } else {
        %4 = arith.cmpf ogt, %1, %arg4 : f32
        %5 = scf.if %4 -> (f32) {
          scf.yield %arg2 : f32
        } else {
          %6 = arith.cmpf olt, %1, %0 : f32
          %7 = arith.select %6, %arg3, %1 : f32
          scf.yield %7 : f32
        }
        scf.yield %5 : f32
      }
      affine.store %3, %arg5[%arg6] : memref<?xf32>
    }
    return
  }
}
