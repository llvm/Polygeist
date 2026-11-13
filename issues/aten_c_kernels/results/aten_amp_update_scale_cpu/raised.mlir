module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_amp_update_scale_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.load %arg2[0] : memref<?xf32>
    %1 = arith.cmpf une, %0, %cst : f32
    scf.if %1 {
      %2 = affine.load %arg0[0] : memref<?xf32>
      %3 = arith.mulf %2, %arg4 : f32
      affine.store %3, %arg0[0] : memref<?xf32>
      affine.store %c0_i32, %arg1[0] : memref<?xi32>
    } else {
      %2 = affine.load %arg1[0] : memref<?xi32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.cmpi eq, %3, %arg5 : i32
      scf.if %4 {
        %5 = affine.load %arg0[0] : memref<?xf32>
        %6 = arith.mulf %5, %arg3 : f32
        affine.store %6, %arg0[0] : memref<?xf32>
        affine.store %c0_i32, %arg1[0] : memref<?xi32>
      } else {
        affine.store %3, %arg1[0] : memref<?xi32>
      }
    }
    return
  }
}

