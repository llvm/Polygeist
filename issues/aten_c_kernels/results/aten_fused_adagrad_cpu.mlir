module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_adagrad_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = arith.subf %arg7, %cst_0 : f32
    %1 = arith.mulf %0, %arg4 : f32
    %2 = arith.addf %1, %cst_0 : f32
    %3 = arith.divf %arg3, %2 : f32
    %4 = arith.cmpi ne, %arg9, %c0_i32 : i32
    %5 = arith.cmpf une, %arg5, %cst : f32
    affine.for %arg10 = 0 to 4096 {
      %6 = affine.load %arg1[%arg10] : memref<?xf32>
      %7 = arith.divf %6, %arg8 : f32
      affine.store %7, %arg1[%arg10] : memref<?xf32>
      %8 = scf.if %4 -> (f32) {
        %19 = arith.negf %7 : f32
        scf.yield %19 : f32
      } else {
        scf.yield %7 : f32
      }
      %9 = scf.if %5 -> (f32) {
        %19 = affine.load %arg0[%arg10] : memref<?xf32>
        %20 = arith.mulf %19, %arg5 : f32
        %21 = arith.addf %8, %20 : f32
        scf.yield %21 : f32
      } else {
        scf.yield %8 : f32
      }
      %10 = arith.mulf %9, %9 : f32
      %11 = affine.load %arg2[%arg10] : memref<?xf32>
      %12 = arith.addf %11, %10 : f32
      affine.store %12, %arg2[%arg10] : memref<?xf32>
      %13 = arith.mulf %3, %9 : f32
      %14 = math.sqrt %12 : f32
      %15 = arith.addf %14, %arg6 : f32
      %16 = arith.divf %13, %15 : f32
      %17 = affine.load %arg0[%arg10] : memref<?xf32>
      %18 = arith.subf %17, %16 : f32
      affine.store %18, %arg0[%arg10] : memref<?xf32>
    }
    return
  }
}
