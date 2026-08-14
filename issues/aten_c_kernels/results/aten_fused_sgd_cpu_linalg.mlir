module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_sgd_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg8, %c0_i32 : i32
    %1 = arith.cmpf une, %arg6, %cst_0 : f32
    %2 = arith.cmpf une, %arg4, %cst_0 : f32
    %3 = arith.cmpi ne, %arg9, %c0_i32 : i32
    %4 = arith.subf %cst, %arg5 : f32
    %5 = arith.cmpi ne, %arg10, %c0_i32 : i32
    affine.for %arg11 = 0 to 4096 {
      %6 = affine.load %arg1[%arg11] : memref<?xf32>
      %7 = arith.divf %6, %arg7 : f32
      affine.store %7, %arg1[%arg11] : memref<?xf32>
      %8 = arith.negf %7 : f32
      %9 = arith.select %0, %8, %7 : f32
      %10 = affine.load %arg0[%arg11] : memref<?xf32>
      %11 = arith.mulf %10, %arg6 : f32
      %12 = arith.addf %9, %11 : f32
      %13 = arith.select %1, %12, %9 : f32
      %14 = scf.if %2 -> (f32) {
        %18 = affine.load %arg2[%arg11] : memref<?xf32>
        %19 = arith.mulf %18, %arg4 : f32
        %20 = arith.mulf %13, %4 : f32
        %21 = arith.addf %19, %20 : f32
        %22 = arith.select %3, %13, %21 : f32
        affine.store %22, %arg2[%arg11] : memref<?xf32>
        %23 = arith.mulf %arg4, %22 : f32
        %24 = arith.addf %13, %23 : f32
        %25 = arith.select %5, %24, %22 : f32
        scf.yield %25 : f32
      } else {
        scf.yield %13 : f32
      }
      %15 = arith.mulf %arg3, %14 : f32
      %16 = affine.load %arg0[%arg11] : memref<?xf32>
      %17 = arith.subf %16, %15 : f32
      affine.store %17, %arg0[%arg11] : memref<?xf32>
    }
    return
  }
}

