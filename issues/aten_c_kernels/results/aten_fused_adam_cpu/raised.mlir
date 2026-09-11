module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fused_adam_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32, %arg12: f32, %arg13: i32, %arg14: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.divf %arg5, %arg8 : f32
    %1 = arith.cmpi ne, %arg13, %c0_i32 : i32
    %2 = arith.cmpf une, %arg10, %cst_0 : f32
    %3 = arith.subf %cst, %arg6 : f32
    %4 = arith.subf %cst, %arg7 : f32
    %5 = arith.cmpi ne, %arg14, %c0_i32 : i32
    affine.for %arg15 = 0 to 4096 {
      %6 = affine.load %arg1[%arg15] : memref<?xf32>
      %7 = arith.divf %6, %arg12 : f32
      affine.store %7, %arg1[%arg15] : memref<?xf32>
      %8 = arith.negf %7 : f32
      %9 = arith.select %1, %8, %7 : f32
      %10 = affine.load %arg0[%arg15] : memref<?xf32>
      %11 = arith.mulf %10, %arg10 : f32
      %12 = arith.addf %9, %11 : f32
      %13 = arith.select %2, %12, %9 : f32
      %14 = affine.load %arg2[%arg15] : memref<?xf32>
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %3, %15 : f32
      %17 = arith.addf %14, %16 : f32
      affine.store %17, %arg2[%arg15] : memref<?xf32>
      %18 = affine.load %arg3[%arg15] : memref<?xf32>
      %19 = arith.mulf %arg7, %18 : f32
      %20 = arith.mulf %4, %13 : f32
      %21 = arith.mulf %20, %13 : f32
      %22 = arith.addf %19, %21 : f32
      affine.store %22, %arg3[%arg15] : memref<?xf32>
      %23 = scf.if %5 -> (f32) {
        %32 = affine.load %arg4[%arg15] : memref<?xf32>
        %33 = arith.cmpf ogt, %32, %22 : f32
        %34 = arith.select %33, %32, %22 : f32
        affine.store %34, %arg4[%arg15] : memref<?xf32>
        scf.yield %34 : f32
      } else {
        scf.yield %22 : f32
      }
      %24 = affine.load %arg2[%arg15] : memref<?xf32>
      %25 = arith.mulf %0, %24 : f32
      %26 = math.sqrt %23 : f32
      %27 = arith.divf %26, %arg9 : f32
      %28 = arith.addf %27, %arg11 : f32
      %29 = arith.divf %25, %28 : f32
      %30 = affine.load %arg0[%arg15] : memref<?xf32>
      %31 = arith.subf %30, %29 : f32
      affine.store %31, %arg0[%arg15] : memref<?xf32>
    }
    return
  }
}

