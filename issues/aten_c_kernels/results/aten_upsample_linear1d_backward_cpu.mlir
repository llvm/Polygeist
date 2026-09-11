module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_linear1d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 7.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    affine.for %arg2 = 0 to 8 {
      affine.store %cst_3, %arg1[%arg2] : memref<?xf32>
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 7 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.addf %3, %cst_2 : f32
        %5 = arith.mulf %4, %cst_1 : f32
        %6 = arith.divf %5, %cst_0 : f32
        %7 = arith.subf %6, %cst_2 : f32
        %8 = arith.cmpf olt, %7, %cst_3 : f32
        %9 = arith.select %8, %cst_3, %7 : f32
        %10 = arith.fptosi %9 : f32 to i32
        %11 = arith.addi %1, %10 : i32
        %12 = arith.index_cast %11 : i32 to index
        %13 = affine.load %arg0[%arg3 + %arg2 * 7] : memref<?xf32>
        %14 = arith.sitofp %10 : i32 to f32
        %15 = arith.subf %9, %14 : f32
        %16 = arith.subf %cst, %15 : f32
        %17 = arith.mulf %13, %16 : f32
        %18 = memref.load %arg1[%12] : memref<?xf32>
        %19 = arith.addf %18, %17 : f32
        memref.store %19, %arg1[%12] : memref<?xf32>
        %20 = arith.addi %10, %c1_i32 : i32
        %21 = arith.cmpi slt, %20, %c4_i32 : i32
        %22 = arith.select %21, %20, %10 : i32
        %23 = arith.addi %1, %22 : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = affine.load %arg0[%arg3 + %arg2 * 7] : memref<?xf32>
        %26 = arith.mulf %25, %15 : f32
        %27 = memref.load %arg1[%24] : memref<?xf32>
        %28 = arith.addf %27, %26 : f32
        memref.store %28, %arg1[%24] : memref<?xf32>
      }
    }
    return
  }
}
