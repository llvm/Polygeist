module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bilinear2d(%arg0: memref<?x3x4x4xf32>, %arg1: memref<?x3x8x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 8 {
          %0 = arith.index_cast %arg4 : index to i32
          %1 = arith.remsi %0, %c2_i32 : i32
          %2 = arith.sitofp %1 : i32 to f32
          %3 = arith.mulf %2, %cst_0 : f32
          %4 = arith.subf %cst, %3 : f32
          %5 = arith.divsi %0, %c2_i32 : i32
          %6 = arith.index_cast %5 : i32 to index
          %7 = arith.addi %5, %c1_i32 : i32
          %8 = arith.cmpi slt, %7, %c4_i32 : i32
          %9 = arith.select %8, %7, %5 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = arith.cmpi slt, %arg4, %c0 : index
          %12 = arith.subi %c-1, %arg4 : index
          %13 = arith.select %11, %12, %arg4 : index
          %14 = arith.divsi %13, %c2 : index
          %15 = arith.subi %c-1, %14 : index
          %16 = arith.select %11, %15, %14 : index
          affine.for %arg5 = 0 to 8 {
            %17 = arith.index_cast %arg5 : index to i32
            %18 = arith.remsi %17, %c2_i32 : i32
            %19 = arith.sitofp %18 : i32 to f32
            %20 = arith.mulf %19, %cst_0 : f32
            %21 = arith.subf %cst, %20 : f32
            %22 = arith.divsi %17, %c2_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = arith.cmpi slt, %arg5, %c0 : index
            %25 = arith.subi %c-1, %arg5 : index
            %26 = arith.select %24, %25, %arg5 : index
            %27 = arith.divsi %26, %c2 : index
            %28 = arith.subi %c-1, %27 : index
            %29 = arith.select %24, %28, %27 : index
            %30 = memref.load %arg0[%arg2, %arg3, %16, %29] : memref<?x3x4x4xf32>
            %31 = arith.mulf %21, %30 : f32
            %32 = arith.addi %22, %c1_i32 : i32
            %33 = arith.cmpi slt, %32, %c4_i32 : i32
            %34 = arith.select %33, %32, %22 : i32
            %35 = arith.index_cast %34 : i32 to index
            %36 = memref.load %arg0[%arg2, %arg3, %6, %35] : memref<?x3x4x4xf32>
            %37 = arith.mulf %20, %36 : f32
            %38 = arith.addf %31, %37 : f32
            %39 = arith.mulf %4, %38 : f32
            %40 = memref.load %arg0[%arg2, %arg3, %10, %23] : memref<?x3x4x4xf32>
            %41 = arith.mulf %21, %40 : f32
            %42 = memref.load %arg0[%arg2, %arg3, %10, %35] : memref<?x3x4x4xf32>
            %43 = arith.mulf %20, %42 : f32
            %44 = arith.addf %41, %43 : f32
            %45 = arith.mulf %3, %44 : f32
            %46 = arith.addf %39, %45 : f32
            affine.store %46, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x3x8x8xf32>
          }
        }
      }
    }
    return
  }
}
