module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_histogramdd_cpu(%arg0: memref<?x2xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: memref<?x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+01 : f32
    %cst_0 = arith.constant 1.600000e+01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c12_i32 = arith.constant 12 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg7 = 0 to 16 {
      affine.for %arg8 = 0 to 12 {
        affine.store %cst_1, %arg6[%arg7, %arg8] : memref<?x12xf32>
      }
    }
    %0 = arith.subf %arg3, %arg2 : f32
    %1 = arith.subf %arg5, %arg4 : f32
    affine.for %arg7 = 0 to 4096 {
      %2 = affine.load %arg0[%arg7, 0] : memref<?x2xf32>
      %3 = arith.subf %2, %arg2 : f32
      %4 = arith.mulf %3, %cst_0 : f32
      %5 = arith.divf %4, %0 : f32
      %6 = arith.fptosi %5 : f32 to i32
      %7 = affine.load %arg0[%arg7, 1] : memref<?x2xf32>
      %8 = arith.subf %7, %arg4 : f32
      %9 = arith.mulf %8, %cst : f32
      %10 = arith.divf %9, %1 : f32
      %11 = arith.fptosi %10 : f32 to i32
      %12 = arith.cmpi sge, %6, %c0_i32 : i32
      %13 = arith.cmpi slt, %6, %c16_i32 : i32
      %14 = arith.cmpi sge, %11, %c0_i32 : i32
      %15 = arith.cmpi slt, %11, %c12_i32 : i32
      %16 = arith.andi %14, %15 : i1
      %17 = arith.andi %13, %16 : i1
      %18 = arith.andi %12, %17 : i1
      scf.if %18 {
        %19 = arith.index_cast %6 : i32 to index
        %20 = arith.index_cast %11 : i32 to index
        %21 = affine.load %arg1[%arg7] : memref<?xf32>
        %22 = memref.load %arg6[%19, %20] : memref<?x12xf32>
        %23 = arith.addf %22, %21 : f32
        memref.store %23, %arg6[%19, %20] : memref<?x12xf32>
      }
    }
    return
  }
}
