module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_2d_scalarized(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 25 {
        %0 = affine.load %arg2[%arg7 + %arg6 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.divf %3, %6 : f64
        %8 = arith.negf %1 : f64
        %9 = arith.divf %8, %6 : f64
        %10 = affine.load %arg4[%arg7 + %arg6 * 100] : memref<?xf64>
        %11 = affine.load %arg4[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %12 = affine.load %arg4[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %13 = affine.load %arg4[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %14 = arith.addf %10, %13 : f64
        %15 = affine.load %arg3[%arg7] : memref<?xf64>
        %16 = arith.mulf %15, %6 : f64
        %17 = affine.load %arg0[%arg7 + %arg6 * 25] : memref<?xf64>
        %18 = affine.load %arg1[%arg7 + %arg6 * 25] : memref<?xf64>
        %19 = arith.mulf %17, %7 : f64
        %20 = arith.mulf %19, %14 : f64
        %21 = arith.addf %10, %10 : f64
        %22 = arith.mulf %7, %21 : f64
        %23 = arith.addf %11, %12 : f64
        %24 = arith.mulf %9, %23 : f64
        %25 = arith.addf %22, %24 : f64
        %26 = arith.mulf %18, %25 : f64
        %27 = arith.addf %20, %26 : f64
        %28 = arith.mulf %16, %27 : f64
        affine.store %28, %arg5[%arg7 + %arg6 * 100] : memref<?xf64>
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 25 {
        %0 = affine.load %arg2[%arg7 + %arg6 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.divf %3, %6 : f64
        %8 = arith.negf %1 : f64
        %9 = arith.divf %8, %6 : f64
        %10 = affine.load %arg4[%arg7 + %arg6 * 100] : memref<?xf64>
        %11 = affine.load %arg4[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %12 = affine.load %arg4[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %13 = affine.load %arg4[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %14 = arith.addf %10, %13 : f64
        %15 = affine.load %arg3[%arg7] : memref<?xf64>
        %16 = arith.mulf %15, %6 : f64
        %17 = affine.load %arg0[%arg7 + %arg6 * 25] : memref<?xf64>
        %18 = affine.load %arg1[%arg7 + %arg6 * 25] : memref<?xf64>
        %19 = arith.mulf %17, %9 : f64
        %20 = arith.mulf %19, %14 : f64
        %21 = arith.addf %12, %11 : f64
        %22 = arith.mulf %7, %21 : f64
        %23 = arith.addf %13, %13 : f64
        %24 = arith.mulf %9, %23 : f64
        %25 = arith.addf %22, %24 : f64
        %26 = arith.mulf %18, %25 : f64
        %27 = arith.addf %20, %26 : f64
        %28 = arith.mulf %16, %27 : f64
        affine.store %28, %arg5[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 25 {
        %0 = affine.load %arg2[%arg7 + %arg6 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.negf %2 : f64
        %8 = arith.divf %7, %6 : f64
        %9 = arith.divf %0, %6 : f64
        %10 = affine.load %arg4[%arg7 + %arg6 * 100] : memref<?xf64>
        %11 = affine.load %arg4[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %12 = affine.load %arg4[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %13 = affine.load %arg4[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %14 = arith.addf %10, %13 : f64
        %15 = affine.load %arg3[%arg7] : memref<?xf64>
        %16 = arith.mulf %15, %6 : f64
        %17 = affine.load %arg0[%arg7 + %arg6 * 25] : memref<?xf64>
        %18 = affine.load %arg1[%arg7 + %arg6 * 25] : memref<?xf64>
        %19 = arith.mulf %17, %8 : f64
        %20 = arith.mulf %19, %14 : f64
        %21 = arith.addf %10, %10 : f64
        %22 = arith.mulf %8, %21 : f64
        %23 = arith.addf %11, %12 : f64
        %24 = arith.mulf %9, %23 : f64
        %25 = arith.addf %22, %24 : f64
        %26 = arith.mulf %18, %25 : f64
        %27 = arith.addf %20, %26 : f64
        %28 = arith.mulf %16, %27 : f64
        affine.store %28, %arg5[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
      }
    }
    affine.for %arg6 = 0 to 2 {
      affine.for %arg7 = 0 to 25 {
        %0 = affine.load %arg2[%arg7 + %arg6 * 100] : memref<?xf64>
        %1 = affine.load %arg2[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %2 = affine.load %arg2[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %3 = affine.load %arg2[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %4 = arith.mulf %0, %3 : f64
        %5 = arith.mulf %1, %2 : f64
        %6 = arith.subf %4, %5 : f64
        %7 = arith.negf %2 : f64
        %8 = arith.divf %7, %6 : f64
        %9 = arith.divf %0, %6 : f64
        %10 = affine.load %arg4[%arg7 + %arg6 * 100] : memref<?xf64>
        %11 = affine.load %arg4[%arg7 + %arg6 * 100 + 25] : memref<?xf64>
        %12 = affine.load %arg4[%arg7 + %arg6 * 100 + 50] : memref<?xf64>
        %13 = affine.load %arg4[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
        %14 = arith.addf %10, %13 : f64
        %15 = affine.load %arg3[%arg7] : memref<?xf64>
        %16 = arith.mulf %15, %6 : f64
        %17 = affine.load %arg0[%arg7 + %arg6 * 25] : memref<?xf64>
        %18 = affine.load %arg1[%arg7 + %arg6 * 25] : memref<?xf64>
        %19 = arith.mulf %17, %9 : f64
        %20 = arith.mulf %19, %14 : f64
        %21 = arith.addf %12, %11 : f64
        %22 = arith.mulf %8, %21 : f64
        %23 = arith.addf %13, %13 : f64
        %24 = arith.mulf %9, %23 : f64
        %25 = arith.addf %22, %24 : f64
        %26 = arith.mulf %18, %25 : f64
        %27 = arith.addf %20, %26 : f64
        %28 = arith.mulf %16, %27 : f64
        affine.store %28, %arg5[%arg7 + %arg6 * 100 + 75] : memref<?xf64>
      }
    }
    return
  }
}
