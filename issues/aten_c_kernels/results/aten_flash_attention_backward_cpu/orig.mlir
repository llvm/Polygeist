module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flash_attention_backward_cpu(%arg0: memref<?x2x16x32xf32>, %arg1: memref<?x2x16x32xf32>, %arg2: memref<?x2x16x32xf32>, %arg3: memref<?x2x16x32xf32>, %arg4: memref<?x2x16x32xf32>, %arg5: memref<?x2x16x32xf32>, %arg6: memref<?x2x16x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.176776692 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<16xf32>
    %alloca_2 = memref.alloca() : memref<16xf32>
    %0 = "polygeist.memref2pointer"(%arg4) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    affine.for %arg7 = 0 to 1024 {
      %3 = arith.index_cast %arg7 : index to i32
      %4 = llvm.getelementptr %0[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %4 : f32, !llvm.ptr
    }
    %1 = "polygeist.memref2pointer"(%arg5) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg6) : (memref<?x2x16x32xf32>) -> !llvm.ptr
    affine.for %arg7 = 0 to 1024 {
      %3 = arith.index_cast %arg7 : index to i32
      %4 = llvm.getelementptr %1[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %4 : f32, !llvm.ptr
      %5 = llvm.getelementptr %2[%3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst_1, %5 : f32, !llvm.ptr
    }
    affine.for %arg7 = 0 to 2 {
      affine.for %arg8 = 0 to 16 {
        %3 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_0) -> (f32) {
          %6 = affine.for %arg11 = 0 to 32 iter_args(%arg12 = %cst_1) -> (f32) {
            %10 = affine.load %arg0[0, %arg7, %arg8, %arg11] : memref<?x2x16x32xf32>
            %11 = affine.load %arg1[0, %arg7, %arg9, %arg11] : memref<?x2x16x32xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = arith.addf %arg12, %12 : f32
            affine.yield %13 : f32
          }
          %7 = arith.mulf %6, %cst : f32
          affine.store %7, %alloca_2[%arg9] : memref<16xf32>
          %8 = arith.cmpf ogt, %7, %arg10 : f32
          %9 = arith.select %8, %7, %arg10 : f32
          affine.yield %9 : f32
        }
        %4 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_1) -> (f32) {
          %6 = affine.load %alloca_2[%arg9] : memref<16xf32>
          %7 = arith.subf %6, %3 : f32
          %8 = math.exp %7 : f32
          affine.store %8, %alloca_2[%arg9] : memref<16xf32>
          %9 = arith.addf %arg10, %8 : f32
          affine.yield %9 : f32
        }
        %5 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_1) -> (f32) {
          %6 = affine.load %alloca_2[%arg9] : memref<16xf32>
          %7 = arith.divf %6, %4 : f32
          affine.store %7, %alloca_2[%arg9] : memref<16xf32>
          %8 = affine.for %arg11 = 0 to 32 iter_args(%arg12 = %cst_1) -> (f32) {
            %11 = affine.load %arg3[0, %arg7, %arg8, %arg11] : memref<?x2x16x32xf32>
            %12 = affine.load %arg2[0, %arg7, %arg9, %arg11] : memref<?x2x16x32xf32>
            %13 = arith.mulf %11, %12 : f32
            %14 = arith.addf %arg12, %13 : f32
            %15 = arith.mulf %7, %11 : f32
            %16 = affine.load %arg6[0, %arg7, %arg9, %arg11] : memref<?x2x16x32xf32>
            %17 = arith.addf %16, %15 : f32
            affine.store %17, %arg6[0, %arg7, %arg9, %arg11] : memref<?x2x16x32xf32>
            affine.yield %14 : f32
          }
          affine.store %8, %alloca[%arg9] : memref<16xf32>
          %9 = arith.mulf %8, %7 : f32
          %10 = arith.addf %arg10, %9 : f32
          affine.yield %10 : f32
        }
        affine.for %arg9 = 0 to 16 {
          %6 = affine.load %alloca_2[%arg9] : memref<16xf32>
          %7 = affine.load %alloca[%arg9] : memref<16xf32>
          %8 = arith.subf %7, %5 : f32
          %9 = arith.mulf %6, %8 : f32
          %10 = arith.mulf %9, %cst : f32
          affine.for %arg10 = 0 to 32 {
            %11 = affine.load %arg1[0, %arg7, %arg9, %arg10] : memref<?x2x16x32xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = affine.load %arg4[0, %arg7, %arg8, %arg10] : memref<?x2x16x32xf32>
            %14 = arith.addf %13, %12 : f32
            affine.store %14, %arg4[0, %arg7, %arg8, %arg10] : memref<?x2x16x32xf32>
            %15 = affine.load %arg0[0, %arg7, %arg8, %arg10] : memref<?x2x16x32xf32>
            %16 = arith.mulf %10, %15 : f32
            %17 = affine.load %arg5[0, %arg7, %arg9, %arg10] : memref<?x2x16x32xf32>
            %18 = arith.addf %17, %16 : f32
            affine.store %18, %arg5[0, %arg7, %arg9, %arg10] : memref<?x2x16x32xf32>
          }
        }
      }
    }
    return
  }
}
