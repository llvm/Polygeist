#set = affine_set<(d0) : (d0 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_divdiv_apply_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c24_i32 = arith.constant 24 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<5xf64>
    %alloca_1 = memref.alloca() : memref<5x5xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = arith.index_cast %arg7 : index to i32
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.store %cst, %alloca_1[%arg8, %arg9] : memref<5x5xf64>
        }
      }
      %1 = arith.muli %0, %c24_i32 : i32
      %2 = affine.for %arg8 = 0 to 2 iter_args(%arg9 = %c0_i32) -> (i32) {
        %3 = arith.index_cast %arg8 : index to i32
        %4 = arith.cmpi eq, %3, %c1_i32 : i32
        %5 = arith.select %4, %c3_i32, %c4_i32 : i32
        %6 = arith.cmpi eq, %3, %c0_i32 : i32
        %7 = arith.select %6, %c3_i32, %c4_i32 : i32
        %8 = arith.index_cast %7 : i32 to index
        %9 = arith.index_cast %5 : i32 to index
        scf.for %arg10 = %c0 to %8 step %c1 {
          %12 = arith.index_cast %arg10 : index to i32
          affine.for %arg11 = 0 to 5 {
            affine.store %cst, %alloca_0[%arg11] : memref<5xf64>
          }
          %13 = arith.muli %12, %5 : i32
          scf.for %arg11 = %c0 to %9 step %c1 {
            %14 = arith.index_cast %arg11 : index to i32
            %15 = arith.addi %14, %13 : i32
            %16 = arith.addi %15, %arg9 : i32
            %17 = arith.addi %16, %1 : i32
            %18 = arith.index_cast %17 : i32 to index
            %19 = memref.load %arg5[%18] : memref<?xf64>
            affine.for %arg12 = 0 to 5 {
              %20 = arith.index_cast %arg12 : index to i32
              %21 = affine.if #set(%arg8) -> f64 {
                %25 = arith.muli %20, %c4_i32 : i32
                %26 = arith.addi %25, %14 : i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = memref.load %arg2[%27] : memref<?xf64>
                affine.yield %28 : f64
              } else {
                %25 = arith.muli %20, %c3_i32 : i32
                %26 = arith.addi %25, %14 : i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = memref.load %arg0[%27] : memref<?xf64>
                affine.yield %28 : f64
              }
              %22 = arith.mulf %19, %21 : f64
              %23 = affine.load %alloca_0[%arg12] : memref<5xf64>
              %24 = arith.addf %23, %22 : f64
              affine.store %24, %alloca_0[%arg12] : memref<5xf64>
            }
          }
          affine.for %arg11 = 0 to 5 {
            %14 = arith.index_cast %arg11 : index to i32
            %15 = affine.if #set(%arg8) -> f64 {
              %16 = arith.muli %14, %c3_i32 : i32
              %17 = arith.addi %16, %12 : i32
              %18 = arith.index_cast %17 : i32 to index
              %19 = memref.load %arg0[%18] : memref<?xf64>
              affine.yield %19 : f64
            } else {
              %16 = arith.muli %14, %c4_i32 : i32
              %17 = arith.addi %16, %12 : i32
              %18 = arith.index_cast %17 : i32 to index
              %19 = memref.load %arg2[%18] : memref<?xf64>
              affine.yield %19 : f64
            }
            affine.for %arg12 = 0 to 5 {
              %16 = affine.load %alloca_0[%arg12] : memref<5xf64>
              %17 = arith.mulf %16, %15 : f64
              %18 = affine.load %alloca_1[%arg11, %arg12] : memref<5x5xf64>
              %19 = arith.addf %18, %17 : f64
              affine.store %19, %alloca_1[%arg11, %arg12] : memref<5x5xf64>
            }
          }
        }
        %10 = arith.muli %5, %7 : i32
        %11 = arith.addi %arg9, %10 : i32
        affine.yield %11 : i32
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          %3 = affine.load %arg4[%arg9 + %arg7 * 25 + %arg8 * 5] : memref<?xf64>
          %4 = affine.load %alloca_1[%arg8, %arg9] : memref<5x5xf64>
          %5 = arith.mulf %4, %3 : f64
          affine.store %5, %alloca_1[%arg8, %arg9] : memref<5x5xf64>
        }
      }
      affine.for %arg8 = 0 to 5 {
        %3 = arith.index_cast %arg8 : index to i32
        %4 = affine.for %arg9 = 0 to 2 iter_args(%arg10 = %c0_i32) -> (i32) {
          %5 = arith.index_cast %arg9 : index to i32
          %6 = arith.cmpi eq, %5, %c1_i32 : i32
          %7 = arith.select %6, %c3_i32, %c4_i32 : i32
          %8 = arith.cmpi eq, %5, %c0_i32 : i32
          %9 = arith.select %8, %c3_i32, %c4_i32 : i32
          %10 = arith.index_cast %7 : i32 to index
          scf.for %arg11 = %c0 to %10 step %c1 {
            memref.store %cst, %alloca[%arg11] : memref<4xf64>
          }
          affine.for %arg11 = 0 to 5 {
            %14 = arith.index_cast %arg11 : index to i32
            %15 = affine.load %alloca_1[%arg8, %arg11] : memref<5x5xf64>
            scf.for %arg12 = %c0 to %10 step %c1 {
              %16 = arith.index_cast %arg12 : index to i32
              %17 = affine.if #set(%arg9) -> f64 {
                %21 = arith.muli %16, %c5_i32 : i32
                %22 = arith.addi %21, %14 : i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = memref.load %arg3[%23] : memref<?xf64>
                affine.yield %24 : f64
              } else {
                %21 = arith.muli %16, %c5_i32 : i32
                %22 = arith.addi %21, %14 : i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = memref.load %arg1[%23] : memref<?xf64>
                affine.yield %24 : f64
              }
              %18 = arith.mulf %15, %17 : f64
              %19 = memref.load %alloca[%arg12] : memref<4xf64>
              %20 = arith.addf %19, %18 : f64
              memref.store %20, %alloca[%arg12] : memref<4xf64>
            }
          }
          %11 = arith.index_cast %9 : i32 to index
          scf.for %arg11 = %c0 to %11 step %c1 {
            %14 = arith.index_cast %arg11 : index to i32
            %15 = affine.if #set(%arg9) -> f64 {
              %17 = arith.muli %14, %c5_i32 : i32
              %18 = arith.addi %17, %3 : i32
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg1[%19] : memref<?xf64>
              affine.yield %20 : f64
            } else {
              %17 = arith.muli %14, %c5_i32 : i32
              %18 = arith.addi %17, %3 : i32
              %19 = arith.index_cast %18 : i32 to index
              %20 = memref.load %arg3[%19] : memref<?xf64>
              affine.yield %20 : f64
            }
            %16 = arith.muli %14, %7 : i32
            scf.for %arg12 = %c0 to %10 step %c1 {
              %17 = arith.index_cast %arg12 : index to i32
              %18 = arith.addi %17, %16 : i32
              %19 = arith.addi %18, %arg10 : i32
              %20 = arith.addi %19, %1 : i32
              %21 = arith.index_cast %20 : i32 to index
              %22 = memref.load %alloca[%arg12] : memref<4xf64>
              %23 = arith.mulf %22, %15 : f64
              %24 = memref.load %arg6[%21] : memref<?xf64>
              %25 = arith.addf %24, %23 : f64
              memref.store %25, %arg6[%21] : memref<?xf64>
            }
          }
          %12 = arith.muli %7, %9 : i32
          %13 = arith.addi %arg10, %12 : i32
          affine.yield %13 : i32
        }
      }
    }
    return
  }
}
