#set = affine_set<()[s0] : (s0 - 2 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: i32, %arg5: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.cmpi eq, %arg4, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %3 = scf.if %1 -> (f32) {
      scf.yield %cst : f32
    } else {
      %7 = arith.cmpi eq, %arg4, %c3_i32 : i32
      %8 = arith.select %7, %cst_0, %cst_1 : f32
      scf.yield %8 : f32
    }
    %4 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = arith.cmpi eq, %arg4, %c1_i32 : i32
      scf.yield %7 : i1
    }
    %5 = arith.addi %0, %c-1 : index
    %6 = arith.cmpi eq, %5, %c0 : index
    affine.for %arg6 = 0 to 16 {
      affine.for %arg7 = 0 to 24 {
        %7 = affine.load %arg0[%arg6] : memref<?xi32>
        %8 = affine.load %arg0[%arg6 + 1] : memref<?xi32>
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.index_cast %7 : i32 to index
        %11 = scf.for %arg8 = %10 to %9 step %c1 iter_args(%arg9 = %3) -> (f32) {
          %15 = memref.load %arg2[%arg8] : memref<?xf32>
          %16 = memref.load %arg1[%arg8] : memref<?xi32>
          %17 = arith.index_cast %16 : i32 to index
          %18 = memref.load %arg3[%17, %arg7] : memref<?x24xf32>
          %19 = arith.mulf %15, %18 : f32
          %20 = scf.if %4 -> (f32) {
            %21 = arith.addf %arg9, %19 : f32
            scf.yield %21 : f32
          } else {
            %21 = affine.if #set()[%0] -> f32 {
              %22 = arith.cmpf ogt, %arg9, %19 : f32
              %23 = arith.select %22, %arg9, %19 : f32
              affine.yield %23 : f32
            } else {
              %22 = arith.cmpf olt, %arg9, %19 : f32
              %23 = arith.select %22, %arg9, %19 : f32
              affine.yield %23 : f32
            }
            scf.yield %21 : f32
          }
          scf.yield %20 : f32
        }
        %12 = arith.cmpi sgt, %8, %7 : i32
        %13 = arith.andi %6, %12 : i1
        %14 = scf.if %13 -> (f32) {
          %15 = arith.subi %8, %7 : i32
          %16 = arith.sitofp %15 : i32 to f32
          %17 = arith.divf %11, %16 : f32
          scf.yield %17 : f32
        } else {
          scf.yield %11 : f32
        }
        affine.store %14, %arg5[%arg6, %arg7] : memref<?x24xf32>
      }
    }
    return
  }
}
