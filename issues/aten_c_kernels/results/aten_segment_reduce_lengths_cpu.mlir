#set = affine_set<()[s0] : (s0 - 2 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_segment_reduce_lengths_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.cmpi eq, %arg2, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg2, %c0_i32 : i32
    %3 = scf.if %1 -> (f32) {
      scf.yield %cst : f32
    } else {
      %8 = arith.cmpi eq, %arg2, %c3_i32 : i32
      %9 = arith.select %8, %cst_0, %cst_1 : f32
      scf.yield %9 : f32
    }
    %4 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = arith.cmpi eq, %arg2, %c1_i32 : i32
      scf.yield %8 : i1
    }
    %5 = arith.addi %0, %c-1 : index
    %6 = arith.cmpi eq, %5, %c0 : index
    %7 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %c0_i32) -> (i32) {
      %8 = affine.load %arg1[%arg4] : memref<?xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %arg5 : i32 to index
      %11 = arith.addi %10, %9 : index
      %12 = arith.index_cast %11 : index to i32
      %13 = scf.for %arg6 = %c0 to %9 step %c1 iter_args(%arg7 = %3) -> (f32) {
        %17 = arith.addi %10, %arg6 : index
        %18 = memref.load %arg0[%17] : memref<?xf32>
        %19 = scf.if %4 -> (f32) {
          %20 = arith.addf %arg7, %18 : f32
          scf.yield %20 : f32
        } else {
          %20 = affine.if #set()[%0] -> f32 {
            %21 = arith.cmpf ogt, %arg7, %18 : f32
            %22 = arith.select %21, %arg7, %18 : f32
            affine.yield %22 : f32
          } else {
            %21 = arith.cmpf olt, %arg7, %18 : f32
            %22 = arith.select %21, %arg7, %18 : f32
            affine.yield %22 : f32
          }
          scf.yield %20 : f32
        }
        scf.yield %19 : f32
      }
      %14 = arith.cmpi ne, %8, %c0_i32 : i32
      %15 = arith.andi %6, %14 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.sitofp %8 : i32 to f32
        %18 = arith.divf %13, %17 : f32
        scf.yield %18 : f32
      } else {
        scf.yield %13 : f32
      }
      affine.store %16, %arg3[%arg4] : memref<?xf32>
      affine.yield %12 : i32
    }
    return
  }
}
