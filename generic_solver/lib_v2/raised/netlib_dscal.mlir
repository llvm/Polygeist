module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dscal(%arg0: i32, %arg1: f64, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.remsi %0, %c5 : index
    %2 = arith.addi %1, %c5 : index
    %3 = arith.cmpi slt, %1, %c0 : index
    %4 = arith.select %3, %2, %1 : index
    %5 = arith.remsi %arg0, %c5_i32 : i32
    %6 = arith.cmpi ne, %5, %c0_i32 : i32
    %7 = scf.if %6 -> (i1) {
      affine.for %arg3 = 0 to %4 {
        %9 = affine.load %arg2[%arg3] : memref<?xf64>
        %10 = arith.mulf %arg1, %9 : f64
        affine.store %10, %arg2[%arg3] : memref<?xf64>
      }
      %8 = arith.cmpi sge, %arg0, %c5_i32 : i32
      scf.yield %8 : i1
    } else {
      scf.yield %true : i1
    }
    scf.if %7 {
      affine.for %arg3 = %4 to %0 step 5 {
        %8 = affine.load %arg2[%arg3] : memref<?xf64>
        %9 = arith.mulf %arg1, %8 : f64
        affine.store %9, %arg2[%arg3] : memref<?xf64>
        %10 = affine.load %arg2[%arg3 + 1] : memref<?xf64>
        %11 = arith.mulf %arg1, %10 : f64
        affine.store %11, %arg2[%arg3 + 1] : memref<?xf64>
        %12 = affine.load %arg2[%arg3 + 2] : memref<?xf64>
        %13 = arith.mulf %arg1, %12 : f64
        affine.store %13, %arg2[%arg3 + 2] : memref<?xf64>
        %14 = affine.load %arg2[%arg3 + 3] : memref<?xf64>
        %15 = arith.mulf %arg1, %14 : f64
        affine.store %15, %arg2[%arg3 + 3] : memref<?xf64>
        %16 = affine.load %arg2[%arg3 + 4] : memref<?xf64>
        %17 = arith.mulf %arg1, %16 : f64
        affine.store %17, %arg2[%arg3 + 4] : memref<?xf64>
      }
    }
    return
  }
}
