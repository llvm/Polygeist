module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @neg_csc_twotargets(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %1 = affine.load %arg1[%arg7] : memref<?xi32>
      %2 = scf.while (%arg8 = %1) : (i32) -> i32 {
        %3 = affine.load %arg1[%arg7 + 1] : memref<?xi32>
        %4 = arith.cmpi slt, %arg8, %3 : i32
        scf.condition(%4) %arg8 : i32
      } do {
      ^bb0(%arg8: i32):
        %3 = arith.index_cast %arg8 : i32 to index
        %4 = memref.load %arg2[%3] : memref<?xi32>
        %5 = arith.index_cast %4 : i32 to index
        %6 = memref.load %arg3[%3] : memref<?xf64>
        %7 = affine.load %arg4[%arg7] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = memref.load %arg5[%5] : memref<?xf64>
        %10 = arith.addf %9, %8 : f64
        memref.store %10, %arg5[%5] : memref<?xf64>
        %11 = memref.load %arg2[%3] : memref<?xi32>
        %12 = arith.index_cast %11 : i32 to index
        %13 = memref.load %arg3[%3] : memref<?xf64>
        %14 = memref.load %arg6[%12] : memref<?xf64>
        %15 = arith.addf %14, %13 : f64
        memref.store %15, %arg6[%12] : memref<?xf64>
        %16 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %16 : i32
      }
    }
    return
  }
}

