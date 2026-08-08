module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_coalesce_sparse_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg5 = 0 to 512 iter_args(%arg6 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = arith.cmpi ne, %1, %c0_i32 : i32
      %3 = scf.if %2 -> (i1) {
        %5 = affine.load %arg0[%arg5] : memref<?xi32>
        %6 = arith.addi %arg6, %c-1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg2[%7] : memref<?xi32>
        %9 = arith.cmpi eq, %5, %8 : i32
        scf.yield %9 : i1
      } else {
        scf.yield %false : i1
      }
      %4 = scf.if %3 -> (i32) {
        %5 = arith.addi %arg6, %c-1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = affine.load %arg1[%arg5] : memref<?xf32>
        %8 = memref.load %arg3[%6] : memref<?xf32>
        %9 = arith.addf %8, %7 : f32
        memref.store %9, %arg3[%6] : memref<?xf32>
        scf.yield %arg6 : i32
      } else {
        %5 = arith.index_cast %arg6 : i32 to index
        %6 = affine.load %arg0[%arg5] : memref<?xi32>
        memref.store %6, %arg2[%5] : memref<?xi32>
        %7 = arith.addi %arg6, %c1_i32 : i32
        %8 = affine.load %arg1[%arg5] : memref<?xf32>
        memref.store %8, %arg3[%5] : memref<?xf32>
        scf.yield %7 : i32
      }
      affine.yield %4 : i32
    }
    affine.store %0, %arg4[0] : memref<?xi32>
    return
  }
}
