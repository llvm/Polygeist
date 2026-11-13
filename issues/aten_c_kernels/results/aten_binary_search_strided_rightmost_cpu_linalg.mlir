module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_binary_search_strided_rightmost_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 128 {
      %0:2 = scf.while (%arg4 = %c512_i32, %arg5 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %2 = arith.cmpi slt, %arg5, %arg4 : i32
        scf.condition(%2) %arg5, %arg4 : i32, i32
      } do {
      ^bb0(%arg4: i32, %arg5: i32):
        %2 = arith.addi %arg4, %arg5 : i32
        %3 = arith.divsi %2, %c2_i32 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg0[%4] : memref<?xi32>
        %6 = affine.load %arg1[%arg3] : memref<?xi32>
        %7 = arith.cmpi sle, %5, %6 : i32
        %8 = arith.select %7, %arg5, %3 : i32
        %9 = arith.addi %3, %c1_i32 : i32
        %10 = arith.select %7, %9, %arg4 : i32
        scf.yield %8, %10 : i32, i32
      }
      %1 = arith.addi %0#0, %c-1_i32 : i32
      affine.store %1, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}

