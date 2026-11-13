module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @neg_hist_saturate(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c255_i32 = arith.constant 255 : i32
    %c1_i8 = arith.constant 1 : i8
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?xi8>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = affine.for %arg3 = 0 to %2 iter_args(%arg4 = %1) -> (tensor<?xi8>) {
      %extracted = tensor.extract %0[%arg3] : tensor<?xi32>
      %5 = arith.index_cast %extracted : i32 to index
      %extracted_0 = tensor.extract %arg4[%5] : tensor<?xi8>
      %6 = arith.extui %extracted_0 : i8 to i32
      %7 = arith.cmpi slt, %6, %c255_i32 : i32
      %8 = arith.index_cast %extracted : i32 to index
      %extracted_1 = tensor.extract %arg4[%8] : tensor<?xi8>
      %9 = arith.addi %extracted_1, %c1_i8 : i8
      %extracted_2 = tensor.extract %arg4[%8] : tensor<?xi8>
      %10 = arith.select %7, %9, %extracted_2 : i8
      %inserted = tensor.insert %10 into %arg4[%8] : tensor<?xi8>
      affine.yield %inserted : tensor<?xi8>
    }
    %4 = bufferization.to_memref %3 : memref<?xi8>
    memref.copy %4, %arg2 : memref<?xi8> to memref<?xi8>
    return
  }
}

