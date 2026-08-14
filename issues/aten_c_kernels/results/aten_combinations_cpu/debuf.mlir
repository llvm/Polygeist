#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_combinations_cpu(%arg0: memref<?xf32>, %arg1: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c63_i32 = arith.constant 63 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x2xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %0) -> (tensor<?x2xf32>) {
      %4 = arith.index_cast %arg2 : index to i32
      %5 = arith.subi %c63_i32, %4 : i32
      %6 = arith.muli %4, %5 : i32
      %7 = arith.divsi %6, %c2_i32 : i32
      %8 = affine.for %arg4 = #map(%arg2) to 32 iter_args(%arg5 = %arg3) -> (tensor<?x2xf32>) {
        %9 = arith.index_cast %arg4 : index to i32
        %10 = arith.subi %9, %4 : i32
        %11 = arith.addi %10, %c-1_i32 : i32
        %12 = arith.addi %7, %11 : i32
        %13 = arith.index_cast %12 : i32 to index
        %extracted = tensor.extract %1[%arg2] : tensor<?xf32>
        %inserted = tensor.insert %extracted into %arg5[%13, %c0] : tensor<?x2xf32>
        %extracted_0 = tensor.extract %1[%arg4] : tensor<?xf32>
        %inserted_1 = tensor.insert %extracted_0 into %inserted[%13, %c1] : tensor<?x2xf32>
        affine.yield %inserted_1 : tensor<?x2xf32>
      }
      affine.yield %8 : tensor<?x2xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x2xf32>
    memref.copy %3, %arg1 : memref<?x2xf32> to memref<?x2xf32>
    return
  }
}

