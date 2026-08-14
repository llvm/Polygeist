#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unpack_pivots_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?xi32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xi32>) {
    ^bb0(%out: i32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      linalg.yield %6 : i32
    } -> tensor<?xi32>
    %3 = affine.for %arg2 = 0 to 128 iter_args(%arg3 = %2) -> (tensor<?xi32>) {
      %extracted = tensor.extract %1[%arg2] : tensor<?xi32>
      %5 = arith.addi %extracted, %c-1_i32 : i32
      %extracted_0 = tensor.extract %arg3[%arg2] : tensor<?xi32>
      %6 = arith.index_cast %5 : i32 to index
      %extracted_1 = tensor.extract %arg3[%6] : tensor<?xi32>
      %inserted = tensor.insert %extracted_1 into %arg3[%arg2] : tensor<?xi32>
      %inserted_2 = tensor.insert %extracted_0 into %inserted[%6] : tensor<?xi32>
      affine.yield %inserted_2 : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

