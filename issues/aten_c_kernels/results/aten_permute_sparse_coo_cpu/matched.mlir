#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_permute_sparse_coo_cpu(%arg0: memref<?x512xi32>, %arg1: memref<?xi32>, %arg2: memref<?x512xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c512 = arith.constant 512 : index
    %c3 = arith.constant 3 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x512xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c3, %c512] [1, 1] : tensor<?x512xi32> to tensor<?x?xi32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = memref.load %arg1[%3] : memref<?xi32>
      %6 = arith.index_cast %5 : i32 to index
      %7 = memref.load %arg0[%6, %4] : memref<?x512xi32>
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0] [%c3, %c512] [1, 1] : tensor<?x?xi32> into tensor<?x512xi32>
    %2 = bufferization.to_memref %inserted_slice : memref<?x512xi32>
    memref.copy %2, %arg2 : memref<?x512xi32> to memref<?x512xi32>
    return
  }
}

