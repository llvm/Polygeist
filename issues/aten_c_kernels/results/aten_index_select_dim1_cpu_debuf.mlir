#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_index_select_dim1_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x16xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c32, %c16] [1, 1] : tensor<?x16xf32> to tensor<?x?xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = memref.load %arg1[%4] : memref<?xi32>
      %6 = arith.index_cast %5 : i32 to index
      %7 = memref.load %arg0[%3, %6] : memref<?x64xf32>
      linalg.yield %7 : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0] [%c32, %c16] [1, 1] : tensor<?x?xf32> into tensor<?x16xf32>
    %2 = bufferization.to_memref %inserted_slice : memref<?x16xf32>
    memref.copy %2, %arg2 : memref<?x16xf32> to memref<?x16xf32>
    return
  }
}

