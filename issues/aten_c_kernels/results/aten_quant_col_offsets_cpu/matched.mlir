#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quant_col_offsets_cpu(%arg0: memref<?x48xi8>, %arg1: i32, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64 = arith.constant 64 : index
    %c48 = arith.constant 48 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x48xi8>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = arith.muli %arg1, %c64_i32 : i32
    %3 = tensor.empty(%c48) : tensor<?xi32>
    %4 = kernel.launch @memset_zero_1D(%3) : (tensor<?xi32>) -> tensor<?xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c64, %c48] [1, 1] : tensor<?x48xi8> to tensor<?x?xi8>
    %extracted_slice_0 = tensor.extract_slice %4[0] [%c48] [1] : tensor<?xi32> to tensor<?xi32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice : tensor<?x?xi8>) outs(%extracted_slice_0 : tensor<?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %8 = arith.extsi %in : i8 to i32
      %9 = arith.addi %out, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<?xi32>
    %inserted_slice = tensor.insert_slice %5 into %4[0] [%c48] [1] : tensor<?xi32> into tensor<?xi32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%inserted_slice : tensor<?xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %8 = arith.subi %in, %2 : i32
      linalg.yield %8 : i32
    } -> tensor<?xi32>
    %7 = bufferization.to_memref %6 : memref<?xi32>
    memref.copy %7, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

