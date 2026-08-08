#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_count_nonzero_impl_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = kernel.launch @memset_zero_1D(%1) : (tensor<?xi32>) -> tensor<?xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c32, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%c32] [1] : tensor<?xi32> to tensor<?xi32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%extracted_slice : tensor<?x?xf32>) outs(%extracted_slice_0 : tensor<?xi32>) {
    ^bb0(%in: f32, %out: i32):
      %5 = arith.cmpf une, %in, %cst : f32
      %6 = arith.extui %5 : i1 to i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<?xi32>
    %inserted_slice = tensor.insert_slice %3 into %2[0] [%c32] [1] : tensor<?xi32> into tensor<?xi32>
    %4 = bufferization.to_memref %inserted_slice : memref<?xi32>
    memref.copy %4, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

