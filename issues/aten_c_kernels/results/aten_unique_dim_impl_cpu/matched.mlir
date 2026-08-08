#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_dim_impl_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127 = arith.constant 127 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = affine.for %arg2 = 0 to 128 iter_args(%arg3 = %0) -> (tensor<?xi32>) {
      %inserted = tensor.insert %c1_i32 into %arg3[%arg2] : tensor<?xi32>
      %alloca = memref.alloca(%arg2) : memref<?xi32>
      %3 = bufferization.to_tensor %alloca : memref<?xi32>
      %extracted_slice = tensor.extract_slice %inserted[%arg2] [1] [1] : tensor<?xi32> to tensor<i32>
      %extracted_slice_0 = tensor.extract_slice %3[0] [%c127] [1] : tensor<?xi32> to tensor<?xi32>
      %4 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_0 : tensor<?xi32>) outs(%extracted_slice : tensor<i32>) {
      ^bb0(%in: i32, %out: i32):
        %5 = arith.cmpi eq, %in, %c0_i32 : i32
        %6 = arith.extui %5 : i1 to i32
        %7 = arith.andi %out, %6 : i32
        %8 = linalg.index 0 : index
        %9 = arith.cmpi slt, %8, %arg2 : index
        %10 = arith.select %9, %7, %out : i32
        linalg.yield %10 : i32
      } -> tensor<i32>
      %inserted_slice = tensor.insert_slice %4 into %inserted[%arg2] [1] [1] : tensor<i32> into tensor<?xi32>
      affine.yield %inserted_slice : tensor<?xi32>
    }
    %2 = bufferization.to_memref %1 : memref<?xi32>
    memref.copy %2, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

