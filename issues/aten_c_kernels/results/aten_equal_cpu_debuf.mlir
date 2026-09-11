#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_equal_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xi32>
    %inserted = tensor.insert %c1_i32 into %2[%c0] : tensor<?xi32>
    %extracted_slice = tensor.extract_slice %inserted[0] [1] [1] : tensor<?xi32> to tensor<i32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%0, %1 : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice : tensor<i32>) {
    ^bb0(%in: f32, %in_0: f32, %out: i32):
      %5 = arith.cmpi ne, %out, %c0_i32 : i32
      %6 = arith.cmpf oeq, %in, %in_0 : f32
      %7 = arith.select %5, %6, %false : i1
      %8 = arith.extsi %7 : i1 to i32
      linalg.yield %8 : i32
    } -> tensor<i32>
    %inserted_slice = tensor.insert_slice %3 into %inserted[0] [1] [1] : tensor<i32> into tensor<?xi32>
    %4 = bufferization.to_memref %inserted_slice : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

