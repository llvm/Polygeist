#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_aminmax_allreduce_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4095 = arith.constant 4095 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %extracted = tensor.extract %0[%c0] : tensor<?xf32>
    %3 = tensor.empty() : tensor<f32>
    %inserted = tensor.insert %extracted into %3[] : tensor<f32>
    %4 = tensor.empty() : tensor<f32>
    %inserted_0 = tensor.insert %extracted into %4[] : tensor<f32>
    %extracted_slice = tensor.extract_slice %0[1] [%c4095] [1] : tensor<?xf32> to tensor<?xf32>
    %5:2 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%inserted, %inserted_0 : tensor<f32>, tensor<f32>) {
    ^bb0(%in: f32, %out: f32, %out_5: f32):
      %8 = arith.cmpf olt, %in, %out_5 : f32
      %9 = arith.select %8, %in, %out_5 : f32
      %10 = arith.cmpf ogt, %in, %out : f32
      %11 = arith.select %10, %in, %out : f32
      linalg.yield %11, %9 : f32, f32
    } -> (tensor<f32>, tensor<f32>)
    %extracted_1 = tensor.extract %5#0[] : tensor<f32>
    %extracted_2 = tensor.extract %5#1[] : tensor<f32>
    %inserted_3 = tensor.insert %extracted_2 into %1[%c0] : tensor<?xf32>
    %6 = bufferization.to_memref %inserted_3 : memref<?xf32>
    memref.copy %6, %arg1 : memref<?xf32> to memref<?xf32>
    %inserted_4 = tensor.insert %extracted_1 into %2[%c0] : tensor<?xf32>
    %7 = bufferization.to_memref %inserted_4 : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

