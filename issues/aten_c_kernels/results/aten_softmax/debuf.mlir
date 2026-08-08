#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_softmax(%arg0: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = tensor.empty() : tensor<f32>
    %2 = llvm.mlir.undef : f32
    %inserted = tensor.insert %2 into %1[] : tensor<f32>
    %3 = tensor.empty() : tensor<f32>
    %inserted_1 = tensor.insert %cst into %3[] : tensor<f32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%0 : tensor<?xf32>) outs(%inserted_1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ogt, %in, %out : f32
      %9 = arith.select %8, %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %inserted_2 = tensor.insert %cst_0 into %inserted[] : tensor<f32>
    %extracted = tensor.extract %4[] : tensor<f32>
    %extracted_slice = tensor.extract_slice %0[0] [%c128] [1] : tensor<?xf32> to tensor<?xf32>
    %5:2 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} outs(%extracted_slice, %inserted_2 : tensor<?xf32>, tensor<f32>) {
    ^bb0(%out: f32, %out_4: f32):
      %8 = arith.subf %out, %extracted : f32
      %9 = math.exp %8 : f32
      %10 = arith.addf %out_4, %9 : f32
      linalg.yield %9, %10 : f32, f32
    } -> (tensor<?xf32>, tensor<f32>)
    %inserted_slice = tensor.insert_slice %5#0 into %0[0] [%c128] [1] : tensor<?xf32> into tensor<?xf32>
    %extracted_3 = tensor.extract %5#1[] : tensor<f32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%inserted_slice : tensor<?xf32>) {
    ^bb0(%out: f32):
      %8 = arith.divf %out, %extracted_3 : f32
      linalg.yield %8 : f32
    } -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg0 : memref<?xf32> to memref<?xf32>
    return
  }
}

