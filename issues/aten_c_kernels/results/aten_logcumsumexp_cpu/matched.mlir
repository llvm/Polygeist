#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_logcumsumexp_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c63 = arith.constant 63 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %2 = tensor.empty(%c32) : tensor<?xf32>
    %3:2 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %1, %arg4 = %2) -> (tensor<?x64xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %0[%arg2, %c0] : tensor<?x64xf32>
      %inserted = tensor.insert %extracted into %arg3[%arg2, %c0] : tensor<?x64xf32>
      %inserted_0 = tensor.insert %extracted into %arg4[%arg2] : tensor<?xf32>
      %extracted_slice = tensor.extract_slice %0[%arg2, 1] [1, %c63] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %inserted[%arg2, 1] [1, %c63] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %inserted_0[%arg2] [1] [1] : tensor<?xf32> to tensor<f32>
      %5:2 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_1, %extracted_slice_2 : tensor<?xf32>, tensor<f32>) {
      ^bb0(%in: f32, %out: f32, %out_4: f32):
        %6 = arith.cmpf ogt, %out_4, %in : f32
        %7 = arith.select %6, %out_4, %in : f32
        %8 = arith.subf %out_4, %in : f32
        %9 = arith.cmpf olt, %8, %cst : f32
        %10 = arith.negf %8 : f32
        %11 = arith.select %9, %10, %8 : f32
        %12 = arith.negf %11 : f32
        %13 = math.exp %12 : f32
        %14 = math.log1p %13 : f32
        %15 = arith.addf %7, %14 : f32
        linalg.yield %15, %15 : f32, f32
      } -> (tensor<?xf32>, tensor<f32>)
      %inserted_slice = tensor.insert_slice %5#0 into %inserted[%arg2, 1] [1, %c63] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      %inserted_slice_3 = tensor.insert_slice %5#1 into %inserted_0[%arg2] [1] [1] : tensor<f32> into tensor<?xf32>
      affine.yield %inserted_slice, %inserted_slice_3 : tensor<?x64xf32>, tensor<?xf32>
    }
    %4 = bufferization.to_memref %3#0 : memref<?x64xf32>
    memref.copy %4, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
  func.func private @log1pf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

