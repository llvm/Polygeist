#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_median_indices_cpu(%arg0: memref<?x63xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c31_i32 = arith.constant 31 : i32
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x63xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?x63xf32>
    %4 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %3) -> (tensor<?x63xf32>) {
      %alloca = memref.alloca(%c32) : memref<?xi32>
      %10 = bufferization.to_tensor %alloca : memref<?xi32>
      %11:2 = affine.for %arg5 = 0 to 32 iter_args(%arg6 = %10, %arg7 = %arg4) -> (tensor<?xi32>, tensor<?x63xf32>) {
        %12 = arith.index_cast %arg5 : index to i32
        %inserted = tensor.insert %12 into %arg6[%arg5] : tensor<?xi32>
        %13 = affine.for %arg8 = #map(%arg5) to 63 iter_args(%arg9 = %inserted) -> (tensor<?xi32>) {
          %extracted_5 = tensor.extract %arg9[%arg5] : tensor<?xi32>
          %15 = arith.index_cast %arg8 : index to i32
          %extracted_6 = tensor.extract %arg7[%arg3, %arg8] : tensor<?x63xf32>
          %16 = arith.index_cast %extracted_5 : i32 to index
          %extracted_7 = tensor.extract %arg7[%arg3, %16] : tensor<?x63xf32>
          %17 = arith.cmpf olt, %extracted_6, %extracted_7 : f32
          %18 = arith.select %17, %15, %extracted_5 : i32
          %inserted_8 = tensor.insert %18 into %arg9[%arg5] : tensor<?xi32>
          affine.yield %inserted_8 : tensor<?xi32>
        }
        %extracted = tensor.extract %13[%arg5] : tensor<?xi32>
        %extracted_1 = tensor.extract %arg7[%arg3, %arg5] : tensor<?x63xf32>
        %14 = arith.index_cast %extracted : i32 to index
        %extracted_2 = tensor.extract %arg7[%arg3, %14] : tensor<?x63xf32>
        %inserted_3 = tensor.insert %extracted_2 into %arg7[%arg3, %arg5] : tensor<?x63xf32>
        %inserted_4 = tensor.insert %extracted_1 into %inserted_3[%arg3, %14] : tensor<?x63xf32>
        affine.yield %13, %inserted_4 : tensor<?xi32>, tensor<?x63xf32>
      }
      affine.yield %11#1 : tensor<?x63xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?x63xf32>
    memref.copy %5, %arg0 : memref<?x63xf32> to memref<?x63xf32>
    %extracted_slice = tensor.extract_slice %2[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 31] [%c16, 1] [1, 1] : tensor<?x63xf32> to tensor<?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_0 : tensor<?xf32>) outs(%extracted_slice : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %6 into %2[0] [%c16] [1] : tensor<?xf32> into tensor<?xf32>
    %7 = bufferization.to_memref %inserted_slice : memref<?xf32>
    memref.copy %7, %arg1 : memref<?xf32> to memref<?xf32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%1 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c31_i32 : i32
    } -> tensor<?xi32>
    %9 = bufferization.to_memref %8 : memref<?xi32>
    memref.copy %9, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

