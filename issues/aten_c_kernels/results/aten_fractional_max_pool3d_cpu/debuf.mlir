#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fractional_max_pool3d_cpu(%arg0: memref<?x2x8x9x10xf32>, %arg1: memref<?x2x3xf32>, %arg2: memref<?x2x3x4x5xf32>, %arg3: memref<?x2x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x2x3x4x5xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x2x3x4x5xf32>
    %2:2 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %1, %arg6 = %0) -> (tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>) {
      %5:2 = affine.for %arg7 = 0 to 3 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>) {
        %6:2 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>) {
          %alloca = memref.alloca(%c5) : memref<?xi32>
          %7 = bufferization.to_tensor %alloca : memref<?xi32>
          %alloca_0 = memref.alloca(%c5) : memref<?xf32>
          %8 = bufferization.to_tensor %alloca_0 : memref<?xf32>
          %extracted_slice = tensor.extract_slice %arg11[0, %arg4, %arg7, %arg10, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : tensor<?x2x3x4x5xf32> to tensor<?xf32>
          %extracted_slice_1 = tensor.extract_slice %8[0] [%c5] [1] : tensor<?xf32> to tensor<?xf32>
          %9 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_1 : tensor<?xf32>) outs(%extracted_slice : tensor<?xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<?xf32>
          %inserted_slice = tensor.insert_slice %9 into %arg11[0, %arg4, %arg7, %arg10, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : tensor<?xf32> into tensor<?x2x3x4x5xf32>
          %extracted_slice_2 = tensor.extract_slice %arg12[0, %arg4, %arg7, %arg10, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : tensor<?x2x3x4x5xi32> to tensor<?xi32>
          %extracted_slice_3 = tensor.extract_slice %7[0] [%c5] [1] : tensor<?xi32> to tensor<?xi32>
          %10 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_3 : tensor<?xi32>) outs(%extracted_slice_2 : tensor<?xi32>) {
          ^bb0(%in: i32, %out: i32):
            linalg.yield %in : i32
          } -> tensor<?xi32>
          %inserted_slice_4 = tensor.insert_slice %10 into %arg12[0, %arg4, %arg7, %arg10, 0] [1, 1, 1, 1, %c5] [1, 1, 1, 1, 1] : tensor<?xi32> into tensor<?x2x3x4x5xi32>
          affine.yield %inserted_slice, %inserted_slice_4 : tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>
        }
        affine.yield %6#0, %6#1 : tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>
      }
      affine.yield %5#0, %5#1 : tensor<?x2x3x4x5xf32>, tensor<?x2x3x4x5xi32>
    }
    %3 = bufferization.to_memref %2#1 : memref<?x2x3x4x5xi32>
    memref.copy %3, %arg3 : memref<?x2x3x4x5xi32> to memref<?x2x3x4x5xi32>
    %4 = bufferization.to_memref %2#0 : memref<?x2x3x4x5xf32>
    memref.copy %4, %arg2 : memref<?x2x3x4x5xf32> to memref<?x2x3x4x5xf32>
    return
  }
}

