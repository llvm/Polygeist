#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_kthvalue_cpu(%arg0: memref<?x63xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x63xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x63xf32>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %2) -> (tensor<?x63xf32>) {
      %8 = affine.apply #map()[%3]
      %alloca = memref.alloca(%8) : memref<?xi32>
      %9 = bufferization.to_tensor %alloca : memref<?xi32>
      %10:2 = affine.for %arg5 = 0 to #map()[%3] iter_args(%arg6 = %9, %arg7 = %arg4) -> (tensor<?xi32>, tensor<?x63xf32>) {
        %11 = arith.index_cast %arg5 : index to i32
        %inserted = tensor.insert %11 into %arg6[%arg5] : tensor<?xi32>
        %12 = affine.for %arg8 = #map1(%arg5) to 63 iter_args(%arg9 = %inserted) -> (tensor<?xi32>) {
          %extracted_5 = tensor.extract %arg9[%arg5] : tensor<?xi32>
          %14 = arith.index_cast %arg8 : index to i32
          %extracted_6 = tensor.extract %arg7[%arg3, %arg8] : tensor<?x63xf32>
          %15 = arith.index_cast %extracted_5 : i32 to index
          %extracted_7 = tensor.extract %arg7[%arg3, %15] : tensor<?x63xf32>
          %16 = arith.cmpf olt, %extracted_6, %extracted_7 : f32
          %17 = arith.select %16, %14, %extracted_5 : i32
          %inserted_8 = tensor.insert %17 into %arg9[%arg5] : tensor<?xi32>
          affine.yield %inserted_8 : tensor<?xi32>
        }
        %extracted = tensor.extract %12[%arg5] : tensor<?xi32>
        %extracted_1 = tensor.extract %arg7[%arg3, %arg5] : tensor<?x63xf32>
        %13 = arith.index_cast %extracted : i32 to index
        %extracted_2 = tensor.extract %arg7[%arg3, %13] : tensor<?x63xf32>
        %inserted_3 = tensor.insert %extracted_2 into %arg7[%arg3, %arg5] : tensor<?x63xf32>
        %inserted_4 = tensor.insert %extracted_1 into %inserted_3[%arg3, %13] : tensor<?x63xf32>
        affine.yield %12, %inserted_4 : tensor<?xi32>, tensor<?x63xf32>
      }
      affine.yield %10#1 : tensor<?x63xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?x63xf32>
    memref.copy %5, %arg0 : memref<?x63xf32> to memref<?x63xf32>
    %extracted_slice = tensor.extract_slice %1[0] [%c16] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, %3] [%c16, 1] [1, 1] : tensor<?x63xf32> to tensor<?xf32>
    %6 = kernel.launch @cudaCopy1D_f32_tensor(%extracted_slice_0, %extracted_slice) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %6 into %1[0] [%c16] [1] : tensor<?xf32> into tensor<?xf32>
    %7 = bufferization.to_memref %inserted_slice : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

