#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>, %arg7: memref<?x?xf64>, %arg8: memref<?x?xf64>, %arg9: memref<?x?xf64>, %arg10: memref<?x?xf64>, %arg11: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg7 restrict : memref<?x?xf64>
    %3 = bufferization.to_tensor %arg8 restrict : memref<?x?xf64>
    %4 = bufferization.to_tensor %arg9 restrict : memref<?x?xf64>
    %5 = bufferization.to_tensor %arg10 restrict : memref<?x?xf64>
    %6 = bufferization.to_tensor %arg11 restrict : memref<?x?xf64>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg4 : i32 to index
    %10 = arith.index_cast %arg3 : i32 to index
    %11 = arith.index_cast %arg0 : i32 to index
    %12 = kernel.launch @memset_zero_2D(%0) : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%11, %8] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %2[0, 0] [%8, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %12[0, 0] [%11, %7] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %13 = kernel.launch @cublasDgemm_simple(%extracted_slice, %extracted_slice_0, %extracted_slice_1) : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %13 into %12[0, 0] [%11, %7] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %14 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %14, %arg5 : memref<?x?xf64> to memref<?x?xf64>
    %15 = kernel.launch @memset_zero_2D(%3) : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %4[0, 0] [%7, %9] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %5[0, 0] [%9, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_4 = tensor.extract_slice %15[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %16 = kernel.launch @cublasDgemm_simple(%extracted_slice_2, %extracted_slice_3, %extracted_slice_4) : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %inserted_slice_5 = tensor.insert_slice %16 into %15[0, 0] [%7, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %17 = bufferization.to_memref %inserted_slice_5 : memref<?x?xf64>
    memref.copy %17, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    %18 = kernel.launch @memset_zero_2D(%6) : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %extracted_slice_6 = tensor.extract_slice %18[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %19 = kernel.launch @cublasDgemm_simple(%13, %16, %extracted_slice_6) : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %inserted_slice_7 = tensor.insert_slice %19 into %18[0, 0] [%11, %10] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %20 = bufferization.to_memref %inserted_slice_7 : memref<?x?xf64>
    memref.copy %20, %arg11 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

