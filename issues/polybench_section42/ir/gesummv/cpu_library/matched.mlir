#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = bufferization.to_tensor %arg3 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg5 restrict : memref<?xf64>
    %3 = bufferization.to_tensor %arg6 restrict : memref<?xf64>
    %4 = bufferization.to_tensor %arg7 restrict : memref<?xf64>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = kernel.launch @memset_zero_1D(%2) : (tensor<?xf64>) -> tensor<?xf64>
    %7 = kernel.launch @memset_zero_1D(%4) : (tensor<?xf64>) -> tensor<?xf64>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %6[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %8 = kernel.launch @cublasDgemv(%extracted_slice, %extracted_slice_0, %extracted_slice_1) : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %inserted_slice = tensor.insert_slice %8 into %6[0] [%5] [1] : tensor<?xf64> into tensor<?xf64>
    %9 = bufferization.to_memref %inserted_slice : memref<?xf64>
    memref.copy %9, %arg5 : memref<?xf64> to memref<?xf64>
    %extracted_slice_2 = tensor.extract_slice %1[0, 0] [%5, %5] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %3[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_4 = tensor.extract_slice %7[0] [%5] [1] : tensor<?xf64> to tensor<?xf64>
    %10 = kernel.launch @cublasDgemv(%extracted_slice_2, %extracted_slice_3, %extracted_slice_4) : (tensor<?x?xf64>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %inserted_slice_5 = tensor.insert_slice %10 into %7[0] [%5] [1] : tensor<?xf64> into tensor<?xf64>
    %11 = kernel.launch @cublasDaxpby(%inserted_slice, %inserted_slice_5, %arg1, %arg2) : (tensor<?xf64>, tensor<?xf64>, f64, f64) -> tensor<?xf64>
    %12 = bufferization.to_memref %11 : memref<?xf64>
    memref.copy %12, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}

