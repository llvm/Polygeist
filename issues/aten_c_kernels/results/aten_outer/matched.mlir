#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_outer(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?x24xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?x24xf64>
    %extracted_slice = tensor.extract_slice %2[0, 0] [%c32, %c24] [1, 1] : tensor<?x24xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %0[0] [%c32] [1] : tensor<?xf64> to tensor<?xf64>
    %extracted_slice_1 = tensor.extract_slice %1[0] [%c24] [1] : tensor<?xf64> to tensor<?xf64>
    %4 = kernel.launch @cublasDgemm_outer_product(%extracted_slice_0, %extracted_slice_1, %extracted_slice) : (tensor<?xf64>, tensor<?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %4 into %2[0, 0] [%c32, %c24] [1, 1] : tensor<?x?xf64> into tensor<?x24xf64>
    %5 = bufferization.to_memref %inserted_slice : memref<?x24xf64>
    memref.copy %5, %arg2 : memref<?x24xf64> to memref<?x24xf64>
    return
  }
}

