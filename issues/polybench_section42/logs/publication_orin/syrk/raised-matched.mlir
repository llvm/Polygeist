#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_syrk(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %5 = arith.subi %3, %c1 : index
    %6 = affine.apply #map1(%5)
    %extracted_slice = tensor.extract_slice %1[0, 0] [%3, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0] [%6, %2] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0] [%3, %6] [1, 1] : tensor<?x?xf64> to tensor<?x?xf64>
    %7 = kernel.launch @cublasDsyrk(%extracted_slice, %0, %arg3, %arg2) : (tensor<?x?xf64>, tensor<?x?xf64>, f64, f64) -> tensor<?x?xf64>
    %inserted_slice = tensor.insert_slice %7 into %0[0, 0] [%3, %6] [1, 1] : tensor<?x?xf64> into tensor<?x?xf64>
    %8 = bufferization.to_memref %inserted_slice : memref<?x?xf64>
    memref.copy %8, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

