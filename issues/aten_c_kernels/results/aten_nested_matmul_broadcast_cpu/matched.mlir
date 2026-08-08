#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_matmul_broadcast_cpu(%arg0: memref<?x16x24xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x16x20xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c24 = arith.constant 24 : index
    %c20 = arith.constant 20 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x16x24xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x20xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x16x20xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0, 0] [%c8, %c16, %c20] [1, 1, 1] : tensor<?x16x20xf32> to tensor<?x?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0, 0] [%c8, %c16, %c24] [1, 1, 1] : tensor<?x16x24xf32> to tensor<?x?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %1[0, 0] [%c24, %c20] [1, 1] : tensor<?x20xf32> to tensor<?x?xf32>
    %4 = kernel.launch @cublasSgemm_strided_batched_broadcast_rhs(%extracted_slice_0, %extracted_slice_1, %extracted_slice) : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %4 into %2[0, 0, 0] [%c8, %c16, %c20] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x16x20xf32>
    %5 = bufferization.to_memref %inserted_slice : memref<?x16x20xf32>
    memref.copy %5, %arg2 : memref<?x16x20xf32> to memref<?x16x20xf32>
    return
  }
}

