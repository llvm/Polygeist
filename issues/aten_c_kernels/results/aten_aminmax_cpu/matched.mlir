#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_aminmax_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4095 = arith.constant 4095 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = llvm.mlir.undef : f32
    %inserted = tensor.insert %4 into %3[] : tensor<f32>
    %5 = tensor.empty() : tensor<f32>
    %inserted_0 = tensor.insert %4 into %5[] : tensor<f32>
    %extracted = tensor.extract %0[%c0] : tensor<?xf32>
    %inserted_1 = tensor.insert %extracted into %inserted_0[] : tensor<f32>
    %extracted_2 = tensor.extract %0[%c0] : tensor<?xf32>
    %inserted_3 = tensor.insert %extracted_2 into %inserted[] : tensor<f32>
    %extracted_slice = tensor.extract_slice %0[1] [%c4095] [1] : tensor<?xf32> to tensor<?xf32>
    %6 = kernel.launch @cudnnReduceMin_f32(%extracted_slice, %inserted_1) : (tensor<?xf32>, tensor<f32>) -> tensor<f32>
    %extracted_slice_4 = tensor.extract_slice %0[1] [%c4095] [1] : tensor<?xf32> to tensor<?xf32>
    %7 = kernel.launch @cudnnReduceMax_f32(%extracted_slice_4, %inserted_3) : (tensor<?xf32>, tensor<f32>) -> tensor<f32>
    %extracted_5 = tensor.extract %6[] : tensor<f32>
    %inserted_6 = tensor.insert %extracted_5 into %1[%c0] : tensor<?xf32>
    %8 = bufferization.to_memref %inserted_6 : memref<?xf32>
    memref.copy %8, %arg1 : memref<?xf32> to memref<?xf32>
    %extracted_7 = tensor.extract %7[] : tensor<f32>
    %inserted_8 = tensor.insert %extracted_7 into %2[%c0] : tensor<?xf32>
    %9 = bufferization.to_memref %inserted_8 : memref<?xf32>
    memref.copy %9, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

