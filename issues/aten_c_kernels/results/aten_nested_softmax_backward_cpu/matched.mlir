#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_softmax_backward_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %5 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %2) -> (tensor<?x64xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %7 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %7[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %4[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %3[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %8 = kernel.launch @cublasSdot(%extracted_slice, %extracted_slice_0, %inserted) : (tensor<?xf32>, tensor<?xf32>, tensor<f32>) -> tensor<f32>
      %extracted = tensor.extract %8[] : tensor<f32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %1[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_3 = tensor.extract_slice %0[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %v9_pw_single_pad_1 = arith.constant 0.0 : f32

      %v9_pw_single_pad_2 = arith.constant 0.0 : f32

      %v9_pw_single_pad_3 = arith.constant 0.0 : f32

      %v9_pw_single_pad_4 = arith.constant 0.0 : f32

      %v9_pw_single_pad_5 = arith.constant 0.0 : f32

      %v9_pw_single_pad_6 = arith.constant 0.0 : f32

      %v9_pw_single_pad_7 = arith.constant 0.0 : f32

      %9 = kernel.launch @cudnnPointwiseGraph_f32(%extracted_slice_3, %extracted_slice_2, %extracted_slice_3, %extracted_slice_3, %extracted_slice_1, %extracted, %v9_pw_single_pad_1, %v9_pw_single_pad_2, %v9_pw_single_pad_3, %v9_pw_single_pad_4, %v9_pw_single_pad_5, %v9_pw_single_pad_6, %v9_pw_single_pad_7) {pointwise_graph = array<i64: 144128382265787392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>, pointwise_num_nodes = 2 : i64} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, f32, f32, f32, f32, f32, f32, f32, f32) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg4[%arg3, 0] [1, %c64] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      affine.yield %inserted_slice : tensor<?x64xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?x64xf32>
    memref.copy %6, %arg2 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

