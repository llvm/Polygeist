#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_host_softmax_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c63 = arith.constant 63 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %2 = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %0) -> (tensor<?x64xf32>) {
      %extracted = tensor.extract %1[%arg2, %c0] : tensor<?x64xf32>
      %alloca = memref.alloca() : memref<f32>
      %4 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %extracted into %4[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %1[%arg2, 1] [1, %c63] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %8 = kernel.launch @cudnnSoftmaxForwardOut_tensor(%extracted_slice_3, %extracted_slice_4) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg3[%arg2, 0] [1, %c64] [1, 1] : tensor<?xf32> into tensor<?x64xf32>
      affine.yield %inserted_slice : tensor<?x64xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x64xf32>
    memref.copy %3, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

