module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_scatter_add_expanded_index_cpu(%arg0: memref<?x128xf32>, %arg1: memref<?x64xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg2 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x128xf32>
    %3 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %2) -> (tensor<?x128xf32>) {
      %5 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %arg4) -> (tensor<?x128xf32>) {
        %extracted = tensor.extract %1[%arg3, %arg5] : tensor<?x64xi32>
        %6 = arith.index_cast %extracted : i32 to index
        %extracted_0 = tensor.extract %0[%arg3, %arg5] : tensor<?x64xf32>
        %extracted_1 = tensor.extract %arg6[%arg3, %6] : tensor<?x128xf32>
        %7 = arith.addf %extracted_1, %extracted_0 : f32
        %inserted = tensor.insert %7 into %arg6[%arg3, %6] : tensor<?x128xf32>
        affine.yield %inserted : tensor<?x128xf32>
      }
      affine.yield %5 : tensor<?x128xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?x128xf32>
    memref.copy %4, %arg0 : memref<?x128xf32> to memref<?x128xf32>
    return
  }
}

