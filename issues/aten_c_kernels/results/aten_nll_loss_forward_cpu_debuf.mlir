module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nll_loss_forward_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?x16xf32>
    %inserted = tensor.insert %cst into %0[%c0] : tensor<?xf32>
    %5:2 = affine.for %arg5 = 0 to 32 iter_args(%arg6 = %1, %arg7 = %inserted) -> (tensor<?xf32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg7[%c0] : tensor<?xf32>
      %extracted_0 = tensor.extract %3[%arg5] : tensor<?xi32>
      %8 = arith.index_cast %extracted_0 : i32 to index
      %extracted_1 = tensor.extract %4[%arg5, %8] : tensor<?x16xf32>
      %9 = arith.negf %extracted_1 : f32
      %extracted_2 = tensor.extract %2[%8] : tensor<?xf32>
      %10 = arith.mulf %9, %extracted_2 : f32
      %inserted_3 = tensor.insert %10 into %arg6[%arg5] : tensor<?xf32>
      %extracted_4 = tensor.extract %2[%8] : tensor<?xf32>
      %11 = arith.addf %extracted, %extracted_4 : f32
      %inserted_5 = tensor.insert %11 into %arg7[%c0] : tensor<?xf32>
      affine.yield %inserted_3, %inserted_5 : tensor<?xf32>, tensor<?xf32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xf32>
    memref.copy %6, %arg4 : memref<?xf32> to memref<?xf32>
    %7 = bufferization.to_memref %5#0 : memref<?xf32>
    memref.copy %7, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

