module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_index_put_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %4 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %2) -> (tensor<?xf32>) {
      %6 = scf.if %3 -> (tensor<?xf32>) {
        %extracted = tensor.extract %1[%arg4] : tensor<?xi32>
        %7 = arith.index_cast %extracted : i32 to index
        %extracted_0 = tensor.extract %0[%arg4] : tensor<?xf32>
        %extracted_1 = tensor.extract %arg5[%7] : tensor<?xf32>
        %8 = arith.addf %extracted_1, %extracted_0 : f32
        %inserted = tensor.insert %8 into %arg5[%7] : tensor<?xf32>
        scf.yield %inserted : tensor<?xf32>
      } else {
        %extracted = tensor.extract %1[%arg4] : tensor<?xi32>
        %7 = arith.index_cast %extracted : i32 to index
        %extracted_0 = tensor.extract %0[%arg4] : tensor<?xf32>
        %inserted = tensor.insert %extracted_0 into %arg5[%7] : tensor<?xf32>
        scf.yield %inserted : tensor<?xf32>
      }
      affine.yield %6 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg0 : memref<?xf32> to memref<?xf32>
    return
  }
}

