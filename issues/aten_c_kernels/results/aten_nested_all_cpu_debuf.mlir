module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_all_cpu(%arg0: memref<?x64xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?x64xi32>
    %3 = affine.for %arg3 = 0 to 8 iter_args(%arg4 = %0) -> (tensor<?xi32>) {
      %extracted = tensor.extract %1[%arg3] : tensor<?xi32>
      %5 = arith.index_cast %extracted : i32 to index
      %6 = scf.for %arg5 = %c0 to %5 step %c1 iter_args(%arg6 = %c1_i32) -> (i32) {
        %extracted_0 = tensor.extract %2[%arg3, %arg5] : tensor<?x64xi32>
        %7 = arith.cmpi ne, %extracted_0, %c0_i32 : i32
        %8 = arith.extui %7 : i1 to i32
        %9 = arith.andi %arg6, %8 : i32
        scf.yield %9 : i32
      }
      %inserted = tensor.insert %6 into %arg4[%arg3] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

