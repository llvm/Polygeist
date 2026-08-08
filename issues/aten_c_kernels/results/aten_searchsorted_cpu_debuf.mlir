module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_searchsorted_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = affine.for %arg3 = 0 to 128 iter_args(%arg4 = %0) -> (tensor<?xi32>) {
      %5:2 = scf.while (%arg5 = %c256_i32, %arg6 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %6 = arith.cmpi slt, %arg6, %arg5 : i32
        scf.condition(%6) %arg6, %arg5 : i32, i32
      } do {
      ^bb0(%arg5: i32, %arg6: i32):
        %6 = arith.addi %arg5, %arg6 : i32
        %7 = arith.divsi %6, %c2_i32 : i32
        %8 = arith.index_cast %7 : i32 to index
        %extracted = tensor.extract %2[%8] : tensor<?xf32>
        %extracted_0 = tensor.extract %1[%arg3] : tensor<?xf32>
        %9 = arith.cmpf olt, %extracted, %extracted_0 : f32
        %10 = arith.select %9, %arg6, %7 : i32
        %11 = arith.addi %7, %c1_i32 : i32
        %12 = arith.select %9, %11, %arg5 : i32
        scf.yield %10, %12 : i32, i32
      }
      %inserted = tensor.insert %5#0 into %arg4[%arg3] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

