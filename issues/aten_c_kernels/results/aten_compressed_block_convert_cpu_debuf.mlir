module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_compressed_block_convert_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x16x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c-1 = arith.constant -1 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x16x4x4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %2 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %0) -> (tensor<?x16x4x4xf32>) {
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c4 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %arg2, %c4 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c4 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg3) -> (tensor<?x16x4x4xf32>) {
        %extracted = tensor.extract %1[%arg2, %arg4] : tensor<?x64xf32>
        %15 = arith.cmpi slt, %arg4, %c0 : index
        %16 = arith.subi %c-1, %arg4 : index
        %17 = arith.select %15, %16, %arg4 : index
        %18 = arith.divsi %17, %c4 : index
        %19 = arith.subi %c-1, %18 : index
        %20 = arith.select %15, %19, %18 : index
        %21 = arith.remsi %arg4, %c4 : index
        %22 = arith.cmpi slt, %21, %c0 : index
        %23 = arith.addi %21, %c4 : index
        %24 = arith.select %22, %23, %21 : index
        %inserted = tensor.insert %extracted into %arg5[%9, %20, %13, %24] : tensor<?x16x4x4xf32>
        affine.yield %inserted : tensor<?x16x4x4xf32>
      }
      affine.yield %14 : tensor<?x16x4x4xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x16x4x4xf32>
    memref.copy %3, %arg1 : memref<?x16x4x4xf32> to memref<?x16x4x4xf32>
    return
  }
}

