module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_binomial_transform_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?x32xf32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x32xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?xi32>
    %4 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %0) -> (tensor<?xi32>) {
      %6:2 = scf.while (%arg6 = %c0_i32, %arg7 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %7 = arith.cmpi slt, %arg6, %c32_i32 : i32
        %extracted = tensor.extract %3[%arg4] : tensor<?xi32>
        %8 = arith.cmpi slt, %arg6, %extracted : i32
        %9 = arith.index_cast %arg6 : i32 to index
        %extracted_0 = tensor.extract %1[%arg4, %9] : tensor<?x32xf32>
        %extracted_1 = tensor.extract %2[%arg4] : tensor<?xf32>
        %10 = arith.cmpf olt, %extracted_0, %extracted_1 : f32
        %11 = arith.extui %10 : i1 to i32
        %12 = arith.addi %arg7, %11 : i32
        %13 = arith.addi %arg6, %c1_i32 : i32
        %14 = arith.select %8, %13, %arg6 : i32
        %15 = arith.select %8, %12, %arg7 : i32
        %16 = arith.select %7, %8, %false : i1
        %17 = arith.select %7, %14, %arg6 : i32
        %18 = arith.select %7, %15, %arg7 : i32
        scf.condition(%16) %17, %18 : i32, i32
      } do {
      ^bb0(%arg6: i32, %arg7: i32):
        scf.yield %arg6, %arg7 : i32, i32
      }
      %inserted = tensor.insert %6#1 into %arg5[%arg4] : tensor<?xi32>
      affine.yield %inserted : tensor<?xi32>
    }
    %5 = bufferization.to_memref %4 : memref<?xi32>
    memref.copy %5, %arg3 : memref<?xi32> to memref<?xi32>
    return
  }
}

