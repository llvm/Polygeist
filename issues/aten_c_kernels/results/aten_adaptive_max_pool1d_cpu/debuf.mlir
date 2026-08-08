module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool1d_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x7xf32>, %arg2: memref<?x7xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c7_i32 = arith.constant 7 : i32
    %c32_i32 = arith.constant 32 : i32
    %c7 = arith.constant 7 : index
    %c32 = arith.constant 32 : index
    %c-1 = arith.constant -1 : index
    %c38 = arith.constant 38 : index
    %c-39 = arith.constant -39 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x7xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?x7xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %3:2 = affine.for %arg3 = 0 to 4 iter_args(%arg4 = %1, %arg5 = %0) -> (tensor<?x7xf32>, tensor<?x7xi32>) {
      %6:2 = affine.for %arg6 = 0 to 7 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<?x7xf32>, tensor<?x7xi32>) {
        %7 = arith.index_cast %arg6 : index to i32
        %8 = arith.muli %7, %c32_i32 : i32
        %9 = arith.divsi %8, %c7_i32 : i32
        %10 = arith.muli %arg6, %c32 : index
        %11 = arith.cmpi slt, %10, %c0 : index
        %12 = arith.subi %c-1, %10 : index
        %13 = arith.select %11, %12, %10 : index
        %14 = arith.divsi %13, %c7 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %extracted = tensor.extract %2[%arg3, %16] : tensor<?x32xf32>
        %17 = arith.addi %16, %c1 : index
        %18 = arith.addi %10, %c38 : index
        %19 = arith.cmpi slt, %18, %c0 : index
        %20 = arith.subi %c-39, %10 : index
        %21 = arith.select %19, %20, %18 : index
        %22 = arith.divsi %21, %c7 : index
        %23 = arith.subi %c-1, %22 : index
        %24 = arith.select %19, %23, %22 : index
        %25:2 = scf.for %arg9 = %17 to %24 step %c1 iter_args(%arg10 = %extracted, %arg11 = %9) -> (f32, i32) {
          %26 = arith.index_cast %arg9 : index to i32
          %extracted_1 = tensor.extract %2[%arg3, %arg9] : tensor<?x32xf32>
          %27 = arith.cmpf ogt, %extracted_1, %arg10 : f32
          %28 = arith.select %27, %26, %arg11 : i32
          %29 = arith.select %27, %extracted_1, %arg10 : f32
          scf.yield %29, %28 : f32, i32
        }
        %inserted = tensor.insert %25#0 into %arg7[%arg3, %arg6] : tensor<?x7xf32>
        %inserted_0 = tensor.insert %25#1 into %arg8[%arg3, %arg6] : tensor<?x7xi32>
        affine.yield %inserted, %inserted_0 : tensor<?x7xf32>, tensor<?x7xi32>
      }
      affine.yield %6#0, %6#1 : tensor<?x7xf32>, tensor<?x7xi32>
    }
    %4 = bufferization.to_memref %3#1 : memref<?x7xi32>
    memref.copy %4, %arg2 : memref<?x7xi32> to memref<?x7xi32>
    %5 = bufferization.to_memref %3#0 : memref<?x7xf32>
    memref.copy %5, %arg1 : memref<?x7xf32> to memref<?x7xf32>
    return
  }
}

