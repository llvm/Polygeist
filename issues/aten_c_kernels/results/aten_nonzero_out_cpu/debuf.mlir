module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nonzero_out_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?xi32>
    %2 = bufferization.to_tensor %arg1 : memref<?xi32>
    %3 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %inserted = tensor.insert %c0_i32 into %0[%c0] : tensor<?xi32>
    %4:3 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %2, %arg6 = %1, %arg7 = %inserted) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) {
      %8 = arith.index_cast %arg4 : index to i32
      %9:3 = affine.for %arg8 = 0 to 64 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) {
        %extracted = tensor.extract %arg11[%c0] : tensor<?xi32>
        %10 = arith.index_cast %arg8 : index to i32
        %extracted_0 = tensor.extract %3[%arg4, %arg8] : tensor<?x64xf32>
        %11 = arith.cmpf une, %extracted_0, %cst : f32
        %12:3 = scf.if %11 -> (i32, tensor<?xi32>, tensor<?xi32>) {
          %13 = arith.index_cast %extracted : i32 to index
          %inserted_2 = tensor.insert %8 into %arg9[%13] : tensor<?xi32>
          %14 = arith.addi %extracted, %c1_i32 : i32
          %inserted_3 = tensor.insert %10 into %arg10[%13] : tensor<?xi32>
          scf.yield %14, %inserted_2, %inserted_3 : i32, tensor<?xi32>, tensor<?xi32>
        } else {
          scf.yield %extracted, %arg9, %arg10 : i32, tensor<?xi32>, tensor<?xi32>
        }
        %inserted_1 = tensor.insert %12#0 into %arg11[%c0] : tensor<?xi32>
        affine.yield %12#1, %12#2, %inserted_1 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
      }
      affine.yield %9#0, %9#1, %9#2 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
    }
    %5 = bufferization.to_memref %4#2 : memref<?xi32>
    memref.copy %5, %arg3 : memref<?xi32> to memref<?xi32>
    %6 = bufferization.to_memref %4#1 : memref<?xi32>
    memref.copy %6, %arg2 : memref<?xi32> to memref<?xi32>
    %7 = bufferization.to_memref %4#0 : memref<?xi32>
    memref.copy %7, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

