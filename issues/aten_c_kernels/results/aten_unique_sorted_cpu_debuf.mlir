#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_unique_sorted_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xi32>
    %3 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %2) -> (tensor<?xi32>) {
      %8 = affine.for %arg5 = #map(%arg3) to 1024 iter_args(%arg6 = %arg4) -> (tensor<?xi32>) {
        %extracted = tensor.extract %arg6[%arg5] : tensor<?xi32>
        %extracted_0 = tensor.extract %arg6[%arg3] : tensor<?xi32>
        %9 = arith.cmpi slt, %extracted, %extracted_0 : i32
        %10 = scf.if %9 -> (tensor<?xi32>) {
          %inserted_1 = tensor.insert %extracted into %arg6[%arg3] : tensor<?xi32>
          %inserted_2 = tensor.insert %extracted_0 into %inserted_1[%arg5] : tensor<?xi32>
          scf.yield %inserted_2 : tensor<?xi32>
        } else {
          scf.yield %arg6 : tensor<?xi32>
        }
        affine.yield %10 : tensor<?xi32>
      }
      affine.yield %8 : tensor<?xi32>
    }
    %4 = bufferization.to_memref %3 : memref<?xi32>
    memref.copy %4, %arg0 : memref<?xi32> to memref<?xi32>
    %inserted = tensor.insert %c0_i32 into %0[%c0] : tensor<?xi32>
    %5:2 = affine.for %arg3 = 0 to 1024 iter_args(%arg4 = %1, %arg5 = %inserted) -> (tensor<?xi32>, tensor<?xi32>) {
      %extracted = tensor.extract %arg5[%c0] : tensor<?xi32>
      %8 = arith.cmpi eq, %arg3, %c0 : index
      %extracted_0 = tensor.extract %3[%arg3] : tensor<?xi32>
      %9 = affine.apply #map1(%arg3)
      %extracted_1 = tensor.extract %3[%9] : tensor<?xi32>
      %10 = arith.cmpi ne, %extracted_0, %extracted_1 : i32
      %11 = arith.select %8, %true, %10 : i1
      %12:2 = scf.if %11 -> (i32, tensor<?xi32>) {
        %13 = arith.addi %extracted, %c1_i32 : i32
        %14 = arith.index_cast %extracted : i32 to index
        %extracted_3 = tensor.extract %3[%arg3] : tensor<?xi32>
        %inserted_4 = tensor.insert %extracted_3 into %arg4[%14] : tensor<?xi32>
        scf.yield %13, %inserted_4 : i32, tensor<?xi32>
      } else {
        scf.yield %extracted, %arg4 : i32, tensor<?xi32>
      }
      %inserted_2 = tensor.insert %12#0 into %arg5[%c0] : tensor<?xi32>
      affine.yield %12#1, %inserted_2 : tensor<?xi32>, tensor<?xi32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xi32>
    memref.copy %6, %arg2 : memref<?xi32> to memref<?xi32>
    %7 = bufferization.to_memref %5#0 : memref<?xi32>
    memref.copy %7, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}

