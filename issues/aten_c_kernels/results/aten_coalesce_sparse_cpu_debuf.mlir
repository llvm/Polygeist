module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_coalesce_sparse_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xi32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xi32>
    %3 = bufferization.to_tensor %arg1 : memref<?xf32>
    %4 = bufferization.to_tensor %arg0 : memref<?xi32>
    %inserted = tensor.insert %c0_i32 into %0[%c0] : tensor<?xi32>
    %5:3 = affine.for %arg5 = 0 to 512 iter_args(%arg6 = %2, %arg7 = %1, %arg8 = %inserted) -> (tensor<?xi32>, tensor<?xf32>, tensor<?xi32>) {
      %extracted = tensor.extract %arg8[%c0] : tensor<?xi32>
      %9 = arith.index_cast %arg5 : index to i32
      %10 = arith.cmpi ne, %9, %c0_i32 : i32
      %extracted_0 = tensor.extract %4[%arg5] : tensor<?xi32>
      %11 = arith.addi %extracted, %c-1_i32 : i32
      %12 = arith.index_cast %11 : i32 to index
      %extracted_1 = tensor.extract %arg6[%12] : tensor<?xi32>
      %13 = arith.cmpi eq, %extracted_0, %extracted_1 : i32
      %14 = arith.select %10, %13, %false : i1
      %15:3 = scf.if %14 -> (i32, tensor<?xi32>, tensor<?xf32>) {
        %16 = arith.addi %extracted, %c-1_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %extracted_3 = tensor.extract %3[%arg5] : tensor<?xf32>
        %extracted_4 = tensor.extract %arg7[%17] : tensor<?xf32>
        %18 = arith.addf %extracted_4, %extracted_3 : f32
        %inserted_5 = tensor.insert %18 into %arg7[%17] : tensor<?xf32>
        scf.yield %extracted, %arg6, %inserted_5 : i32, tensor<?xi32>, tensor<?xf32>
      } else {
        %16 = arith.index_cast %extracted : i32 to index
        %extracted_3 = tensor.extract %4[%arg5] : tensor<?xi32>
        %inserted_4 = tensor.insert %extracted_3 into %arg6[%16] : tensor<?xi32>
        %17 = arith.addi %extracted, %c1_i32 : i32
        %extracted_5 = tensor.extract %3[%arg5] : tensor<?xf32>
        %inserted_6 = tensor.insert %extracted_5 into %arg7[%16] : tensor<?xf32>
        scf.yield %17, %inserted_4, %inserted_6 : i32, tensor<?xi32>, tensor<?xf32>
      }
      %inserted_2 = tensor.insert %15#0 into %arg8[%c0] : tensor<?xi32>
      affine.yield %15#1, %15#2, %inserted_2 : tensor<?xi32>, tensor<?xf32>, tensor<?xi32>
    }
    %6 = bufferization.to_memref %5#2 : memref<?xi32>
    memref.copy %6, %arg4 : memref<?xi32> to memref<?xi32>
    %7 = bufferization.to_memref %5#1 : memref<?xf32>
    memref.copy %7, %arg3 : memref<?xf32> to memref<?xf32>
    %8 = bufferization.to_memref %5#0 : memref<?xi32>
    memref.copy %8, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

