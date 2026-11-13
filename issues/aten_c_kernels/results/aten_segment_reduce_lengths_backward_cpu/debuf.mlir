#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_segment_reduce_lengths_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: i32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg5 : memref<?xf32>
    %1 = bufferization.to_tensor %arg3 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg1 : memref<?xi32>
    %4 = bufferization.to_tensor %arg0 : memref<?xf32>
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %6[] : tensor<i32>
    %7:2 = affine.for %arg6 = 0 to 16 iter_args(%arg7 = %inserted, %arg8 = %0) -> (tensor<i32>, tensor<?xf32>) {
      %extracted = tensor.extract %arg7[] : tensor<i32>
      %9:3 = scf.while (%arg9 = %c0_i32, %arg10 = %extracted, %arg11 = %arg8) : (i32, i32, tensor<?xf32>) -> (i32, i32, tensor<?xf32>) {
        %extracted_1 = tensor.extract %3[%arg6] : tensor<?xi32>
        %10 = arith.cmpi slt, %arg9, %extracted_1 : i32
        scf.condition(%10) %arg10, %arg9, %arg11 : i32, i32, tensor<?xf32>
      } do {
      ^bb0(%arg9: i32, %arg10: i32, %arg11: tensor<?xf32>):
        %10 = arith.cmpi eq, %5, %c0 : index
        %extracted_1 = tensor.extract %1[%arg6] : tensor<?xf32>
        %11 = affine.apply #map()[%5]
        %12 = arith.cmpi eq, %11, %c0 : index
        %extracted_2 = tensor.extract %1[%arg6] : tensor<?xf32>
        %extracted_3 = tensor.extract %3[%arg6] : tensor<?xi32>
        %13 = arith.sitofp %extracted_3 : i32 to f32
        %14 = arith.divf %extracted_2, %13 : f32
        %15 = arith.index_cast %arg9 : i32 to index
        %extracted_4 = tensor.extract %4[%15] : tensor<?xf32>
        %extracted_5 = tensor.extract %2[%arg6] : tensor<?xf32>
        %16 = arith.cmpf oeq, %extracted_4, %extracted_5 : f32
        %extracted_6 = tensor.extract %1[%arg6] : tensor<?xf32>
        %17 = arith.select %16, %extracted_6, %cst : f32
        %18 = arith.select %12, %14, %17 : f32
        %19 = arith.select %10, %extracted_1, %18 : f32
        %20 = arith.addi %arg9, %c1_i32 : i32
        %21 = arith.index_cast %arg9 : i32 to index
        %inserted_7 = tensor.insert %19 into %arg11[%21] : tensor<?xf32>
        %22 = arith.addi %arg10, %c1_i32 : i32
        scf.yield %22, %20, %inserted_7 : i32, i32, tensor<?xf32>
      }
      %inserted_0 = tensor.insert %9#0 into %arg7[] : tensor<i32>
      affine.yield %inserted_0, %9#2 : tensor<i32>, tensor<?xf32>
    }
    %8 = bufferization.to_memref %7#1 : memref<?xf32>
    memref.copy %8, %arg5 : memref<?xf32> to memref<?xf32>
    return
  }
}

