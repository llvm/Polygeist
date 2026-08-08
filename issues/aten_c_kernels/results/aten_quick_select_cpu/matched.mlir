#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_quick_select_cpu(%arg0: memref<?xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.apply #map()[%2]
    %4 = tensor.empty(%3) : tensor<?xi32>
    %5:2 = affine.for %arg3 = 0 to #map()[%2] iter_args(%arg4 = %4, %arg5 = %1) -> (tensor<?xi32>, tensor<?xf32>) {
      %8 = arith.index_cast %arg3 : index to i32
      %inserted_0 = tensor.insert %8 into %arg4[%arg3] : tensor<?xi32>
      %9 = affine.for %arg6 = #map1(%arg3) to 127 iter_args(%arg7 = %inserted_0) -> (tensor<?xi32>) {
        %extracted_6 = tensor.extract %arg7[%arg3] : tensor<?xi32>
        %11 = arith.index_cast %arg6 : index to i32
        %extracted_7 = tensor.extract %arg5[%arg6] : tensor<?xf32>
        %12 = arith.index_cast %extracted_6 : i32 to index
        %extracted_8 = tensor.extract %arg5[%12] : tensor<?xf32>
        %13 = arith.cmpf olt, %extracted_7, %extracted_8 : f32
        %14 = arith.select %13, %11, %extracted_6 : i32
        %inserted_9 = tensor.insert %14 into %arg7[%arg3] : tensor<?xi32>
        affine.yield %inserted_9 : tensor<?xi32>
      }
      %extracted_1 = tensor.extract %9[%arg3] : tensor<?xi32>
      %extracted_2 = tensor.extract %arg5[%arg3] : tensor<?xf32>
      %10 = arith.index_cast %extracted_1 : i32 to index
      %extracted_3 = tensor.extract %arg5[%10] : tensor<?xf32>
      %inserted_4 = tensor.insert %extracted_3 into %arg5[%arg3] : tensor<?xf32>
      %inserted_5 = tensor.insert %extracted_2 into %inserted_4[%10] : tensor<?xf32>
      affine.yield %9, %inserted_5 : tensor<?xi32>, tensor<?xf32>
    }
    %6 = bufferization.to_memref %5#1 : memref<?xf32>
    memref.copy %6, %arg0 : memref<?xf32> to memref<?xf32>
    %extracted = tensor.extract %5#1[%2] : tensor<?xf32>
    %inserted = tensor.insert %extracted into %0[%c0] : tensor<?xf32>
    %7 = bufferization.to_memref %inserted : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

