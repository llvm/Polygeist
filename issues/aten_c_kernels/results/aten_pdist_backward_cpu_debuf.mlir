#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_pdist_backward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?xf32>, %arg2: memref<?x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c31_i32 = arith.constant 31 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %0 = bufferization.to_tensor %arg2 : memref<?x32xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %3 = tensor.empty() : tensor<120xf32>
    %4 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %3) -> (tensor<120xf32>) {
      %8 = arith.index_cast %arg3 : index to i32
      %9 = arith.subi %c31_i32, %8 : i32
      %10 = arith.muli %8, %9 : i32
      %11 = arith.divsi %10, %c2_i32 : i32
      %12 = affine.for %arg5 = #map(%arg3) to 16 iter_args(%arg6 = %arg4) -> (tensor<120xf32>) {
        %13 = arith.index_cast %arg5 : index to i32
        %14 = arith.subi %13, %8 : i32
        %15 = arith.addi %14, %c-1_i32 : i32
        %16 = arith.addi %11, %15 : i32
        %17 = arith.index_cast %16 : i32 to index
        %inserted = tensor.insert %cst into %arg6[%17] : tensor<120xf32>
        %alloca = memref.alloca() : memref<f32>
        %18 = bufferization.to_tensor %alloca : memref<f32>
        %inserted_0 = tensor.insert %cst into %18[] : tensor<f32>
        %19:2 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %inserted, %arg9 = %inserted_0) -> (tensor<120xf32>, tensor<f32>) {
          %extracted_2 = tensor.extract %arg9[] : tensor<f32>
          %extracted_3 = tensor.extract %2[%arg3, %arg7] : tensor<?x32xf32>
          %extracted_4 = tensor.extract %2[%arg5, %arg7] : tensor<?x32xf32>
          %21 = arith.subf %extracted_3, %extracted_4 : f32
          %22 = arith.mulf %21, %21 : f32
          %23 = arith.addf %extracted_2, %22 : f32
          %inserted_5 = tensor.insert %23 into %arg8[%17] : tensor<120xf32>
          %inserted_6 = tensor.insert %23 into %arg9[] : tensor<f32>
          affine.yield %inserted_5, %inserted_6 : tensor<120xf32>, tensor<f32>
        }
        %extracted = tensor.extract %19#0[%17] : tensor<120xf32>
        %20 = math.sqrt %extracted : f32
        %inserted_1 = tensor.insert %20 into %19#0[%17] : tensor<120xf32>
        affine.yield %inserted_1 : tensor<120xf32>
      }
      affine.yield %12 : tensor<120xf32>
    }
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c16, %c32] [1, 1] : tensor<?x32xf32> to tensor<?x?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %0[0, 0] [%c16, %c32] [1, 1] : tensor<?x?xf32> into tensor<?x32xf32>
    %6 = affine.for %arg3 = 0 to 16 iter_args(%arg4 = %inserted_slice) -> (tensor<?x32xf32>) {
      %8 = arith.index_cast %arg3 : index to i32
      %9 = arith.subi %c31_i32, %8 : i32
      %10 = arith.muli %8, %9 : i32
      %11 = arith.divsi %10, %c2_i32 : i32
      %12 = affine.for %arg5 = #map(%arg3) to 16 iter_args(%arg6 = %arg4) -> (tensor<?x32xf32>) {
        %13 = arith.index_cast %arg5 : index to i32
        %14 = arith.subi %13, %8 : i32
        %15 = arith.addi %14, %c-1_i32 : i32
        %16 = arith.addi %11, %15 : i32
        %17 = arith.index_cast %16 : i32 to index
        %extracted = tensor.extract %4[%17] : tensor<120xf32>
        %18 = arith.cmpf oeq, %extracted, %cst : f32
        %extracted_0 = tensor.extract %1[%17] : tensor<?xf32>
        %19 = arith.divf %extracted_0, %extracted : f32
        %20 = arith.select %18, %cst, %19 : f32
        %21 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %arg6) -> (tensor<?x32xf32>) {
          %extracted_1 = tensor.extract %2[%arg3, %arg7] : tensor<?x32xf32>
          %extracted_2 = tensor.extract %2[%arg5, %arg7] : tensor<?x32xf32>
          %22 = arith.subf %extracted_1, %extracted_2 : f32
          %23 = arith.mulf %20, %22 : f32
          %extracted_3 = tensor.extract %arg8[%arg3, %arg7] : tensor<?x32xf32>
          %24 = arith.addf %extracted_3, %23 : f32
          %inserted = tensor.insert %24 into %arg8[%arg3, %arg7] : tensor<?x32xf32>
          %extracted_4 = tensor.extract %inserted[%arg5, %arg7] : tensor<?x32xf32>
          %25 = arith.subf %extracted_4, %23 : f32
          %inserted_5 = tensor.insert %25 into %inserted[%arg5, %arg7] : tensor<?x32xf32>
          affine.yield %inserted_5 : tensor<?x32xf32>
        }
        affine.yield %21 : tensor<?x32xf32>
      }
      affine.yield %12 : tensor<?x32xf32>
    }
    %7 = bufferization.to_memref %6 : memref<?x32xf32>
    memref.copy %7, %arg2 : memref<?x32xf32> to memref<?x32xf32>
    return
  }
}

