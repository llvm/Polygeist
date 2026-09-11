#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 restrict : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg5 restrict : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg6 restrict : memref<?x?xf64>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = tensor.empty() : tensor<f64>
    %5 = llvm.mlir.undef : f64
    %inserted = tensor.insert %5 into %4[] : tensor<f64>
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.subi %6, %c1 : index
    %8 = arith.subi %6, %c1 : index
    %9:2 = affine.for %arg7 = 0 to %6 iter_args(%arg8 = %inserted, %arg9 = %0) -> (tensor<f64>, tensor<?x?xf64>) {
      %11:2 = affine.for %arg10 = 0 to %3 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<f64>, tensor<?x?xf64>) {
        %inserted_0 = tensor.insert %cst into %arg11[] : tensor<f64>
        %extracted_slice = tensor.extract_slice %2[%arg7, %arg10] [1, 1] [1, 1] : tensor<?x?xf64> to tensor<f64>
        %extracted_slice_1 = tensor.extract_slice %1[%arg7, 0] [1, %7] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_2 = tensor.extract_slice %arg12[0, %arg10] [%7, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %12 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_1 : tensor<f64>, tensor<?xf64>) outs(%extracted_slice_2 : tensor<?xf64>) {
        ^bb0(%in: f64, %in_9: f64, %out: f64):
          %20 = arith.mulf %arg2, %in : f64
          %21 = arith.mulf %20, %in_9 : f64
          %22 = arith.addf %out, %21 : f64
          %23 = linalg.index 0 : index
          %24 = arith.cmpi slt, %23, %arg7 : index
          %25 = arith.select %24, %22, %out : f64
          linalg.yield %25 : f64
        } -> tensor<?xf64>
        %inserted_slice = tensor.insert_slice %12 into %arg12[0, %arg10] [%7, 1] [1, 1] : tensor<?xf64> into tensor<?x?xf64>
        %extracted_slice_3 = tensor.extract_slice %2[0, %arg10] [%8, 1] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %extracted_slice_4 = tensor.extract_slice %1[%arg7, 0] [1, %8] [1, 1] : tensor<?x?xf64> to tensor<?xf64>
        %13 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?xf64>, tensor<?xf64>) outs(%inserted_0 : tensor<f64>) {
        ^bb0(%in: f64, %in_9: f64, %out: f64):
          %20 = arith.mulf %in, %in_9 : f64
          %21 = arith.addf %out, %20 : f64
          %22 = linalg.index 0 : index
          %23 = arith.cmpi slt, %22, %arg7 : index
          %24 = arith.select %23, %21, %out : f64
          linalg.yield %24 : f64
        } -> tensor<f64>
        %extracted = tensor.extract %inserted_slice[%arg7, %arg10] : tensor<?x?xf64>
        %14 = arith.mulf %arg3, %extracted : f64
        %extracted_5 = tensor.extract %2[%arg7, %arg10] : tensor<?x?xf64>
        %15 = arith.mulf %arg2, %extracted_5 : f64
        %extracted_6 = tensor.extract %1[%arg7, %arg7] : tensor<?x?xf64>
        %16 = arith.mulf %15, %extracted_6 : f64
        %17 = arith.addf %14, %16 : f64
        %extracted_7 = tensor.extract %13[] : tensor<f64>
        %18 = arith.mulf %arg2, %extracted_7 : f64
        %19 = arith.addf %17, %18 : f64
        %inserted_8 = tensor.insert %19 into %inserted_slice[%arg7, %arg10] : tensor<?x?xf64>
        affine.yield %13, %inserted_8 : tensor<f64>, tensor<?x?xf64>
      }
      affine.yield %11#0, %11#1 : tensor<f64>, tensor<?x?xf64>
    }
    %10 = bufferization.to_memref %9#1 : memref<?x?xf64>
    memref.copy %10, %arg4 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

