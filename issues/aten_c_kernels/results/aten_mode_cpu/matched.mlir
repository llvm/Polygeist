#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_mode_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xi32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?xf32>
    %3 = tensor.empty() : tensor<64xi32>
    %4 = tensor.empty() : tensor<64xf32>
    %alloca = memref.alloca() : memref<64xf32>
    %5 = bufferization.to_tensor %alloca : memref<64xf32>
    %extracted_slice = tensor.extract_slice %4[0] [%c64] [1] : tensor<64xf32> to tensor<?xf32>
    %extracted_slice_0 = tensor.extract_slice %2[0] [%c64] [1] : tensor<?xf32> to tensor<?xf32>
    %6 = kernel.launch @cudaCopy1D_f32_tensor(%extracted_slice_0, %extracted_slice) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %inserted_slice = tensor.insert_slice %6 into %4[0] [%c64] [1] : tensor<?xf32> into tensor<64xf32>
    %extracted_slice_1 = tensor.extract_slice %3[0] [%c64] [1] : tensor<64xi32> to tensor<?xi32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%extracted_slice_1 : tensor<?xi32>) {
    ^bb0(%out: i32):
      %16 = linalg.index 0 : index
      %17 = arith.index_cast %16 : index to i32
      linalg.yield %17 : i32
    } -> tensor<?xi32>
    %inserted_slice_2 = tensor.insert_slice %7 into %3[0] [%c64] [1] : tensor<?xi32> into tensor<64xi32>
    %8:2 = affine.for %arg3 = 1 to 64 iter_args(%arg4 = %inserted_slice_2, %arg5 = %inserted_slice) -> (tensor<64xi32>, tensor<64xf32>) {
      %16 = arith.index_cast %arg3 : index to i32
      %extracted_11 = tensor.extract %arg5[%arg3] : tensor<64xf32>
      %extracted_12 = tensor.extract %arg4[%arg3] : tensor<64xi32>
      %17 = arith.addi %16, %c-1_i32 : i32
      %18:3 = scf.while (%arg6 = %17, %arg7 = %arg4, %arg8 = %arg5) : (i32, tensor<64xi32>, tensor<64xf32>) -> (i32, tensor<64xi32>, tensor<64xf32>) {
        %21 = arith.cmpi sge, %arg6, %c0_i32 : i32
        %22:4 = scf.if %21 -> (i1, i32, tensor<64xi32>, tensor<64xf32>) {
          %23 = arith.index_cast %arg6 : i32 to index
          %extracted_15 = tensor.extract %arg8[%23] : tensor<64xf32>
          %24 = arith.cmpf ogt, %extracted_15, %extracted_11 : f32
          %25:3 = scf.if %24 -> (i32, tensor<64xi32>, tensor<64xf32>) {
            %26 = arith.addi %arg6, %c1_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %inserted_16 = tensor.insert %extracted_15 into %arg8[%27] : tensor<64xf32>
            %extracted_17 = tensor.extract %arg7[%23] : tensor<64xi32>
            %inserted_18 = tensor.insert %extracted_17 into %arg7[%27] : tensor<64xi32>
            %28 = arith.addi %arg6, %c-1_i32 : i32
            scf.yield %28, %inserted_18, %inserted_16 : i32, tensor<64xi32>, tensor<64xf32>
          } else {
            scf.yield %arg6, %arg7, %arg8 : i32, tensor<64xi32>, tensor<64xf32>
          }
          scf.yield %24, %25#0, %25#1, %25#2 : i1, i32, tensor<64xi32>, tensor<64xf32>
        } else {
          scf.yield %false, %arg6, %arg7, %arg8 : i1, i32, tensor<64xi32>, tensor<64xf32>
        }
        scf.condition(%22#0) %22#1, %22#2, %22#3 : i32, tensor<64xi32>, tensor<64xf32>
      } do {
      ^bb0(%arg6: i32, %arg7: tensor<64xi32>, %arg8: tensor<64xf32>):
        scf.yield %arg6, %arg7, %arg8 : i32, tensor<64xi32>, tensor<64xf32>
      }
      %19 = arith.addi %18#0, %c1_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %inserted_13 = tensor.insert %extracted_11 into %18#2[%20] : tensor<64xf32>
      %inserted_14 = tensor.insert %extracted_12 into %18#1[%20] : tensor<64xi32>
      affine.yield %inserted_14, %inserted_13 : tensor<64xi32>, tensor<64xf32>
    }
    %9 = tensor.empty() : tensor<i32>
    %inserted = tensor.insert %c0_i32 into %9[] : tensor<i32>
    %10 = tensor.empty() : tensor<i32>
    %inserted_3 = tensor.insert %c1_i32 into %10[] : tensor<i32>
    %11 = tensor.empty() : tensor<i32>
    %inserted_4 = tensor.insert %c1_i32 into %11[] : tensor<i32>
    %extracted_slice_5 = tensor.extract_slice %5[1] [%c63] [1] : tensor<64xf32> to tensor<?xf32>
    %extracted_slice_6 = tensor.extract_slice %5[0] [%c63] [1] : tensor<64xf32> to tensor<?xf32>
    %12:3 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map1, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_5, %extracted_slice_6 : tensor<?xf32>, tensor<?xf32>) outs(%inserted, %inserted_3, %inserted_4 : tensor<i32>, tensor<i32>, tensor<i32>) {
    ^bb0(%in: f32, %in_11: f32, %out: i32, %out_12: i32, %out_13: i32):
      %16 = linalg.index 0 : index
      %17 = arith.index_cast %16 : index to i32
      %18 = arith.cmpf oeq, %in, %in_11 : f32
      %19 = arith.addi %out_12, %c1_i32 : i32
      %20 = arith.select %18, %19, %c1_i32 : i32
      %21 = arith.cmpi sgt, %20, %out_13 : i32
      %22 = arith.select %21, %17, %out : i32
      %23 = arith.select %21, %20, %out_13 : i32
      linalg.yield %22, %20, %23 : i32, i32, i32
    } -> (tensor<i32>, tensor<i32>, tensor<i32>)
    %extracted = tensor.extract %12#0[] : tensor<i32>
    %13 = arith.index_cast %extracted : i32 to index
    %extracted_7 = tensor.extract %8#1[%13] : tensor<64xf32>
    %inserted_8 = tensor.insert %extracted_7 into %1[%c0] : tensor<?xf32>
    %14 = bufferization.to_memref %inserted_8 : memref<?xf32>
    memref.copy %14, %arg1 : memref<?xf32> to memref<?xf32>
    %extracted_9 = tensor.extract %8#0[%13] : tensor<64xi32>
    %inserted_10 = tensor.insert %extracted_9 into %0[%c0] : tensor<?xi32>
    %15 = bufferization.to_memref %inserted_10 : memref<?xi32>
    memref.copy %15, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}

