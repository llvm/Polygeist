#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dyn_quant_pack_4bit_weight_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x32xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.500000e+01 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c15_i32 = arith.constant 15 : i32
    %c4_i32 = arith.constant 4 : i32
    %false = arith.constant false
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c63 = arith.constant 63 : index
    %c48 = arith.constant 48 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x32xi8>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg3 : memref<?xf32>
    %4 = tensor.empty(%c48) : tensor<?xf32>
    %5 = tensor.empty(%c48) : tensor<?xf32>
    %6:5 = affine.for %arg4 = 0 to 48 iter_args(%arg5 = %4, %arg6 = %5, %arg7 = %2, %arg8 = %3, %arg9 = %1) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x32xi8>) {
      %extracted = tensor.extract %0[%arg4, %c0] : tensor<?x64xf32>
      %inserted = tensor.insert %extracted into %arg5[%arg4] : tensor<?xf32>
      %inserted_1 = tensor.insert %extracted into %arg6[%arg4] : tensor<?xf32>
      %extracted_slice = tensor.extract_slice %0[%arg4, 1] [1, %c63] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %inserted[%arg4] [1] [1] : tensor<?xf32> to tensor<f32>
      %extracted_slice_3 = tensor.extract_slice %inserted_1[%arg4] [1] [1] : tensor<?xf32> to tensor<f32>
      %10:2 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_2, %extracted_slice_3 : tensor<f32>, tensor<f32>) {
      ^bb0(%in: f32, %out: f32, %out_9: f32):
        %16 = arith.cmpf olt, %in, %out_9 : f32
        %17 = arith.select %16, %in, %out_9 : f32
        %18 = arith.cmpf ogt, %in, %out : f32
        %19 = arith.select %18, %in, %out : f32
        linalg.yield %19, %17 : f32, f32
      } -> (tensor<f32>, tensor<f32>)
      %inserted_slice = tensor.insert_slice %10#0 into %inserted[%arg4] [1] [1] : tensor<f32> into tensor<?xf32>
      %inserted_slice_4 = tensor.insert_slice %10#1 into %inserted_1[%arg4] [1] [1] : tensor<f32> into tensor<?xf32>
      %extracted_5 = tensor.extract %inserted_slice[%arg4] : tensor<?xf32>
      %extracted_6 = tensor.extract %inserted_slice_4[%arg4] : tensor<?xf32>
      %11 = arith.subf %extracted_5, %extracted_6 : f32
      %12 = arith.divf %11, %cst : f32
      %inserted_7 = tensor.insert %12 into %arg7[%arg4] : tensor<?xf32>
      %13 = arith.negf %extracted_6 : f32
      %14 = arith.divf %13, %12 : f32
      %inserted_8 = tensor.insert %14 into %arg8[%arg4] : tensor<?xf32>
      %15 = affine.for %arg10 = 0 to 64 step 2 iter_args(%arg11 = %arg9) -> (tensor<?x32xi8>) {
        %extracted_9 = tensor.extract %0[%arg4, %arg10] : tensor<?x64xf32>
        %extracted_10 = tensor.extract %inserted_7[%arg4] : tensor<?xf32>
        %16 = arith.divf %extracted_9, %extracted_10 : f32
        %extracted_11 = tensor.extract %inserted_8[%arg4] : tensor<?xf32>
        %17 = arith.addf %16, %extracted_11 : f32
        %18 = arith.addf %17, %cst_0 : f32
        %19 = arith.fptosi %18 : f32 to i32
        %20 = affine.apply #map2(%arg4, %arg10)
        %extracted_12 = tensor.extract %0[%arg4, %20] : tensor<?x64xf32>
        %21 = arith.divf %extracted_12, %extracted_10 : f32
        %22 = arith.addf %21, %extracted_11 : f32
        %23 = arith.addf %22, %cst_0 : f32
        %24 = arith.fptosi %23 : f32 to i32
        %25 = arith.cmpi slt, %19, %c0_i32 : i32
        %26 = arith.select %25, %c0_i32, %19 : i32
        %27 = arith.cmpi sgt, %19, %c15_i32 : i32
        %28 = arith.select %25, %false, %27 : i1
        %29 = arith.select %28, %c15_i32, %26 : i32
        %30 = arith.cmpi slt, %24, %c0_i32 : i32
        %31 = arith.select %30, %c0_i32, %24 : i32
        %32 = arith.cmpi sgt, %24, %c15_i32 : i32
        %33 = arith.select %30, %false, %32 : i1
        %34 = arith.select %33, %c15_i32, %31 : i32
        %35 = arith.shli %34, %c4_i32 : i32
        %36 = arith.ori %29, %35 : i32
        %37 = arith.trunci %36 : i32 to i8
        %38 = arith.cmpi slt, %arg10, %c0 : index
        %39 = arith.subi %c-1, %arg10 : index
        %40 = arith.select %38, %39, %arg10 : index
        %41 = arith.divsi %40, %c2 : index
        %42 = arith.subi %c-1, %41 : index
        %43 = arith.select %38, %42, %41 : index
        %inserted_13 = tensor.insert %37 into %arg11[%arg4, %43] : tensor<?x32xi8>
        affine.yield %inserted_13 : tensor<?x32xi8>
      }
      affine.yield %inserted_slice, %inserted_slice_4, %inserted_7, %inserted_8, %15 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x32xi8>
    }
    %7 = bufferization.to_memref %6#4 : memref<?x32xi8>
    memref.copy %7, %arg1 : memref<?x32xi8> to memref<?x32xi8>
    %8 = bufferization.to_memref %6#2 : memref<?xf32>
    memref.copy %8, %arg2 : memref<?xf32> to memref<?xf32>
    %9 = bufferization.to_memref %6#3 : memref<?xf32>
    memref.copy %9, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}

