#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (-d0 + 23)>
#map4 = affine_map<(d0) -> (-d0 + 9)>
#map5 = affine_map<(d0, d1) -> (d1 + 1)>
#map6 = affine_map<(d0) -> (-d0 + 8)>
#map7 = affine_map<(d0, d1) -> (d1 + 2)>
#map8 = affine_map<(d0, d1) -> (-d0 + 22)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_backward_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?x24x11xf32>, %arg4: memref<?xf32>, %arg5: memref<?x4x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c22 = arith.constant 22 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c-2 = arith.constant -2 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c10 = arith.constant 10 : index
    %c23 = arith.constant 23 : index
    %c9 = arith.constant 9 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x4x12xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x5xi32>
    %2 = bufferization.to_tensor %arg3 : memref<?x24x11xf32>
    %3 = bufferization.to_tensor %arg4 : memref<?xf32>
    %4 = bufferization.to_tensor %arg5 : memref<?x4x12xf32>
    %5 = tensor.empty() : tensor<24x11xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0] [%c24, %c4, %c12] [1, 1, 1] : tensor<?x4x12xf32> to tensor<?x?x?xf32>
    %extracted_slice_1 = tensor.extract_slice %3[0] [%c4] [1] : tensor<?xf32> to tensor<?xf32>
    %extracted_slice_2 = tensor.extract_slice %4[0, 0, 0] [%c24, %c4, %c12] [1, 1, 1] : tensor<?x4x12xf32> to tensor<?x?x?xf32>
    %6 = linalg.generic {doc = "", indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%extracted_slice, %extracted_slice_1 : tensor<?x?x?xf32>, tensor<?xf32>) outs(%extracted_slice_2 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %9 = math.exp %in : f32
      %10 = arith.mulf %9, %in_3 : f32
      linalg.yield %10 : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %6 into %4[0, 0, 0] [%c24, %c4, %c12] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x4x12xf32>
    %7:2 = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %5, %arg8 = %inserted_slice) -> (tensor<24x11xf32>, tensor<?x4x12xf32>) {
      %extracted_slice_3 = tensor.extract_slice %arg7[0, 0] [%c24, %c11] [1, 1] : tensor<24x11xf32> to tensor<?x?xf32>
      %9 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice_3 : tensor<?x?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<?x?xf32>
      %inserted_slice_4 = tensor.insert_slice %9 into %arg7[0, 0] [%c24, %c11] [1, 1] : tensor<?x?xf32> into tensor<24x11xf32>
      %inserted = tensor.insert %cst_0 into %inserted_slice_4[%c23, %c9] : tensor<24x11xf32>
      %inserted_5 = tensor.insert %cst_0 into %inserted[%c23, %c10] : tensor<24x11xf32>
      %10 = affine.for %arg9 = 0 to 23 iter_args(%arg10 = %inserted_5) -> (tensor<24x11xf32>) {
        %13 = arith.subi %c22, %arg9 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.addi %14, %c1_i32 : i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = affine.for %arg11 = 0 to 11 iter_args(%arg12 = %arg10) -> (tensor<24x11xf32>) {
          %18 = arith.index_cast %arg11 : index to i32
          %19 = arith.andi %18, %c1_i32 : i32
          %20 = arith.cmpi ne, %19, %c0_i32 : i32
          %21 = arith.cmpi slt, %arg11, %c0 : index
          %22 = arith.subi %c-1, %arg11 : index
          %23 = arith.select %21, %22, %arg11 : index
          %24 = arith.divsi %23, %c2 : index
          %25 = arith.subi %c-1, %24 : index
          %26 = arith.select %21, %25, %24 : index
          %extracted_7 = tensor.extract %1[%arg6, %26] : tensor<?x5xi32>
          %27 = arith.select %20, %extracted_7, %arg2 : i32
          %28 = affine.apply #map3(%arg9, %arg11)
          %extracted_8 = tensor.extract %arg12[%28, %arg11] : tensor<24x11xf32>
          %29 = arith.index_cast %27 : i32 to index
          %extracted_9 = tensor.extract %0[%16, %arg6, %29] : tensor<?x4x12xf32>
          %30 = math.exp %extracted_9 : f32
          %31 = arith.mulf %extracted_8, %30 : f32
          %32 = arith.addi %18, %c1_i32 : i32
          %33 = affine.apply #map4(%arg11)
          %34 = arith.cmpi sge, %33, %c0 : index
          %35 = arith.andi %32, %c1_i32 : i32
          %36 = arith.cmpi ne, %35, %c0_i32 : i32
          %37 = arith.addi %arg11, %c1 : index
          %38 = arith.cmpi slt, %37, %c0 : index
          %39 = arith.subi %c-2, %arg11 : index
          %40 = arith.select %38, %39, %37 : index
          %41 = arith.divsi %40, %c2 : index
          %42 = arith.subi %c-1, %41 : index
          %43 = arith.select %38, %42, %41 : index
          %extracted_10 = tensor.extract %1[%arg6, %43] : tensor<?x5xi32>
          %44 = arith.select %36, %extracted_10, %arg2 : i32
          %45 = affine.apply #map3(%arg9, %arg11)
          %46 = affine.apply #map5(%arg9, %arg11)
          %extracted_11 = tensor.extract %arg12[%45, %46] : tensor<24x11xf32>
          %47 = arith.index_cast %44 : i32 to index
          %extracted_12 = tensor.extract %0[%16, %arg6, %47] : tensor<?x4x12xf32>
          %48 = math.exp %extracted_12 : f32
          %49 = arith.mulf %extracted_11, %48 : f32
          %50 = arith.addf %31, %49 : f32
          %51 = arith.select %34, %50, %31 : f32
          %52 = arith.addi %18, %c2_i32 : i32
          %53 = affine.apply #map6(%arg11)
          %54 = arith.cmpi sge, %53, %c0 : index
          %55 = arith.andi %52, %c1_i32 : i32
          %56 = arith.cmpi ne, %55, %c0_i32 : i32
          %57 = arith.cmpi slt, %arg11, %c0 : index
          %58 = arith.subi %c-1, %arg11 : index
          %59 = arith.select %57, %58, %arg11 : index
          %60 = arith.divsi %59, %c2 : index
          %61 = arith.subi %c-1, %60 : index
          %62 = arith.select %57, %61, %60 : index
          %63 = arith.addi %62, %c1 : index
          %extracted_13 = tensor.extract %1[%arg6, %63] : tensor<?x5xi32>
          %64 = arith.select %56, %extracted_13, %arg2 : i32
          %65 = arith.cmpi ne, %27, %arg2 : i32
          %66 = arith.cmpi ne, %27, %64 : i32
          %67 = arith.andi %65, %66 : i1
          %68 = affine.apply #map3(%arg9, %arg11)
          %69 = affine.apply #map7(%arg9, %arg11)
          %extracted_14 = tensor.extract %arg12[%68, %69] : tensor<24x11xf32>
          %70 = arith.index_cast %64 : i32 to index
          %extracted_15 = tensor.extract %0[%16, %arg6, %70] : tensor<?x4x12xf32>
          %71 = math.exp %extracted_15 : f32
          %72 = arith.mulf %extracted_14, %71 : f32
          %73 = arith.addf %51, %72 : f32
          %74 = arith.select %67, %73, %51 : f32
          %75 = arith.select %54, %74, %51 : f32
          %76 = affine.apply #map8(%arg9, %arg11)
          %inserted_16 = tensor.insert %75 into %arg12[%76, %arg11] : tensor<24x11xf32>
          affine.yield %inserted_16 : tensor<24x11xf32>
        }
        affine.yield %17 : tensor<24x11xf32>
      }
      %extracted = tensor.extract %2[%arg6, %c23, %c10] : tensor<?x24x11xf32>
      %extracted_6 = tensor.extract %2[%arg6, %c23, %c9] : tensor<?x24x11xf32>
      %11 = arith.addf %extracted, %extracted_6 : f32
      %12 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %arg8) -> (tensor<?x4x12xf32>) {
        %13 = affine.for %arg11 = 0 to 11 iter_args(%arg12 = %arg10) -> (tensor<?x4x12xf32>) {
          %14 = arith.index_cast %arg11 : index to i32
          %15 = arith.andi %14, %c1_i32 : i32
          %16 = arith.cmpi ne, %15, %c0_i32 : i32
          %17 = arith.cmpi slt, %arg11, %c0 : index
          %18 = arith.subi %c-1, %arg11 : index
          %19 = arith.select %17, %18, %arg11 : index
          %20 = arith.divsi %19, %c2 : index
          %21 = arith.subi %c-1, %20 : index
          %22 = arith.select %17, %21, %20 : index
          %extracted_7 = tensor.extract %1[%arg6, %22] : tensor<?x5xi32>
          %23 = arith.select %16, %extracted_7, %arg2 : i32
          %24 = arith.index_cast %23 : i32 to index
          %extracted_8 = tensor.extract %3[%arg6] : tensor<?xf32>
          %extracted_9 = tensor.extract %2[%arg6, %arg9, %arg11] : tensor<?x24x11xf32>
          %25 = arith.mulf %extracted_8, %extracted_9 : f32
          %extracted_10 = tensor.extract %10[%arg9, %arg11] : tensor<24x11xf32>
          %26 = arith.mulf %25, %extracted_10 : f32
          %27 = arith.divf %26, %11 : f32
          %extracted_11 = tensor.extract %arg12[%arg9, %arg6, %24] : tensor<?x4x12xf32>
          %28 = arith.subf %extracted_11, %27 : f32
          %inserted_12 = tensor.insert %28 into %arg12[%arg9, %arg6, %24] : tensor<?x4x12xf32>
          affine.yield %inserted_12 : tensor<?x4x12xf32>
        }
        affine.yield %13 : tensor<?x4x12xf32>
      }
      affine.yield %10, %12 : tensor<24x11xf32>, tensor<?x4x12xf32>
    }
    %8 = bufferization.to_memref %7#1 : memref<?x4x12xf32>
    memref.copy %8, %arg5 : memref<?x4x12xf32> to memref<?x4x12xf32>
    return
  }
}

