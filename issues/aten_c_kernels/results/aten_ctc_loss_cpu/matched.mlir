#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1 - 1)>
#map3 = affine_map<(d0) -> (d0 - 1)>
#map4 = affine_map<(d0, d1, d2) -> (d2 - 1)>
#map5 = affine_map<(d0, d1, d2) -> (d2 - 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: memref<?x24x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c-2_i32 = arith.constant -2 : i32
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c11 = arith.constant 11 : index
    %c24 = arith.constant 24 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x4x12xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x5xi32>
    %2 = bufferization.to_tensor %arg3 : memref<?xf32>
    %3 = bufferization.to_tensor %arg4 : memref<?x24x11xf32>
    %4 = arith.index_cast %arg2 : i32 to index
    %extracted_slice = tensor.extract_slice %3[0, 0, 0] [%c4, %c24, %c11] [1, 1, 1] : tensor<?x24x11xf32> to tensor<?x?x?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %3[0, 0, 0] [%c4, %c24, %c11] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x24x11xf32>
    %extracted_slice_0 = tensor.extract_slice %0[0, 0, %4] [1, %c4, 1] [1, 1, 1] : tensor<?x4x12xf32> to tensor<?xf32>
    %extracted_slice_1 = tensor.extract_slice %inserted_slice[0, 0, 0] [%c4, 1, 1] [1, 1, 1] : tensor<?x24x11xf32> to tensor<?xf32>
    %6 = kernel.launch @cutensorUnary_exp_f32(%extracted_slice_0, %extracted_slice_1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %inserted_slice_2 = tensor.insert_slice %6 into %inserted_slice[0, 0, 0] [%c4, 1, 1] [1, 1, 1] : tensor<?xf32> into tensor<?x24x11xf32>
    %7 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %inserted_slice_2) -> (tensor<?x24x11xf32>) {
      %extracted = tensor.extract %1[%arg5, %c0] : tensor<?x5xi32>
      %12 = arith.index_cast %extracted : i32 to index
      %extracted_7 = tensor.extract %0[%c0, %arg5, %12] : tensor<?x4x12xf32>
      %13 = math.exp %extracted_7 : f32
      %inserted = tensor.insert %13 into %arg6[%arg5, %c0, %c1] : tensor<?x24x11xf32>
      affine.yield %inserted : tensor<?x24x11xf32>
    }
    %8 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %7) -> (tensor<?x24x11xf32>) {
      %12 = affine.for %arg7 = 1 to 24 iter_args(%arg8 = %arg6) -> (tensor<?x24x11xf32>) {
        %13 = affine.for %arg9 = 0 to 11 iter_args(%arg10 = %arg8) -> (tensor<?x24x11xf32>) {
          %14 = arith.index_cast %arg9 : index to i32
          %15 = arith.andi %14, %c1_i32 : i32
          %16 = arith.cmpi ne, %15, %c0_i32 : i32
          %17 = arith.cmpi slt, %arg9, %c0 : index
          %18 = arith.subi %c-1, %arg9 : index
          %19 = arith.select %17, %18, %arg9 : index
          %20 = arith.divsi %19, %c2 : index
          %21 = arith.subi %c-1, %20 : index
          %22 = arith.select %17, %21, %20 : index
          %extracted = tensor.extract %1[%arg5, %22] : tensor<?x5xi32>
          %23 = arith.select %16, %extracted, %arg2 : i32
          %24 = affine.apply #map2(%arg5, %arg7, %arg9)
          %extracted_7 = tensor.extract %arg10[%arg5, %24, %arg9] : tensor<?x24x11xf32>
          %25 = affine.apply #map3(%arg9)
          %26 = arith.cmpi sge, %25, %c0 : index
          %27 = affine.apply #map2(%arg5, %arg7, %arg9)
          %28 = affine.apply #map4(%arg5, %arg7, %arg9)
          %extracted_8 = tensor.extract %arg10[%arg5, %27, %28] : tensor<?x24x11xf32>
          %29 = arith.addf %extracted_7, %extracted_8 : f32
          %30 = arith.select %26, %29, %extracted_7 : f32
          %31 = arith.cmpi sgt, %14, %c1_i32 : i32
          %32 = arith.cmpi ne, %23, %arg2 : i32
          %33 = arith.andi %31, %32 : i1
          %34 = arith.addi %14, %c-2_i32 : i32
          %35 = arith.andi %34, %c1_i32 : i32
          %36 = arith.cmpi ne, %35, %c0_i32 : i32
          %37 = arith.cmpi slt, %arg9, %c0 : index
          %38 = arith.subi %c-1, %arg9 : index
          %39 = arith.select %37, %38, %arg9 : index
          %40 = arith.divsi %39, %c2 : index
          %41 = arith.subi %c-1, %40 : index
          %42 = arith.select %37, %41, %40 : index
          %43 = arith.addi %42, %c-1 : index
          %extracted_9 = tensor.extract %1[%arg5, %43] : tensor<?x5xi32>
          %44 = arith.select %36, %extracted_9, %arg2 : i32
          %45 = arith.cmpi ne, %23, %44 : i32
          %46 = affine.apply #map2(%arg5, %arg7, %arg9)
          %47 = affine.apply #map5(%arg5, %arg7, %arg9)
          %extracted_10 = tensor.extract %arg10[%arg5, %46, %47] : tensor<?x24x11xf32>
          %48 = arith.addf %30, %extracted_10 : f32
          %49 = arith.select %45, %48, %30 : f32
          %50 = arith.select %33, %49, %30 : f32
          %51 = arith.index_cast %23 : i32 to index
          %extracted_11 = tensor.extract %0[%arg7, %arg5, %51] : tensor<?x4x12xf32>
          %52 = math.exp %extracted_11 : f32
          %53 = arith.mulf %50, %52 : f32
          %inserted = tensor.insert %53 into %arg10[%arg5, %arg7, %arg9] : tensor<?x24x11xf32>
          affine.yield %inserted : tensor<?x24x11xf32>
        }
        affine.yield %13 : tensor<?x24x11xf32>
      }
      affine.yield %12 : tensor<?x24x11xf32>
    }
    %9 = bufferization.to_memref %8 : memref<?x24x11xf32>
    memref.copy %9, %arg4 : memref<?x24x11xf32> to memref<?x24x11xf32>
    %extracted_slice_3 = tensor.extract_slice %8[0, 23, 10] [%c4, 1, 1] [1, 1, 1] : tensor<?x24x11xf32> to tensor<?xf32>
    %extracted_slice_4 = tensor.extract_slice %8[0, 23, 9] [%c4, 1, 1] [1, 1, 1] : tensor<?x24x11xf32> to tensor<?xf32>
    %extracted_slice_5 = tensor.extract_slice %2[0] [%c4] [1] : tensor<?xf32> to tensor<?xf32>
    %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_3, %extracted_slice_4 : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_5 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %12 = arith.addf %in, %in_7 : f32
      %13 = math.log %12 : f32
      %14 = arith.negf %13 : f32
      linalg.yield %14 : f32
    } -> tensor<?xf32>
    %inserted_slice_6 = tensor.insert_slice %10 into %2[0] [%c4] [1] : tensor<?xf32> into tensor<?xf32>
    %11 = bufferization.to_memref %inserted_slice_6 : memref<?xf32>
    memref.copy %11, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

