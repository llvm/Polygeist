#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_grid_sampler_2d_backward_cpu(%arg0: memref<?x3x8x8xf32>, %arg1: memref<?x6x6x2xf32>, %arg2: memref<?x3x6x6xf32>, %arg3: memref<?x3x8x8xf32>, %arg4: memref<?x6x6x2xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 7.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x6x6x2xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?x3x6x6xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?x6x6x2xf32>
    %3 = bufferization.to_tensor %arg0 : memref<?x3x8x8xf32>
    %4 = "polygeist.memref2pointer"(%arg3) : (memref<?x3x8x8xf32>) -> !llvm.ptr
    affine.for %arg5 = 0 to 192 {
      %7 = arith.index_cast %arg5 : index to i32
      %8 = llvm.getelementptr %4[%7] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %cst, %8 : f32, !llvm.ptr
    }
    %5 = affine.for %arg5 = 0 to 6 iter_args(%arg6 = %0) -> (tensor<?x6x6x2xf32>) {
      %alloca = memref.alloca(%c6) : memref<?xf32>
      %7 = bufferization.to_tensor %alloca : memref<?xf32>
      %8 = bufferization.to_tensor %alloca : memref<?xf32>
      %alloca_3 = memref.alloca(%c6) : memref<?xf32>
      %9 = bufferization.to_tensor %alloca_3 : memref<?xf32>
      %10 = bufferization.to_tensor %alloca_3 : memref<?xf32>
      %11 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%8 : tensor<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<?xf32>
      %12 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%10 : tensor<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      } -> tensor<?xf32>
      %13:2 = affine.for %arg7 = 0 to 6 iter_args(%arg8 = %11, %arg9 = %12) -> (tensor<?xf32>, tensor<?xf32>) {
        %extracted = tensor.extract %2[%c0, %arg5, %arg7, %c0] : tensor<?x6x6x2xf32>
        %16 = arith.addf %extracted, %cst_0 : f32
        %17 = arith.mulf %16, %cst_1 : f32
        %18 = arith.mulf %17, %cst_2 : f32
        %extracted_8 = tensor.extract %2[%c0, %arg5, %arg7, %c1] : tensor<?x6x6x2xf32>
        %19 = arith.addf %extracted_8, %cst_0 : f32
        %20 = arith.mulf %19, %cst_1 : f32
        %21 = arith.mulf %20, %cst_2 : f32
        %22 = arith.fptosi %18 : f32 to i32
        %23 = arith.fptosi %21 : f32 to i32
        %24 = arith.addi %22, %c1_i32 : i32
        %25 = arith.addi %23, %c1_i32 : i32
        %26 = arith.sitofp %22 : i32 to f32
        %27 = arith.subf %18, %26 : f32
        %28 = arith.sitofp %23 : i32 to f32
        %29 = arith.subf %21, %28 : f32
        %30 = arith.cmpi sge, %22, %c0_i32 : i32
        %31 = arith.cmpi slt, %22, %c8_i32 : i32
        %32 = arith.cmpi sge, %23, %c0_i32 : i32
        %33 = arith.cmpi slt, %23, %c8_i32 : i32
        %34 = arith.andi %32, %33 : i1
        %35 = arith.andi %31, %34 : i1
        %36 = arith.andi %30, %35 : i1
        %37 = arith.cmpi sge, %24, %c0_i32 : i32
        %38 = arith.cmpi slt, %24, %c8_i32 : i32
        %39 = arith.andi %38, %34 : i1
        %40 = arith.andi %37, %39 : i1
        %41 = arith.cmpi sge, %25, %c0_i32 : i32
        %42 = arith.cmpi slt, %25, %c8_i32 : i32
        %43 = arith.andi %41, %42 : i1
        %44 = arith.andi %31, %43 : i1
        %45 = arith.andi %30, %44 : i1
        %46 = arith.andi %38, %43 : i1
        %47 = arith.andi %37, %46 : i1
        %48 = arith.subf %cst_0, %29 : f32
        %49 = arith.subf %cst_0, %27 : f32
        %50 = arith.index_cast %23 : i32 to index
        %51 = arith.index_cast %22 : i32 to index
        %52 = arith.index_cast %24 : i32 to index
        %53 = arith.index_cast %25 : i32 to index
        %54:2 = affine.for %arg10 = 0 to 3 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<?xf32>, tensor<?xf32>) {
          %extracted_9 = tensor.extract %arg11[%arg7] : tensor<?xf32>
          %extracted_10 = tensor.extract %arg12[%arg7] : tensor<?xf32>
          %extracted_11 = tensor.extract %1[%c0, %arg10, %arg5, %arg7] : tensor<?x3x6x6xf32>
          %55 = scf.if %36 -> (f32) {
            %extracted_13 = tensor.extract %3[%c0, %arg10, %50, %51] : tensor<?x3x8x8xf32>
            %73 = arith.mulf %extracted_11, %49 : f32
            %74 = arith.mulf %73, %48 : f32
            %75 = memref.load %arg3[%c0, %arg10, %50, %51] : memref<?x3x8x8xf32>
            %76 = arith.addf %75, %74 : f32
            memref.store %76, %arg3[%c0, %arg10, %50, %51] : memref<?x3x8x8xf32>
            scf.yield %extracted_13 : f32
          } else {
            scf.yield %cst : f32
          }
          %56 = scf.if %40 -> (f32) {
            %extracted_13 = tensor.extract %3[%c0, %arg10, %50, %52] : tensor<?x3x8x8xf32>
            %73 = arith.mulf %extracted_11, %27 : f32
            %74 = arith.mulf %73, %48 : f32
            %75 = memref.load %arg3[%c0, %arg10, %50, %52] : memref<?x3x8x8xf32>
            %76 = arith.addf %75, %74 : f32
            memref.store %76, %arg3[%c0, %arg10, %50, %52] : memref<?x3x8x8xf32>
            scf.yield %extracted_13 : f32
          } else {
            scf.yield %cst : f32
          }
          %57 = scf.if %45 -> (f32) {
            %extracted_13 = tensor.extract %3[%c0, %arg10, %53, %51] : tensor<?x3x8x8xf32>
            %73 = arith.mulf %extracted_11, %49 : f32
            %74 = arith.mulf %73, %29 : f32
            %75 = memref.load %arg3[%c0, %arg10, %53, %51] : memref<?x3x8x8xf32>
            %76 = arith.addf %75, %74 : f32
            memref.store %76, %arg3[%c0, %arg10, %53, %51] : memref<?x3x8x8xf32>
            scf.yield %extracted_13 : f32
          } else {
            scf.yield %cst : f32
          }
          %58 = scf.if %47 -> (f32) {
            %extracted_13 = tensor.extract %3[%c0, %arg10, %53, %52] : tensor<?x3x8x8xf32>
            %73 = arith.mulf %extracted_11, %27 : f32
            %74 = arith.mulf %73, %29 : f32
            %75 = memref.load %arg3[%c0, %arg10, %53, %52] : memref<?x3x8x8xf32>
            %76 = arith.addf %75, %74 : f32
            memref.store %76, %arg3[%c0, %arg10, %53, %52] : memref<?x3x8x8xf32>
            scf.yield %extracted_13 : f32
          } else {
            scf.yield %cst : f32
          }
          %59 = arith.subf %56, %55 : f32
          %60 = arith.mulf %59, %48 : f32
          %61 = arith.subf %58, %57 : f32
          %62 = arith.mulf %61, %29 : f32
          %63 = arith.addf %60, %62 : f32
          %64 = arith.mulf %extracted_11, %63 : f32
          %65 = arith.addf %extracted_10, %64 : f32
          %66 = arith.subf %57, %55 : f32
          %67 = arith.mulf %66, %49 : f32
          %68 = arith.subf %58, %56 : f32
          %69 = arith.mulf %68, %27 : f32
          %70 = arith.addf %67, %69 : f32
          %71 = arith.mulf %extracted_11, %70 : f32
          %72 = arith.addf %extracted_9, %71 : f32
          %inserted = tensor.insert %72 into %arg11[%arg7] : tensor<?xf32>
          %inserted_12 = tensor.insert %65 into %arg12[%arg7] : tensor<?xf32>
          affine.yield %inserted, %inserted_12 : tensor<?xf32>, tensor<?xf32>
        }
        affine.yield %54#0, %54#1 : tensor<?xf32>, tensor<?xf32>
      }
      %extracted_slice = tensor.extract_slice %arg6[0, %arg5, 0, 0] [1, 1, %c6, 1] [1, 1, 1, 1] : tensor<?x6x6x2xf32> to tensor<?xf32>
      %extracted_slice_4 = tensor.extract_slice %9[0] [%c6] [1] : tensor<?xf32> to tensor<?xf32>
      %14 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_4 : tensor<?xf32>) outs(%extracted_slice : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.mulf %in, %cst_1 : f32
        %17 = arith.mulf %16, %cst_2 : f32
        linalg.yield %17 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %14 into %arg6[0, %arg5, 0, 0] [1, 1, %c6, 1] [1, 1, 1, 1] : tensor<?xf32> into tensor<?x6x6x2xf32>
      %extracted_slice_5 = tensor.extract_slice %inserted_slice[0, %arg5, 0, 1] [1, 1, %c6, 1] [1, 1, 1, 1] : tensor<?x6x6x2xf32> to tensor<?xf32>
      %extracted_slice_6 = tensor.extract_slice %7[0] [%c6] [1] : tensor<?xf32> to tensor<?xf32>
      %15 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%extracted_slice_6 : tensor<?xf32>) outs(%extracted_slice_5 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %16 = arith.mulf %in, %cst_1 : f32
        %17 = arith.mulf %16, %cst_2 : f32
        linalg.yield %17 : f32
      } -> tensor<?xf32>
      %inserted_slice_7 = tensor.insert_slice %15 into %inserted_slice[0, %arg5, 0, 1] [1, 1, %c6, 1] [1, 1, 1, 1] : tensor<?xf32> into tensor<?x6x6x2xf32>
      affine.yield %inserted_slice_7 : tensor<?x6x6x2xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?x6x6x2xf32>
    memref.copy %6, %arg4 : memref<?x6x6x2xf32> to memref<?x6x6x2xf32>
    return
  }
}

