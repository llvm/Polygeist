#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d1 + s0 * 20 + d0 * 5)>
#map3 = affine_map<(d0, d1, d2) -> (d0 + d1 * 56 + d2 * 8)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_lanczos2d_aa_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 3.000000e+00 : f32
    %cst_3 = arith.constant 3.14159274 : f32
    %cst_4 = arith.constant 3.28986812 : f32
    %cst_5 = arith.constant 6.250000e-01 : f32
    %cst_6 = arith.constant 0.571428597 : f32
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %4 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %5 = arith.index_cast %arg4 : index to i32
        %6 = arith.sitofp %5 : i32 to f32
        %7 = arith.addf %6, %cst_0 : f32
        %8 = arith.mulf %7, %cst_6 : f32
        %9 = arith.subf %8, %cst_0 : f32
        %10 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %arg5) -> (tensor<?xf32>) {
          %11 = arith.index_cast %arg6 : index to i32
          %12 = arith.sitofp %11 : i32 to f32
          %13 = arith.addf %12, %cst_0 : f32
          %14 = arith.mulf %13, %cst_5 : f32
          %15 = arith.subf %14, %cst_0 : f32
          %alloca = memref.alloca() : memref<f32>
          %16 = bufferization.to_tensor %alloca : memref<f32>
          %inserted = tensor.insert %cst_1 into %16[] : tensor<f32>
          %17 = polygeist.submap(%inserted, %c4, %c5) {map = #map} : (tensor<f32>, index, index) -> tensor<?x?xf32>
          %18 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["reduction", "reduction"], library_call = ""} outs(%17 : tensor<?x?xf32>) {
          ^bb0(%out: f32):
            %25 = linalg.index 0 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %9, %27 : f32
            %29 = arith.cmpf olt, %28, %cst_1 : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_2 : f32
            %33 = arith.cmpf oeq, %31, %cst_1 : f32
            %34 = arith.mulf %31, %cst_3 : f32
            %35 = arith.divf %34, %cst_2 : f32
            %36 = arith.mulf %31, %cst_4 : f32
            %37 = arith.mulf %36, %31 : f32
            %38 = linalg.index 1 : index
            %39 = arith.index_cast %38 : index to i32
            %40 = arith.sitofp %39 : i32 to f32
            %41 = arith.subf %15, %40 : f32
            %42 = arith.cmpf olt, %41, %cst_1 : f32
            %43 = arith.negf %41 : f32
            %44 = arith.select %42, %43, %41 : f32
            %45 = scf.if %32 -> (f32) {
              %51 = scf.if %33 -> (f32) {
                scf.yield %cst : f32
              } else {
                %52 = math.sin %34 : f32
                %53 = math.sin %35 : f32
                %54 = arith.mulf %52, %53 : f32
                %55 = arith.divf %54, %37 : f32
                scf.yield %55 : f32
              }
              scf.yield %51 : f32
            } else {
              scf.yield %cst_1 : f32
            }
            %46 = arith.cmpf olt, %44, %cst_2 : f32
            %47 = arith.cmpf oeq, %44, %cst_1 : f32
            %48 = scf.if %46 -> (f32) {
              %51 = scf.if %47 -> (f32) {
                scf.yield %cst : f32
              } else {
                %52 = arith.mulf %44, %cst_3 : f32
                %53 = math.sin %52 : f32
                %54 = arith.divf %52, %cst_2 : f32
                %55 = math.sin %54 : f32
                %56 = arith.mulf %53, %55 : f32
                %57 = arith.mulf %44, %cst_4 : f32
                %58 = arith.mulf %57, %44 : f32
                %59 = arith.divf %56, %58 : f32
                scf.yield %59 : f32
              }
              scf.yield %51 : f32
            } else {
              scf.yield %cst_1 : f32
            }
            %49 = arith.mulf %45, %48 : f32
            %50 = arith.addf %out, %49 : f32
            linalg.yield %50 : f32
          } -> tensor<?x?xf32>
          %19 = polygeist.submapInverse(%inserted, %18, %c4, %c5) {map = #map} : (tensor<f32>, tensor<?x?xf32>, index, index) -> tensor<f32>
          %extracted = tensor.extract %19[] : tensor<f32>
          %alloca_7 = memref.alloca() : memref<f32>
          %20 = bufferization.to_tensor %alloca_7 : memref<f32>
          %inserted_8 = tensor.insert %cst_1 into %20[] : tensor<f32>
          %21 = polygeist.submap(%1, %arg2, %c4, %c5) {map = #map2} : (tensor<?xf32>, index, index, index) -> tensor<?x?xf32>
          %22 = linalg.generic {doc = "", indexing_maps = [#map1, #map], iterator_types = ["reduction", "reduction"], library_call = ""} ins(%21 : tensor<?x?xf32>) outs(%inserted_8 : tensor<f32>) {
          ^bb0(%in: f32, %out: f32):
            %25 = linalg.index 0 : index
            %26 = arith.index_cast %25 : index to i32
            %27 = arith.sitofp %26 : i32 to f32
            %28 = arith.subf %9, %27 : f32
            %29 = arith.cmpf olt, %28, %cst_1 : f32
            %30 = arith.negf %28 : f32
            %31 = arith.select %29, %30, %28 : f32
            %32 = arith.cmpf olt, %31, %cst_2 : f32
            %33 = arith.cmpf oeq, %31, %cst_1 : f32
            %34 = arith.mulf %31, %cst_3 : f32
            %35 = arith.divf %34, %cst_2 : f32
            %36 = arith.mulf %31, %cst_4 : f32
            %37 = arith.mulf %36, %31 : f32
            %38 = linalg.index 1 : index
            %39 = arith.index_cast %38 : index to i32
            %40 = arith.sitofp %39 : i32 to f32
            %41 = arith.subf %15, %40 : f32
            %42 = arith.cmpf olt, %41, %cst_1 : f32
            %43 = arith.negf %41 : f32
            %44 = arith.select %42, %43, %41 : f32
            %45 = scf.if %32 -> (f32) {
              %52 = scf.if %33 -> (f32) {
                scf.yield %cst : f32
              } else {
                %53 = math.sin %34 : f32
                %54 = math.sin %35 : f32
                %55 = arith.mulf %53, %54 : f32
                %56 = arith.divf %55, %37 : f32
                scf.yield %56 : f32
              }
              scf.yield %52 : f32
            } else {
              scf.yield %cst_1 : f32
            }
            %46 = arith.cmpf olt, %44, %cst_2 : f32
            %47 = arith.cmpf oeq, %44, %cst_1 : f32
            %48 = scf.if %46 -> (f32) {
              %52 = scf.if %47 -> (f32) {
                scf.yield %cst : f32
              } else {
                %53 = arith.mulf %44, %cst_3 : f32
                %54 = math.sin %53 : f32
                %55 = arith.divf %53, %cst_2 : f32
                %56 = math.sin %55 : f32
                %57 = arith.mulf %54, %56 : f32
                %58 = arith.mulf %44, %cst_4 : f32
                %59 = arith.mulf %58, %44 : f32
                %60 = arith.divf %57, %59 : f32
                scf.yield %60 : f32
              }
              scf.yield %52 : f32
            } else {
              scf.yield %cst_1 : f32
            }
            %49 = arith.mulf %in, %45 : f32
            %50 = arith.mulf %49, %48 : f32
            %51 = arith.addf %out, %50 : f32
            linalg.yield %51 : f32
          } -> tensor<f32>
          %extracted_9 = tensor.extract %22[] : tensor<f32>
          %23 = arith.divf %extracted_9, %extracted : f32
          %24 = affine.apply #map3(%arg6, %arg2, %arg4)
          %inserted_10 = tensor.insert %23 into %arg7[%24] : tensor<?xf32>
          affine.yield %inserted_10 : tensor<?xf32>
        }
        affine.yield %10 : tensor<?xf32>
      }
      affine.yield %4 : tensor<?xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?xf32>
    memref.copy %3, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @sinf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}

