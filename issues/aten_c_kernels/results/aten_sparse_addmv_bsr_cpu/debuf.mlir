#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sparse_addmv_bsr_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x4x4xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = bufferization.to_tensor %arg0 : memref<?xi32>
    %3 = affine.for %arg5 = 0 to 16 iter_args(%arg6 = %0) -> (tensor<?xf32>) {
      %5 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %arg6) -> (tensor<?xf32>) {
        %extracted = tensor.extract %2[%arg5] : tensor<?xi32>
        %6 = affine.apply #map(%arg5)
        %extracted_0 = tensor.extract %2[%6] : tensor<?xi32>
        %7 = arith.index_cast %extracted_0 : i32 to index
        %8 = arith.index_cast %extracted : i32 to index
        %9 = scf.for %arg9 = %8 to %7 step %c1 iter_args(%arg10 = %cst) -> (f32) {
          %extracted_1 = tensor.extract %1[%arg9] : tensor<?xi32>
          %11 = arith.muli %extracted_1, %c4_i32 : i32
          %alloca = memref.alloca() : memref<f32>
          %12 = bufferization.to_tensor %alloca : memref<f32>
          %inserted_2 = tensor.insert %arg10 into %12[] : tensor<f32>
          %13 = polygeist.submap(%inserted_2, %c4) {map = #map1} : (tensor<f32>, index) -> tensor<?xf32>
          %14 = linalg.generic {doc = "", indexing_maps = [#map2], iterator_types = ["reduction"], library_call = ""} outs(%13 : tensor<?xf32>) {
          ^bb0(%out: f32):
            %16 = linalg.index 0 : index
            %17 = arith.index_cast %16 : index to i32
            %18 = memref.load %arg2[%arg9, %arg7, %16] : memref<?x4x4xf32>
            %19 = arith.addi %11, %17 : i32
            %20 = arith.index_cast %19 : i32 to index
            %21 = memref.load %arg3[%20] : memref<?xf32>
            %22 = arith.mulf %18, %21 : f32
            %23 = arith.addf %out, %22 : f32
            linalg.yield %23 : f32
          } -> tensor<?xf32>
          %15 = polygeist.submapInverse(%inserted_2, %14, %c4) {map = #map1} : (tensor<f32>, tensor<?xf32>, index) -> tensor<f32>
          %extracted_3 = tensor.extract %15[] : tensor<f32>
          scf.yield %extracted_3 : f32
        }
        %10 = affine.apply #map3(%arg7, %arg5)
        %inserted = tensor.insert %9 into %arg8[%10] : tensor<?xf32>
        affine.yield %inserted : tensor<?xf32>
      }
      affine.yield %5 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg4 : memref<?xf32> to memref<?xf32>
    return
  }
}

