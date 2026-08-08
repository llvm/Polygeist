#map = affine_map<(d0)[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0) -> (d0 * 2 + 2)>
#map4 = affine_map<(d0, d1, d2) -> (d0 + d1 * 9 + d2 * 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_avg_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c7 = arith.constant 7 : index
    %c-1 = arith.constant -1 : index
    %c42 = arith.constant 42 : index
    %c6 = arith.constant 6 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %4 = arith.muli %arg2, %c42 : index
      %5 = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %6 = bufferization.to_tensor %alloca : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        %7 = bufferization.to_tensor %alloca_0 : memref<?xf32>
        %8:3 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %arg5) -> (tensor<?xi32>, tensor<?xf32>, tensor<?xf32>) {
          %9 = arith.index_cast %arg6 : index to i32
          %10 = arith.muli %9, %c7_i32 : i32
          %11 = arith.divsi %10, %c3_i32 : i32
          %12 = arith.addi %9, %c1_i32 : i32
          %13 = arith.muli %12, %c7_i32 : i32
          %14 = arith.addi %13, %c2_i32 : i32
          %15 = arith.divsi %14, %c3_i32 : i32
          %16 = arith.index_cast %15 : i32 to index
          %17 = arith.index_cast %11 : i32 to index
          %18 = arith.subi %16, %17 : index
          %19 = arith.muli %arg6, %c7 : index
          %20 = arith.cmpi slt, %19, %c0 : index
          %21 = arith.subi %c-1, %19 : index
          %22 = arith.select %20, %21, %19 : index
          %23 = arith.divsi %22, %c3 : index
          %24 = arith.subi %c-1, %23 : index
          %25 = arith.select %20, %24, %23 : index
          %26 = arith.addi %25, %c3 : index
          %inserted = tensor.insert %c0_i32 into %arg7[%arg6] : tensor<?xi32>
          %inserted_1 = tensor.insert %cst into %arg8[%arg6] : tensor<?xf32>
          %27 = polygeist.submap(%inserted, %arg6, %c6) {map = #map} : (tensor<?xi32>, index, index) -> tensor<?xi32>
          %28 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["reduction"], library_call = ""} outs(%27 : tensor<?xi32>) {
          ^bb0(%out: i32):
            %34 = arith.index_cast %out : i32 to index
            %35 = arith.addi %34, %18 : index
            %36 = arith.index_cast %35 : index to i32
            %37 = linalg.index 0 : index
            %38 = affine.apply #map2(%arg4)
            %39 = arith.cmpi sge, %37, %38 : index
            %40 = affine.apply #map3(%arg4)
            %41 = arith.cmpi slt, %37, %40 : index
            %42 = arith.andi %39, %41 : i1
            %43 = arith.select %42, %36, %out : i32
            linalg.yield %43 : i32
          } -> tensor<?xi32>
          %29 = polygeist.submapInverse(%inserted, %28, %arg6, %c6) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index, index) -> tensor<?xi32>
          %30 = affine.for %arg10 = #map2(%arg4) to #map3(%arg4) iter_args(%arg11 = %inserted_1) -> (tensor<?xf32>) {
            %extracted_4 = tensor.extract %arg11[%arg6] : tensor<?xf32>
            %34 = arith.muli %arg10, %c7 : index
            %35 = scf.for %arg12 = %25 to %26 step %c1 iter_args(%arg13 = %extracted_4) -> (f32) {
              %36 = arith.addi %arg12, %4 : index
              %37 = arith.addi %36, %34 : index
              %extracted_6 = tensor.extract %1[%37] : tensor<?xf32>
              %38 = arith.addf %arg13, %extracted_6 : f32
              scf.yield %38 : f32
            }
            %inserted_5 = tensor.insert %35 into %arg11[%arg6] : tensor<?xf32>
            affine.yield %inserted_5 : tensor<?xf32>
          }
          %extracted = tensor.extract %29[%arg6] : tensor<?xi32>
          %extracted_2 = tensor.extract %30[%arg6] : tensor<?xf32>
          %31 = arith.sitofp %extracted : i32 to f32
          %32 = arith.divf %extracted_2, %31 : f32
          %33 = affine.apply #map4(%arg6, %arg2, %arg4)
          %inserted_3 = tensor.insert %32 into %arg9[%33] : tensor<?xf32>
          affine.yield %29, %30, %inserted_3 : tensor<?xi32>, tensor<?xf32>, tensor<?xf32>
        }
        affine.yield %8#2 : tensor<?xf32>
      }
      affine.yield %5 : tensor<?xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?xf32>
    memref.copy %3, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

