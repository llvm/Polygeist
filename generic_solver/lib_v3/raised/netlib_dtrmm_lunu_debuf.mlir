#map = affine_map<(d0)[s0] -> (s0, d0)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_lunu(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.subi %3, %c1 : index
    %5 = affine.for %arg7 = 0 to %2 iter_args(%arg8 = %1) -> (tensor<?x?xf64>) {
      %7 = affine.for %arg9 = 0 to %3 iter_args(%arg10 = %arg8) -> (tensor<?x?xf64>) {
        %extracted = tensor.extract %arg10[%arg7, %arg9] : tensor<?x?xf64>
        %8 = arith.mulf %arg2, %extracted : f64
        %9 = polygeist.submap(%0, %arg9, %4) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %10 = polygeist.submap(%arg10, %arg7, %4) {map = #map} : (tensor<?x?xf64>, index, index) -> tensor<?xf64>
        %11 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%9 : tensor<?xf64>) outs(%10 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %13 = arith.mulf %8, %in : f64
          %14 = arith.addf %out, %13 : f64
          %15 = linalg.index 0 : index
          %16 = arith.cmpi slt, %15, %arg9 : index
          %17 = arith.select %16, %14, %out : f64
          linalg.yield %17 : f64
        } -> tensor<?xf64>
        %12 = polygeist.submapInverse(%arg10, %11, %arg7, %4) {map = #map} : (tensor<?x?xf64>, tensor<?xf64>, index, index) -> tensor<?x?xf64>
        %inserted = tensor.insert %8 into %12[%arg7, %arg9] : tensor<?x?xf64>
        affine.yield %inserted : tensor<?x?xf64>
      }
      affine.yield %7 : tensor<?x?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf64>
    memref.copy %6, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

