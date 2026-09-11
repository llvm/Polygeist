#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 * 7)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest1d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %2) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %8 = arith.index_cast %arg4 : index to i32
        %9 = arith.muli %8, %c4_i32 : i32
        %10 = arith.divsi %9, %c7_i32 : i32
        %11 = arith.cmpi sge, %10, %c4_i32 : i32
        %12 = arith.select %11, %c3_i32, %10 : i32
        %13 = arith.addi %6, %12 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = affine.apply #map1(%arg4, %arg2)
        %extracted = tensor.extract %1[%15] : tensor<?xf32>
        %extracted_0 = tensor.extract %arg5[%14] : tensor<?xf32>
        %16 = arith.addf %extracted_0, %extracted : f32
        %inserted = tensor.insert %16 into %arg5[%14] : tensor<?xf32>
        affine.yield %inserted : tensor<?xf32>
      }
      affine.yield %7 : tensor<?xf32>
    }
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

