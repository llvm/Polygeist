#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_pdist_forward_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c31_i32 = arith.constant 31 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x32xf32>
    %2 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %0) -> (tensor<?xf32>) {
      %5 = arith.index_cast %arg2 : index to i32
      %6 = arith.subi %c31_i32, %5 : i32
      %7 = arith.muli %5, %6 : i32
      %8 = arith.divsi %7, %c2_i32 : i32
      %9 = affine.for %arg4 = #map(%arg2) to 16 iter_args(%arg5 = %arg3) -> (tensor<?xf32>) {
        %10 = arith.index_cast %arg4 : index to i32
        %11 = arith.subi %10, %5 : i32
        %12 = arith.addi %11, %c-1_i32 : i32
        %13 = arith.addi %8, %12 : i32
        %14 = arith.index_cast %13 : i32 to index
        %inserted = tensor.insert %cst into %arg5[%14] : tensor<?xf32>
        %15 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %inserted) -> (tensor<?xf32>) {
          %extracted = tensor.extract %1[%arg2, %arg6] : tensor<?x32xf32>
          %extracted_0 = tensor.extract %1[%arg4, %arg6] : tensor<?x32xf32>
          %16 = arith.subf %extracted, %extracted_0 : f32
          %17 = arith.mulf %16, %16 : f32
          %extracted_1 = tensor.extract %arg7[%14] : tensor<?xf32>
          %18 = arith.addf %extracted_1, %17 : f32
          %inserted_2 = tensor.insert %18 into %arg7[%14] : tensor<?xf32>
          affine.yield %inserted_2 : tensor<?xf32>
        }
        affine.yield %15 : tensor<?xf32>
      }
      affine.yield %9 : tensor<?xf32>
    }
    %3 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xf32>) {
    ^bb0(%out: f32):
      %5 = math.sqrt %out : f32
      linalg.yield %5 : f32
    } -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

