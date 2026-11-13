#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_std_var_cpu(%arg0: memref<?x64xf32>, %arg1: i32, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 6.400000e+01 : f32
    %c64 = arith.constant 64 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %2 = arith.subi %c64_i32, %arg1 : i32
    %3 = arith.sitofp %2 : i32 to f32
    %4 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %0) -> (tensor<?xf32>) {
      %alloca = memref.alloca() : memref<f32>
      %6 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %cst into %6[] : tensor<f32>
      %extracted_slice = tensor.extract_slice %1[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %7 = kernel.launch @cudnnReduceSum_f32(%extracted_slice, %inserted) : (tensor<?xf32>, tensor<f32>) -> tensor<f32>
      %extracted = tensor.extract %7[] : tensor<f32>
      %8 = arith.divf %extracted, %cst_0 : f32
      %alloca_1 = memref.alloca() : memref<f32>
      %9 = bufferization.to_tensor %alloca_1 : memref<f32>
      %inserted_2 = tensor.insert %cst into %9[] : tensor<f32>
      %extracted_slice_3 = tensor.extract_slice %1[%arg3, 0] [1, %c64] [1, 1] : tensor<?x64xf32> to tensor<?xf32>
      %10 = linalg.generic {doc = "", indexing_maps = [#map, #map1], iterator_types = ["reduction"], library_call = ""} ins(%extracted_slice_3 : tensor<?xf32>) outs(%inserted_2 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.subf %in, %8 : f32
        %14 = arith.mulf %13, %13 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<f32>
      %extracted_4 = tensor.extract %10[] : tensor<f32>
      %11 = arith.divf %extracted_4, %3 : f32
      %12 = math.sqrt %11 : f32
      %inserted_5 = tensor.insert %12 into %arg4[%arg3] : tensor<?xf32>
      affine.yield %inserted_5 : tensor<?xf32>
    }
    %5 = bufferization.to_memref %4 : memref<?xf32>
    memref.copy %5, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

