#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 - 9)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_reflection_pad2d(%arg0: memref<?x8x8xf32>, %arg1: memref<?x10x10xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x10x10xf32>
    %extracted_slice = tensor.extract_slice %0[0, 0, 0] [%c3, %c10, %c10] [1, 1, 1] : tensor<?x10x10xf32> to tensor<?x?x?xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.cmpi eq, %4, %c0 : index
      %7 = affine.apply #map1(%4)
      %8 = arith.cmpi eq, %7, %c0 : index
      %9 = arith.addi %5, %c-1_i32 : i32
      %10 = arith.select %8, %c6_i32, %9 : i32
      %11 = arith.select %6, %c1_i32, %10 : i32
      %12 = arith.index_cast %11 : i32 to index
      %13 = linalg.index 2 : index
      %14 = arith.index_cast %13 : index to i32
      %15 = arith.cmpi eq, %13, %c0 : index
      %16 = affine.apply #map1(%13)
      %17 = arith.cmpi eq, %16, %c0 : index
      %18 = arith.addi %14, %c-1_i32 : i32
      %19 = arith.select %17, %c6_i32, %18 : i32
      %20 = arith.select %15, %c1_i32, %19 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg0[%3, %12, %21] : memref<?x8x8xf32>
      linalg.yield %22 : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0, 0] [%c3, %c10, %c10] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x10x10xf32>
    %2 = bufferization.to_memref %inserted_slice : memref<?x10x10xf32>
    memref.copy %2, %arg1 : memref<?x10x10xf32> to memref<?x10x10xf32>
    return
  }
}

