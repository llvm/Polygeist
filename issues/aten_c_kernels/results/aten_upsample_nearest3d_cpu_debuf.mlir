#map = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 504 + d1 * 72 + d2 * 9)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c5_i32 = arith.constant 5 : i32
    %c8_i32 = arith.constant 8 : i32
    %c6_i32 = arith.constant 6 : i32
    %c9_i32 = arith.constant 9 : i32
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = polygeist.submap(%0, %c2, %c7, %c8, %c9) {map = #map} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%1 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.muli %6, %c4_i32 : i32
      %8 = linalg.index 1 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = arith.muli %9, %c4_i32 : i32
      %11 = arith.divsi %10, %c7_i32 : i32
      %12 = arith.cmpi sge, %11, %c4_i32 : i32
      %13 = arith.select %12, %c3_i32, %11 : i32
      %14 = arith.addi %7, %13 : i32
      %15 = arith.muli %14, %c5_i32 : i32
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %16 : index to i32
      %18 = arith.muli %17, %c5_i32 : i32
      %19 = arith.divsi %18, %c8_i32 : i32
      %20 = arith.cmpi sge, %19, %c5_i32 : i32
      %21 = arith.select %20, %c4_i32, %19 : i32
      %22 = arith.addi %15, %21 : i32
      %23 = arith.muli %22, %c6_i32 : i32
      %24 = linalg.index 3 : index
      %25 = arith.index_cast %24 : index to i32
      %26 = arith.muli %25, %c6_i32 : i32
      %27 = arith.divsi %26, %c9_i32 : i32
      %28 = arith.cmpi sge, %27, %c6_i32 : i32
      %29 = arith.select %28, %c5_i32, %27 : i32
      %30 = arith.addi %23, %29 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = memref.load %arg0[%31] : memref<?xf32>
      linalg.yield %32 : f32
    } -> tensor<?x?x?x?xf32>
    %3 = polygeist.submapInverse(%0, %2, %c2, %c7, %c8, %c9) {map = #map} : (tensor<?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

