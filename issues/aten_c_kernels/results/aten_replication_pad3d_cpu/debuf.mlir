#map = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 720 + d1 * 90 + d2 * 10)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c3_i32 = arith.constant 3 : i32
    %false = arith.constant false
    %c-2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = polygeist.submap(%0, %c2, %c8, %c9, %c10) {map = #map} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%1 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.muli %6, %c4_i32 : i32
      %8 = linalg.index 1 : index
      %9 = arith.index_cast %8 : index to i32
      %10 = arith.addi %9, %c-2_i32 : i32
      %11 = arith.cmpi slt, %10, %c0_i32 : i32
      %12 = arith.select %11, %c0_i32, %10 : i32
      %13 = affine.apply #map2(%8)
      %14 = arith.cmpi sge, %13, %c0 : index
      %15 = arith.cmpi sge, %10, %c4_i32 : i32
      %16 = arith.select %14, %false, %15 : i1
      %17 = arith.select %16, %c3_i32, %12 : i32
      %18 = arith.addi %7, %17 : i32
      %19 = arith.muli %18, %c5_i32 : i32
      %20 = linalg.index 2 : index
      %21 = arith.index_cast %20 : index to i32
      %22 = arith.addi %21, %c-2_i32 : i32
      %23 = arith.cmpi slt, %22, %c0_i32 : i32
      %24 = arith.select %23, %c0_i32, %22 : i32
      %25 = affine.apply #map2(%20)
      %26 = arith.cmpi sge, %25, %c0 : index
      %27 = arith.cmpi sge, %22, %c5_i32 : i32
      %28 = arith.select %26, %false, %27 : i1
      %29 = arith.select %28, %c4_i32, %24 : i32
      %30 = arith.addi %19, %29 : i32
      %31 = arith.muli %30, %c6_i32 : i32
      %32 = linalg.index 3 : index
      %33 = arith.index_cast %32 : index to i32
      %34 = arith.addi %33, %c-2_i32 : i32
      %35 = arith.cmpi slt, %34, %c0_i32 : i32
      %36 = arith.select %35, %c0_i32, %34 : i32
      %37 = affine.apply #map2(%32)
      %38 = arith.cmpi sge, %37, %c0 : index
      %39 = arith.cmpi sge, %34, %c6_i32 : i32
      %40 = arith.select %38, %false, %39 : i1
      %41 = arith.select %40, %c5_i32, %36 : i32
      %42 = arith.addi %31, %41 : i32
      %43 = arith.index_cast %42 : i32 to index
      %44 = memref.load %arg0[%43] : memref<?xf32>
      linalg.yield %44 : f32
    } -> tensor<?x?x?x?xf32>
    %3 = polygeist.submapInverse(%0, %2, %c2, %c8, %c9, %c10) {map = #map} : (tensor<?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

