#map = affine_map<(d0, d1) -> (d1 + d0 * 8)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad1d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %false = arith.constant false
    %c-2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = polygeist.submap(%0, %c2, %c8) {map = #map} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%1 : tensor<?x?xf32>) {
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
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %arg0[%19] : memref<?xf32>
      linalg.yield %20 : f32
    } -> tensor<?x?xf32>
    %3 = polygeist.submapInverse(%0, %2, %c2, %c8) {map = #map} : (tensor<?xf32>, tensor<?x?xf32>, index, index) -> tensor<?xf32>
    %4 = bufferization.to_memref %3 : memref<?xf32>
    memref.copy %4, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

