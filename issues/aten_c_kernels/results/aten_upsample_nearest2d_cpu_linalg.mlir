#map = affine_map<(d0, d1, d2) -> (d2 + d0 * 56 + d1 * 8)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = polygeist.submap(%arg1, %c2, %c7, %c8) {map = #map} : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.muli %2, %c4_i32 : i32
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.muli %5, %c4_i32 : i32
      %7 = arith.divsi %6, %c7_i32 : i32
      %8 = arith.cmpi sge, %7, %c4_i32 : i32
      %9 = arith.select %8, %c3_i32, %7 : i32
      %10 = arith.addi %3, %9 : i32
      %11 = arith.muli %10, %c5_i32 : i32
      %12 = linalg.index 2 : index
      %13 = arith.index_cast %12 : index to i32
      %14 = arith.muli %13, %c5_i32 : i32
      %15 = arith.divsi %14, %c8_i32 : i32
      %16 = arith.cmpi sge, %15, %c5_i32 : i32
      %17 = arith.select %16, %c4_i32, %15 : i32
      %18 = arith.addi %11, %17 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %arg0[%19] : memref<?xf32>
      linalg.yield %20 : f32
    }
    return
  }
}

