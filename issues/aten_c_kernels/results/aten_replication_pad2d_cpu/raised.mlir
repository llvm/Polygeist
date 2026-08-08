#map = affine_map<(d0, d1, d2) -> (d2 + d0 * 72 + d1 * 9)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_replication_pad2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c0_i32 = arith.constant 0 : i32
    %c-2_i32 = arith.constant -2 : i32
    %false = arith.constant false
    %c3_i32 = arith.constant 3 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = polygeist.submap(%arg1, %c2, %c8, %c9) {map = #map} : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.muli %2, %c4_i32 : i32
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.addi %5, %c-2_i32 : i32
      %7 = arith.cmpi slt, %6, %c0_i32 : i32
      %8 = arith.select %7, %c0_i32, %6 : i32
      %9 = affine.apply #map2(%4)
      %10 = arith.cmpi sge, %9, %c0 : index
      %11 = arith.cmpi sge, %6, %c4_i32 : i32
      %12 = arith.select %10, %false, %11 : i1
      %13 = arith.select %12, %c3_i32, %8 : i32
      %14 = arith.addi %3, %13 : i32
      %15 = arith.muli %14, %c5_i32 : i32
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %16 : index to i32
      %18 = arith.addi %17, %c-2_i32 : i32
      %19 = arith.cmpi slt, %18, %c0_i32 : i32
      %20 = arith.select %19, %c0_i32, %18 : i32
      %21 = affine.apply #map2(%16)
      %22 = arith.cmpi sge, %21, %c0 : index
      %23 = arith.cmpi sge, %18, %c5_i32 : i32
      %24 = arith.select %22, %false, %23 : i1
      %25 = arith.select %24, %c4_i32, %20 : i32
      %26 = arith.addi %15, %25 : i32
      %27 = arith.index_cast %26 : i32 to index
      %28 = memref.load %arg0[%27] : memref<?xf32>
      linalg.yield %28 : f32
    }
    return
  }
}

