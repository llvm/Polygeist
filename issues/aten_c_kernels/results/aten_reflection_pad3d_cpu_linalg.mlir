#map = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 720 + d1 * 90 + d2 * 10)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_reflection_pad3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c-2_i32 = arith.constant -2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c8_i32 = arith.constant 8 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = polygeist.submap(%arg1, %c2, %c8, %c9, %c10) {map = #map} : (memref<?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.muli %2, %c4_i32 : i32
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.addi %5, %c-2_i32 : i32
      %7 = affine.apply #map2(%4)
      %8 = arith.cmpi sge, %7, %c0 : index
      %9 = arith.subi %c2_i32, %5 : i32
      %10 = arith.select %8, %9, %6 : i32
      %11 = arith.cmpi sge, %10, %c4_i32 : i32
      %12 = arith.subi %c6_i32, %10 : i32
      %13 = arith.select %11, %12, %10 : i32
      %14 = arith.addi %3, %13 : i32
      %15 = arith.muli %14, %c5_i32 : i32
      %16 = linalg.index 2 : index
      %17 = arith.index_cast %16 : index to i32
      %18 = arith.addi %17, %c-2_i32 : i32
      %19 = affine.apply #map2(%16)
      %20 = arith.cmpi sge, %19, %c0 : index
      %21 = arith.subi %c2_i32, %17 : i32
      %22 = arith.select %20, %21, %18 : i32
      %23 = arith.cmpi sge, %22, %c5_i32 : i32
      %24 = arith.subi %c8_i32, %22 : i32
      %25 = arith.select %23, %24, %22 : i32
      %26 = arith.addi %15, %25 : i32
      %27 = arith.muli %26, %c6_i32 : i32
      %28 = linalg.index 3 : index
      %29 = arith.index_cast %28 : index to i32
      %30 = arith.addi %29, %c-2_i32 : i32
      %31 = affine.apply #map2(%28)
      %32 = arith.cmpi sge, %31, %c0 : index
      %33 = arith.subi %c2_i32, %29 : i32
      %34 = arith.select %32, %33, %30 : i32
      %35 = arith.cmpi sge, %34, %c6_i32 : i32
      %36 = arith.subi %c10_i32, %34 : i32
      %37 = arith.select %35, %36, %34 : i32
      %38 = arith.addi %27, %37 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = memref.load %arg0[%39] : memref<?xf32>
      linalg.yield %40 : f32
    }
    return
  }
}

