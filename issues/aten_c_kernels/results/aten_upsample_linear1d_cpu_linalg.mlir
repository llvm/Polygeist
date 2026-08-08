#map = affine_map<(d0, d1) -> (d1 + d0 * 7)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_linear1d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 7.000000e+00 : f32
    %cst_2 = arith.constant 4.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %0 = polygeist.submap(%arg1, %c2, %c7) {map = #map} : (memref<?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%0 : memref<?x?xf32>) {
    ^bb0(%out: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i32
      %3 = arith.muli %2, %c4_i32 : i32
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.addf %6, %cst_3 : f32
      %8 = arith.mulf %7, %cst_2 : f32
      %9 = arith.divf %8, %cst_1 : f32
      %10 = arith.subf %9, %cst_3 : f32
      %11 = arith.cmpf olt, %10, %cst_0 : f32
      %12 = arith.select %11, %cst_0, %10 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %3, %13 : i32
      %15 = arith.index_cast %14 : i32 to index
      %16 = memref.load %arg0[%15] : memref<?xf32>
      %17 = arith.sitofp %13 : i32 to f32
      %18 = arith.subf %12, %17 : f32
      %19 = arith.subf %cst, %18 : f32
      %20 = arith.mulf %16, %19 : f32
      %21 = arith.addi %13, %c1_i32 : i32
      %22 = arith.cmpi slt, %21, %c4_i32 : i32
      %23 = arith.select %22, %21, %13 : i32
      %24 = arith.addi %3, %23 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = memref.load %arg0[%25] : memref<?xf32>
      %27 = arith.mulf %26, %18 : f32
      %28 = arith.addf %20, %27 : f32
      linalg.yield %28 : f32
    }
    return
  }
}

