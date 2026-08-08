#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_multinomial_with_replacement_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x16xf32>, %arg2: memref<?x16xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %false = arith.constant false
    %c31_i32 = arith.constant 31 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<32xf32>
    %alloca_0 = memref.alloca(%c8) : memref<?xf32>
    affine.for %arg3 = 0 to 8 {
      affine.store %cst, %alloca_0[%arg3] : memref<?xf32>
      %subview = memref.subview %arg0[%arg3, 0] [1, %c32] [1, 1] : memref<?x32xf32> to memref<?xf32, strided<[1], offset: ?>>
      %subview_1 = memref.subview %alloca[0] [%c32] [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
      %subview_2 = memref.subview %alloca_0[%arg3] [1] [1] : memref<?xf32> to memref<f32, strided<[], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_1, %subview_2 : memref<?xf32, strided<[1]>>, memref<f32, strided<[], offset: ?>>) {
      ^bb0(%in: f32, %out: f32, %out_3: f32):
        %1 = arith.addf %out_3, %in : f32
        linalg.yield %1, %1 : f32, f32
      }
      %0 = affine.load %alloca_0[%arg3] : memref<?xf32>
      affine.for %arg4 = 0 to 16 {
        %1 = affine.load %arg1[%arg3, %arg4] : memref<?x16xf32>
        %2 = arith.mulf %1, %0 : f32
        %3 = scf.while (%arg5 = %c0_i32) : (i32) -> i32 {
          %4 = arith.cmpi slt, %arg5, %c31_i32 : i32
          %5 = arith.index_cast %arg5 : i32 to index
          %6 = memref.load %alloca[%5] : memref<32xf32>
          %7 = arith.cmpf olt, %6, %2 : f32
          %8 = arith.addi %arg5, %c1_i32 : i32
          %9 = arith.select %7, %8, %arg5 : i32
          %10 = arith.select %4, %7, %false : i1
          %11 = arith.select %4, %9, %arg5 : i32
          scf.condition(%10) %11 : i32
        } do {
        ^bb0(%arg5: i32):
          scf.yield %arg5 : i32
        }
        affine.store %3, %arg2[%arg3, %arg4] : memref<?x16xi32>
      }
    }
    return
  }
}

