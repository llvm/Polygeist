#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nll_loss2d_forward_cpu(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x16x16xi32>, %arg2: memref<?xf32>, %arg3: memref<?x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %subview = memref.subview %arg3[0, 0, 0] [%c4, %c16, %c16] [1, 1, 1] : memref<?x16x16xf32> to memref<?x?x?xf32, strided<[256, 16, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<?x?x?xf32, strided<[256, 16, 1]>>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = memref.load %arg1[%0, %1, %2] : memref<?x16x16xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %arg0[%0, %4, %1, %2] : memref<?x8x16x16xf32>
      %6 = arith.negf %5 : f32
      %7 = memref.load %arg2[%4] : memref<?xf32>
      %8 = arith.mulf %6, %7 : f32
      linalg.yield %8 : f32
    }
    return
  }
}

