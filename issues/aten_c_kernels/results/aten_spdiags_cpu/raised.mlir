#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spdiags_cpu(%arg0: memref<?x16xf32>, %arg1: memref<?xi32>, %arg2: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %subview = memref.subview %arg2[0, 0] [%c16, %c16] [1, 1] : memref<?x16xf32> to memref<?x?xf32, strided<[16, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[16, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg3 = 0 to 5 {
      affine.for %arg4 = 0 to 16 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = affine.load %arg1[%arg3] : memref<?xi32>
        %2 = arith.addi %0, %1 : i32
        %3 = arith.cmpi sge, %2, %c0_i32 : i32
        %4 = arith.cmpi slt, %2, %c16_i32 : i32
        %5 = arith.andi %3, %4 : i1
        scf.if %5 {
          %6 = arith.index_cast %2 : i32 to index
          %7 = affine.load %arg1[%arg3] : memref<?xi32>
          %8 = arith.cmpi sge, %7, %c0_i32 : i32
          %9 = arith.subi %0, %7 : i32
          %10 = arith.select %8, %0, %9 : i32
          %11 = arith.index_cast %10 : i32 to index
          %12 = memref.load %arg0[%arg3, %11] : memref<?x16xf32>
          memref.store %12, %arg2[%arg4, %6] : memref<?x16xf32>
        }
      }
    }
    return
  }
}

