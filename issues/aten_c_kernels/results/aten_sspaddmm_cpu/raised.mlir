#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sspaddmm_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x48xf32>, %arg4: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %c48 = arith.constant 48 : index
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg4[0, 0] [%c64, %c48] [1, 1] : memref<?x48xf32> to memref<?x?xf32, strided<[48, 1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<?x?xf32, strided<[48, 1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg5 = 0 to 512 {
      affine.for %arg6 = 0 to 48 {
        %0 = affine.load %arg0[%arg5] : memref<?xi32>
        %1 = arith.index_cast %0 : i32 to index
        %2 = affine.load %arg2[%arg5] : memref<?xf32>
        %3 = affine.load %arg1[%arg5] : memref<?xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg3[%4, %arg6] : memref<?x48xf32>
        %6 = arith.mulf %2, %5 : f32
        %7 = memref.load %arg4[%1, %arg6] : memref<?x48xf32>
        %8 = arith.addf %7, %6 : f32
        memref.store %8, %arg4[%1, %arg6] : memref<?x48xf32>
      }
    }
    return
  }
}

