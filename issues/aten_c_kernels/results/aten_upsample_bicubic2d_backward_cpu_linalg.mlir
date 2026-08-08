#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_bicubic2d_backward_cpu(%arg0: memref<?x8xf32>, %arg1: memref<?x5xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c25 = arith.constant 25 : index
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %c5 = arith.constant 5 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %subview = memref.subview %arg1[0, 0] [1, %c25] [1, 1] : memref<?x5xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%subview : memref<?xf32, strided<[1]>>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg2 = 0 to 8 {
      %0 = arith.muli %arg2, %c5 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.subi %c-1, %0 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.divsi %3, %c8 : index
      %5 = arith.subi %c-1, %4 : index
      %6 = arith.select %1, %5, %4 : index
      affine.for %arg3 = 0 to 8 {
        %7 = affine.load %arg0[%arg2, %arg3] : memref<?x8xf32>
        %8 = arith.muli %arg3, %c5 : index
        %9 = arith.cmpi slt, %8, %c0 : index
        %10 = arith.subi %c-1, %8 : index
        %11 = arith.select %9, %10, %8 : index
        %12 = arith.divsi %11, %c8 : index
        %13 = arith.subi %c-1, %12 : index
        %14 = arith.select %9, %13, %12 : index
        %15 = memref.load %arg1[%6, %14] : memref<?x5xf32>
        %16 = arith.addf %15, %7 : f32
        memref.store %16, %arg1[%6, %14] : memref<?x5xf32>
      }
    }
    return
  }
}

