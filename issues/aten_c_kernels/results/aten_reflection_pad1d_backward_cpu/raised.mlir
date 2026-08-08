#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_reflection_pad1d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c-2_i32 = arith.constant -2 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg2 = 0 to 2 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      affine.for %arg3 = 0 to 8 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.addi %2, %c-2_i32 : i32
        %4 = affine.apply #map1(%arg3)
        %5 = arith.cmpi sge, %4, %c0 : index
        %6 = arith.subi %c2_i32, %2 : i32
        %7 = arith.select %5, %6, %3 : i32
        %8 = arith.cmpi sge, %7, %c4_i32 : i32
        %9 = arith.subi %c6_i32, %7 : i32
        %10 = arith.select %8, %9, %7 : i32
        %11 = arith.addi %1, %10 : i32
        %12 = arith.index_cast %11 : i32 to index
        %13 = affine.load %arg0[%arg3 + %arg2 * 8] : memref<?xf32>
        %14 = memref.load %arg1[%12] : memref<?xf32>
        %15 = arith.addf %14, %13 : f32
        memref.store %15, %arg1[%12] : memref<?xf32>
      }
    }
    return
  }
}

