#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_segment_reduce_lengths_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: i32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg4 : i32 to index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg6 = 0 to 16 {
      %1 = affine.load %alloca[] : memref<i32>
      %2:2 = scf.while (%arg7 = %c0_i32, %arg8 = %1) : (i32, i32) -> (i32, i32) {
        %3 = affine.load %arg1[%arg6] : memref<?xi32>
        %4 = arith.cmpi slt, %arg7, %3 : i32
        scf.condition(%4) %arg8, %arg7 : i32, i32
      } do {
      ^bb0(%arg7: i32, %arg8: i32):
        %3 = arith.cmpi eq, %0, %c0 : index
        %4 = affine.load %arg3[%arg6] : memref<?xf32>
        %5 = affine.apply #map()[%0]
        %6 = arith.cmpi eq, %5, %c0 : index
        %7 = affine.load %arg3[%arg6] : memref<?xf32>
        %8 = affine.load %arg1[%arg6] : memref<?xi32>
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.divf %7, %9 : f32
        %11 = arith.index_cast %arg7 : i32 to index
        %12 = memref.load %arg0[%11] : memref<?xf32>
        %13 = affine.load %arg2[%arg6] : memref<?xf32>
        %14 = arith.cmpf oeq, %12, %13 : f32
        %15 = affine.load %arg3[%arg6] : memref<?xf32>
        %16 = arith.select %14, %15, %cst : f32
        %17 = arith.select %6, %10, %16 : f32
        %18 = arith.select %3, %4, %17 : f32
        %19 = arith.addi %arg7, %c1_i32 : i32
        %20 = arith.index_cast %arg7 : i32 to index
        memref.store %18, %arg5[%20] : memref<?xf32>
        %21 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %21, %19 : i32, i32
      }
      affine.store %2#0, %alloca[] : memref<i32>
    }
    return
  }
}

