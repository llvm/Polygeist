module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sobol_draw_cpu(%arg0: memref<?xi32>, %arg1: memref<?x32xi32>, %arg2: memref<?x8xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.32830644E-10 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 256 {
      %0 = arith.index_cast %arg3 : index to i32
      %1:2 = scf.while (%arg4 = %0, %arg5 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %3 = arith.andi %arg4, %c1_i32 : i32
        %4 = arith.cmpi ne, %3, %c0_i32 : i32
        scf.condition(%4) %arg5, %arg4 : i32, i32
      } do {
      ^bb0(%arg4: i32, %arg5: i32):
        %3 = arith.addi %arg4, %c1_i32 : i32
        %4 = arith.shrsi %arg5, %c1_i32 : i32
        scf.yield %4, %3 : i32, i32
      }
      %2 = arith.index_cast %1#0 : i32 to index
      affine.for %arg4 = 0 to 8 {
        %3 = memref.load %arg1[%arg4, %2] : memref<?x32xi32>
        %4 = affine.load %arg0[%arg4] : memref<?xi32>
        %5 = arith.xori %4, %3 : i32
        affine.store %5, %arg0[%arg4] : memref<?xi32>
        %6 = arith.uitofp %5 : i32 to f32
        %7 = arith.mulf %6, %cst : f32
        affine.store %7, %arg2[%arg3, %arg4] : memref<?x8xf32>
      }
    }
    return
  }
}
