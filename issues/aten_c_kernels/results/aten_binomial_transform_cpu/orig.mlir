module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_binomial_transform_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?x32xf32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 1024 {
      %0:2 = scf.while (%arg5 = %c0_i32, %arg6 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %1 = arith.cmpi slt, %arg5, %c32_i32 : i32
        %2:3 = scf.if %1 -> (i1, i32, i32) {
          %3 = affine.load %arg0[%arg4] : memref<?xi32>
          %4 = arith.cmpi slt, %arg5, %3 : i32
          %5:2 = scf.if %4 -> (i32, i32) {
            %6 = arith.index_cast %arg5 : i32 to index
            %7 = memref.load %arg2[%arg4, %6] : memref<?x32xf32>
            %8 = affine.load %arg1[%arg4] : memref<?xf32>
            %9 = arith.cmpf olt, %7, %8 : f32
            %10 = arith.extui %9 : i1 to i32
            %11 = arith.addi %arg6, %10 : i32
            %12 = arith.addi %arg5, %c1_i32 : i32
            scf.yield %12, %11 : i32, i32
          } else {
            scf.yield %arg5, %arg6 : i32, i32
          }
          scf.yield %4, %5#0, %5#1 : i1, i32, i32
        } else {
          scf.yield %false, %arg5, %arg6 : i1, i32, i32
        }
        scf.condition(%2#0) %2#1, %2#2 : i32, i32
      } do {
      ^bb0(%arg5: i32, %arg6: i32):
        scf.yield %arg5, %arg6 : i32, i32
      }
      affine.store %0#1, %arg3[%arg4] : memref<?xi32>
    }
    return
  }
}
