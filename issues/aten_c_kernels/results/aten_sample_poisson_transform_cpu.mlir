module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sample_poisson_transform_cpu(%arg0: memref<?xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg0[%arg3] : memref<?xf32>
      %1 = arith.negf %0 : f32
      %2 = math.exp %1 : f32
      %3:3 = scf.while (%arg4 = %c0_i32, %arg5 = %2, %arg6 = %2) : (i32, f32, f32) -> (i32, f32, f32) {
        %4 = arith.cmpi slt, %arg4, %c31_i32 : i32
        %5:4 = scf.if %4 -> (i1, i32, f32, f32) {
          %6 = arith.index_cast %arg4 : i32 to index
          %7 = memref.load %arg1[%arg3, %6] : memref<?x32xf32>
          %8 = arith.cmpf ogt, %7, %arg5 : f32
          %9:3 = scf.if %8 -> (i32, f32, f32) {
            %10 = arith.addi %arg4, %c1_i32 : i32
            %11 = arith.sitofp %10 : i32 to f32
            %12 = arith.divf %0, %11 : f32
            %13 = arith.mulf %arg6, %12 : f32
            %14 = arith.addf %arg5, %13 : f32
            scf.yield %10, %14, %13 : i32, f32, f32
          } else {
            scf.yield %arg4, %arg5, %arg6 : i32, f32, f32
          }
          scf.yield %8, %9#0, %9#1, %9#2 : i1, i32, f32, f32
        } else {
          scf.yield %false, %arg4, %arg5, %arg6 : i1, i32, f32, f32
        }
        scf.condition(%5#0) %5#1, %5#2, %5#3 : i32, f32, f32
      } do {
      ^bb0(%arg4: i32, %arg5: f32, %arg6: f32):
        scf.yield %arg4, %arg5, %arg6 : i32, f32, f32
      }
      affine.store %3#0, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}
