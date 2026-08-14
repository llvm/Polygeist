module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_poisson_transform_cpu(%arg0: memref<?xf32>, %arg1: memref<?x64xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 1024 {
      %0:2 = scf.while (%arg4 = %c0_i32, %arg5 = %cst) : (i32, f32) -> (i32, f32) {
        %2 = arith.cmpi slt, %arg4, %c64_i32 : i32
        %3 = affine.load %arg0[%arg3] : memref<?xf32>
        %4 = arith.negf %3 : f32
        %5 = math.exp %4 : f32
        %6 = arith.cmpf ogt, %arg5, %5 : f32
        %7 = arith.addi %arg4, %c1_i32 : i32
        %8 = arith.index_cast %arg4 : i32 to index
        %9 = memref.load %arg1[%arg3, %8] : memref<?x64xf32>
        %10 = arith.mulf %arg5, %9 : f32
        %11 = arith.select %6, %7, %arg4 : i32
        %12 = arith.select %6, %10, %arg5 : f32
        %13 = arith.select %2, %6, %false : i1
        %14 = arith.select %2, %11, %arg4 : i32
        %15 = arith.select %2, %12, %arg5 : f32
        scf.condition(%13) %14, %15 : i32, f32
      } do {
      ^bb0(%arg4: i32, %arg5: f32):
        scf.yield %arg4, %arg5 : i32, f32
      }
      %1 = arith.addi %0#0, %c-1_i32 : i32
      affine.store %1, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
}

