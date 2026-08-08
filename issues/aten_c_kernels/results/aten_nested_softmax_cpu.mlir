module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_nested_softmax_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>, %arg2: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 8 {
      %0:2 = scf.while (%arg4 = %c0_i32, %arg5 = %cst) : (i32, f32) -> (f32, i32) {
        %2 = affine.load %arg1[%arg3] : memref<?xi32>
        %3 = arith.cmpi slt, %arg4, %2 : i32
        scf.condition(%3) %arg5, %arg4 : f32, i32
      } do {
      ^bb0(%arg4: f32, %arg5: i32):
        %2 = arith.index_cast %arg5 : i32 to index
        %3 = memref.load %arg0[%arg3, %2] : memref<?x64xf32>
        %4 = math.exp %3 : f32
        memref.store %4, %arg2[%arg3, %2] : memref<?x64xf32>
        %5 = memref.load %arg2[%arg3, %2] : memref<?x64xf32>
        %6 = arith.addf %arg4, %5 : f32
        %7 = arith.addi %arg5, %c1_i32 : i32
        scf.yield %7, %6 : i32, f32
      }
      %1 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        %2 = affine.load %arg1[%arg3] : memref<?xi32>
        %3 = arith.cmpi slt, %arg4, %2 : i32
        scf.condition(%3) %arg4 : i32
      } do {
      ^bb0(%arg4: i32):
        %2 = arith.index_cast %arg4 : i32 to index
        %3 = memref.load %arg2[%arg3, %2] : memref<?x64xf32>
        %4 = arith.divf %3, %0#0 : f32
        memref.store %4, %arg2[%arg3, %2] : memref<?x64xf32>
        %5 = arith.addi %arg4, %c1_i32 : i32
        scf.yield %5 : i32
      }
    }
    return
  }
}
