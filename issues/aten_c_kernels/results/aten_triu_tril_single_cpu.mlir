module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_triu_tril_single_cpu(%arg0: memref<?x24xf32>, %arg1: i32, %arg2: i32, %arg3: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg2, %c0_i32 : i32
    affine.for %arg4 = 0 to 32 {
      %1 = arith.index_cast %arg4 : index to i32
      affine.for %arg5 = 0 to 24 {
        %2 = arith.index_cast %arg5 : index to i32
        %3 = scf.if %0 -> (i1) {
          %5 = arith.subi %2, %1 : i32
          %6 = arith.cmpi sge, %5, %arg1 : i32
          scf.yield %6 : i1
        } else {
          %5 = arith.subi %2, %1 : i32
          %6 = arith.cmpi sle, %5, %arg1 : i32
          scf.yield %6 : i1
        }
        %4 = scf.if %3 -> (f32) {
          %5 = affine.load %arg0[%arg4, %arg5] : memref<?x24xf32>
          scf.yield %5 : f32
        } else {
          scf.yield %cst : f32
        }
        affine.store %4, %arg3[%arg4, %arg5] : memref<?x24xf32>
      }
    }
    return
  }
}
