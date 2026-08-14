module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_cummax_cummin_cpu(%arg0: memref<?x64xf32>, %arg1: i32, %arg2: memref<?x64xf32>, %arg3: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %c0_i32 : i32
    affine.for %arg4 = 0 to 16 {
      %1 = affine.load %arg0[%arg4, 0] : memref<?x64xf32>
      %2:2 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %c0_i32, %arg7 = %1) -> (i32, f32) {
        %3 = arith.index_cast %arg5 : index to i32
        %4 = scf.if %0 -> (i1) {
          %8 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
          %9 = arith.cmpf oge, %8, %arg7 : f32
          scf.yield %9 : i1
        } else {
          scf.yield %false : i1
        }
        %5 = scf.if %4 -> (i1) {
          scf.yield %true : i1
        } else {
          %8 = scf.if %0 -> (i1) {
            scf.yield %false : i1
          } else {
            %9 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
            %10 = arith.cmpf ole, %9, %arg7 : f32
            scf.yield %10 : i1
          }
          scf.yield %8 : i1
        }
        %6 = arith.select %5, %3, %arg6 : i32
        %7 = scf.if %5 -> (f32) {
          %8 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
          scf.yield %8 : f32
        } else {
          scf.yield %arg7 : f32
        }
        affine.store %7, %arg2[%arg4, %arg5] : memref<?x64xf32>
        affine.store %6, %arg3[%arg4, %arg5] : memref<?x64xi32>
        affine.yield %6, %7 : i32, f32
      }
    }
    return
  }
}
