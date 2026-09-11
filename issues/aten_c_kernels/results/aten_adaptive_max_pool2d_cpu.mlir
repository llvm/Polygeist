#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool2d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c42 = arith.constant 42 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c7_i32 = arith.constant 7 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg3 = 0 to 2 {
      %0 = arith.muli %arg3, %c42 : index
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 3 {
          %1 = arith.muli %arg5, %c7 : index
          %2 = arith.cmpi slt, %1, %c0 : index
          %3 = arith.subi %c-1, %1 : index
          %4 = arith.select %2, %3, %1 : index
          %5 = arith.divsi %4, %c3 : index
          %6 = arith.subi %c-1, %5 : index
          %7 = arith.select %2, %6, %5 : index
          %8 = arith.addi %7, %c3 : index
          %9:2 = affine.for %arg6 = #map(%arg4) to #map1(%arg4) iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f32) {
            %10 = arith.index_cast %arg6 : index to i32
            %11 = arith.muli %10, %c7_i32 : i32
            %12 = arith.muli %arg6, %c7 : index
            %13:2 = scf.for %arg9 = %7 to %8 step %c1 iter_args(%arg10 = %arg7, %arg11 = %arg8) -> (i32, f32) {
              %14 = arith.index_cast %arg9 : index to i32
              %15 = arith.addi %arg9, %0 : index
              %16 = arith.addi %15, %12 : index
              %17 = memref.load %arg0[%16] : memref<?xf32>
              %18 = arith.cmpf ogt, %17, %arg11 : f32
              %19 = arith.select %18, %17, %arg11 : f32
              %20 = scf.if %18 -> (i32) {
                %21 = arith.addi %11, %14 : i32
                scf.yield %21 : i32
              } else {
                scf.yield %arg10 : i32
              }
              scf.yield %20, %19 : i32, f32
            }
            affine.yield %13#0, %13#1 : i32, f32
          }
          affine.store %9#1, %arg1[%arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xf32>
          affine.store %9#0, %arg2[%arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xi32>
        }
      }
    }
    return
  }
}
