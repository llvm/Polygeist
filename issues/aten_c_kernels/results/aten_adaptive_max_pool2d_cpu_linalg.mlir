#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
        %alloca = memref.alloca(%c3) : memref<?xi32>
        %alloca_0 = memref.alloca(%c3) : memref<?xf32>
        affine.for %arg5 = 0 to 3 {
          %1 = arith.muli %arg5, %c7 : index
          %2 = arith.cmpi slt, %1, %c0 : index
          %3 = arith.subi %c-1, %1 : index
          %4 = arith.select %2, %3, %1 : index
          %5 = arith.divsi %4, %c3 : index
          %6 = arith.subi %c-1, %5 : index
          %7 = arith.select %2, %6, %5 : index
          %8 = arith.addi %7, %c3 : index
          affine.store %c0_i32, %alloca[%arg5] : memref<?xi32>
          affine.store %cst, %alloca_0[%arg5] : memref<?xf32>
          affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
            %11 = affine.load %alloca[%arg5] : memref<?xi32>
            %12 = affine.load %alloca_0[%arg5] : memref<?xf32>
            %13 = arith.index_cast %arg6 : index to i32
            %14 = arith.muli %13, %c7_i32 : i32
            %15 = arith.muli %arg6, %c7 : index
            %16:2 = scf.for %arg7 = %7 to %8 step %c1 iter_args(%arg8 = %11, %arg9 = %12) -> (i32, f32) {
              %17 = arith.index_cast %arg7 : index to i32
              %18 = arith.addi %arg7, %0 : index
              %19 = arith.addi %18, %15 : index
              %20 = memref.load %arg0[%19] : memref<?xf32>
              %21 = arith.cmpf ogt, %20, %arg9 : f32
              %22 = arith.select %21, %20, %arg9 : f32
              %23 = arith.addi %14, %17 : i32
              %24 = arith.select %21, %23, %arg8 : i32
              scf.yield %24, %22 : i32, f32
            }
            affine.store %16#0, %alloca[%arg5] : memref<?xi32>
            affine.store %16#1, %alloca_0[%arg5] : memref<?xf32>
          }
          %9 = affine.load %alloca[%arg5] : memref<?xi32>
          %10 = affine.load %alloca_0[%arg5] : memref<?xf32>
          affine.store %10, %arg1[%arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xf32>
          affine.store %9, %arg2[%arg5 + %arg3 * 9 + %arg4 * 3] : memref<?xi32>
        }
      }
    }
    return
  }
}

