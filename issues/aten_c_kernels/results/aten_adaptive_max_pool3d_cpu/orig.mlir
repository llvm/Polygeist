#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool3d_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-11 = arith.constant -11 : index
    %c56 = arith.constant 56 : index
    %c336 = arith.constant 336 : index
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c-1 = arith.constant -1 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg3 = 0 to 2 {
      %0 = arith.muli %arg3, %c336 : index
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
          affine.for %arg6 = 0 to 3 {
            %9 = arith.muli %arg6, %c8 : index
            %10 = arith.cmpi slt, %9, %c0 : index
            %11 = arith.subi %c-1, %9 : index
            %12 = arith.select %10, %11, %9 : index
            %13 = arith.divsi %12, %c3 : index
            %14 = arith.subi %c-1, %13 : index
            %15 = arith.select %10, %14, %13 : index
            %16 = arith.addi %9, %c10 : index
            %17 = arith.cmpi slt, %16, %c0 : index
            %18 = arith.subi %c-11, %9 : index
            %19 = arith.select %17, %18, %16 : index
            %20 = arith.divsi %19, %c3 : index
            %21 = arith.subi %c-1, %20 : index
            %22 = arith.select %17, %21, %20 : index
            %23:2 = affine.for %arg7 = #map(%arg4) to #map1(%arg4) iter_args(%arg8 = %c0_i32, %arg9 = %cst) -> (i32, f32) {
              %24 = arith.index_cast %arg7 : index to i32
              %25 = arith.muli %24, %c7_i32 : i32
              %26 = arith.muli %arg7, %c56 : index
              %27:2 = scf.for %arg10 = %7 to %8 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (i32, f32) {
                %28 = arith.index_cast %arg10 : index to i32
                %29 = arith.addi %25, %28 : i32
                %30 = arith.muli %29, %c8_i32 : i32
                %31 = arith.muli %arg10, %c8 : index
                %32:2 = scf.for %arg13 = %15 to %22 step %c1 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (i32, f32) {
                  %33 = arith.index_cast %arg13 : index to i32
                  %34 = arith.addi %arg13, %31 : index
                  %35 = arith.addi %34, %0 : index
                  %36 = arith.addi %35, %26 : index
                  %37 = memref.load %arg0[%36] : memref<?xf32>
                  %38 = arith.cmpf ogt, %37, %arg15 : f32
                  %39 = arith.select %38, %37, %arg15 : f32
                  %40 = scf.if %38 -> (i32) {
                    %41 = arith.addi %30, %33 : i32
                    scf.yield %41 : i32
                  } else {
                    scf.yield %arg14 : i32
                  }
                  scf.yield %40, %39 : i32, f32
                }
                scf.yield %32#0, %32#1 : i32, f32
              }
              affine.yield %27#0, %27#1 : i32, f32
            }
            affine.store %23#1, %arg1[%arg3 * 27 + %arg6 + %arg4 * 9 + %arg5 * 3] : memref<?xf32>
            affine.store %23#0, %arg2[%arg3 * 27 + %arg6 + %arg4 * 9 + %arg5 * 3] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}
