#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool3d_legacy_cpu(%arg0: memref<?x8x9x10xf32>, %arg1: memref<?x3x4x5xf32>, %arg2: memref<?x3x4x5xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-11 = arith.constant -11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32 = arith.constant 8 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 3 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c8_i32 : i32
        %2 = arith.divsi %1, %c3_i32 : i32
        %3 = arith.muli %2, %c9_i32 : i32
        %4 = arith.muli %arg4, %c8 : index
        %5 = arith.cmpi slt, %4, %c0 : index
        %6 = arith.subi %c-1, %4 : index
        %7 = arith.select %5, %6, %4 : index
        %8 = arith.divsi %7, %c3 : index
        %9 = arith.subi %c-1, %8 : index
        %10 = arith.select %5, %9, %8 : index
        %11 = arith.addi %4, %c10 : index
        %12 = arith.cmpi slt, %11, %c0 : index
        %13 = arith.subi %c-11, %4 : index
        %14 = arith.select %12, %13, %11 : index
        %15 = arith.divsi %14, %c3 : index
        %16 = arith.subi %c-1, %15 : index
        %17 = arith.select %12, %16, %15 : index
        affine.for %arg5 = 0 to 4 {
          %18 = arith.index_cast %arg5 : index to i32
          %19 = arith.muli %18, %c9_i32 : i32
          %20 = arith.divsi %19, %c4_i32 : i32
          %21 = arith.addi %3, %20 : i32
          %22 = arith.muli %21, %c10_i32 : i32
          %23 = arith.muli %arg5, %c9 : index
          %24 = arith.cmpi slt, %23, %c0 : index
          %25 = arith.subi %c-1, %23 : index
          %26 = arith.select %24, %25, %23 : index
          %27 = arith.divsi %26, %c4 : index
          %28 = arith.subi %c-1, %27 : index
          %29 = arith.select %24, %28, %27 : index
          %30 = arith.addi %29, %c3 : index
          affine.for %arg6 = 0 to 5 {
            %31 = arith.index_cast %arg6 : index to i32
            %32 = arith.muli %31, %c10_i32 : i32
            %33 = arith.divsi %32, %c5_i32 : i32
            %34 = arith.addi %22, %33 : i32
            %35 = arith.muli %arg6, %c2 : index
            %36 = memref.load %arg0[%arg3, %10, %29, %35] : memref<?x8x9x10xf32>
            %37:2 = scf.for %arg7 = %10 to %17 step %c1 iter_args(%arg8 = %36, %arg9 = %34) -> (f32, i32) {
              %38 = arith.index_cast %arg7 : index to i32
              %39 = arith.muli %38, %c9_i32 : i32
              %40:2 = scf.for %arg10 = %29 to %30 step %c1 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (f32, i32) {
                %41 = arith.index_cast %arg10 : index to i32
                %42 = arith.addi %39, %41 : i32
                %43 = arith.muli %42, %c10_i32 : i32
                %44:2 = affine.for %arg13 = #map(%arg6) to #map1(%arg6) iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (f32, i32) {
                  %45 = arith.index_cast %arg13 : index to i32
                  %46 = memref.load %arg0[%arg3, %arg7, %arg10, %arg13] : memref<?x8x9x10xf32>
                  %47 = arith.cmpf ogt, %46, %arg14 : f32
                  %48 = arith.select %47, %46, %arg14 : f32
                  %49 = scf.if %47 -> (i32) {
                    %50 = arith.addi %43, %45 : i32
                    scf.yield %50 : i32
                  } else {
                    scf.yield %arg15 : i32
                  }
                  affine.yield %48, %49 : f32, i32
                }
                scf.yield %44#0, %44#1 : f32, i32
              }
              scf.yield %40#0, %40#1 : f32, i32
            }
            affine.store %37#0, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<?x3x4x5xf32>
            affine.store %37#1, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x3x4x5xi32>
          }
        }
      }
    }
    return
  }
}
