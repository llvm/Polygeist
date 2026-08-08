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
          %alloca = memref.alloca(%c3) : memref<?xi32>
          %alloca_0 = memref.alloca(%c3) : memref<?xf32>
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
            affine.store %c0_i32, %alloca[%arg6] : memref<?xi32>
            affine.store %cst, %alloca_0[%arg6] : memref<?xf32>
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              %25 = affine.load %alloca[%arg6] : memref<?xi32>
              %26 = affine.load %alloca_0[%arg6] : memref<?xf32>
              %27 = arith.index_cast %arg7 : index to i32
              %28 = arith.muli %27, %c7_i32 : i32
              %29 = arith.muli %arg7, %c56 : index
              %30:2 = scf.for %arg8 = %7 to %8 step %c1 iter_args(%arg9 = %25, %arg10 = %26) -> (i32, f32) {
                %31 = arith.index_cast %arg8 : index to i32
                %32 = arith.addi %28, %31 : i32
                %33 = arith.muli %32, %c8_i32 : i32
                %34 = arith.muli %arg8, %c8 : index
                %35:2 = scf.for %arg11 = %15 to %22 step %c1 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (i32, f32) {
                  %36 = arith.index_cast %arg11 : index to i32
                  %37 = arith.addi %arg11, %34 : index
                  %38 = arith.addi %37, %0 : index
                  %39 = arith.addi %38, %29 : index
                  %40 = memref.load %arg0[%39] : memref<?xf32>
                  %41 = arith.cmpf ogt, %40, %arg13 : f32
                  %42 = arith.select %41, %40, %arg13 : f32
                  %43 = arith.addi %33, %36 : i32
                  %44 = arith.select %41, %43, %arg12 : i32
                  scf.yield %44, %42 : i32, f32
                }
                scf.yield %35#0, %35#1 : i32, f32
              }
              affine.store %30#0, %alloca[%arg6] : memref<?xi32>
              affine.store %30#1, %alloca_0[%arg6] : memref<?xf32>
            }
            %23 = affine.load %alloca[%arg6] : memref<?xi32>
            %24 = affine.load %alloca_0[%arg6] : memref<?xf32>
            affine.store %24, %arg1[%arg3 * 27 + %arg6 + %arg4 * 9 + %arg5 * 3] : memref<?xf32>
            affine.store %23, %arg2[%arg3 * 27 + %arg6 + %arg4 * 9 + %arg5 * 3] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}

