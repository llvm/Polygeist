#map = affine_map<()[s0] -> (s0 - 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: i32, %arg5: memref<?x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant -3.40282347E+38 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_0 = arith.constant 3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg4 : i32 to index
    %1 = arith.cmpi eq, %arg4, %c2_i32 : i32
    %2 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %3 = arith.cmpi eq, %arg4, %c3_i32 : i32
    %4 = arith.select %3, %cst_0, %cst_1 : f32
    %5 = arith.select %1, %cst, %4 : f32
    %6 = arith.cmpi eq, %arg4, %c1_i32 : i32
    %7 = arith.select %2, %true, %6 : i1
    %8 = arith.addi %0, %c-1 : index
    %9 = arith.cmpi eq, %8, %c0 : index
    affine.for %arg6 = 0 to 16 {
      affine.for %arg7 = 0 to 24 {
        %10 = affine.load %arg0[%arg6] : memref<?xi32>
        %11 = affine.load %arg0[%arg6 + 1] : memref<?xi32>
        %12 = arith.index_cast %11 : i32 to index
        %13 = arith.index_cast %10 : i32 to index
        %14 = scf.for %arg8 = %13 to %12 step %c1 iter_args(%arg9 = %5) -> (f32) {
          %21 = memref.load %arg2[%arg8] : memref<?xf32>
          %22 = memref.load %arg1[%arg8] : memref<?xi32>
          %23 = arith.index_cast %22 : i32 to index
          %24 = memref.load %arg3[%23, %arg7] : memref<?x24xf32>
          %25 = arith.mulf %21, %24 : f32
          %26 = scf.if %7 -> (f32) {
            %27 = arith.addf %arg9, %25 : f32
            scf.yield %27 : f32
          } else {
            %27 = affine.apply #map()[%0]
            %28 = arith.cmpi eq, %27, %c0 : index
            %29 = arith.cmpf ogt, %arg9, %25 : f32
            %30 = arith.select %29, %arg9, %25 : f32
            %31 = arith.cmpf olt, %arg9, %25 : f32
            %32 = arith.select %31, %arg9, %25 : f32
            %33 = arith.select %28, %30, %32 : f32
            scf.yield %33 : f32
          }
          scf.yield %26 : f32
        }
        %15 = arith.cmpi sgt, %11, %10 : i32
        %16 = arith.andi %9, %15 : i1
        %17 = arith.subi %11, %10 : i32
        %18 = arith.sitofp %17 : i32 to f32
        %19 = arith.divf %14, %18 : f32
        %20 = arith.select %16, %19, %14 : f32
        affine.store %20, %arg5[%arg6, %arg7] : memref<?x24xf32>
      }
    }
    return
  }
}

