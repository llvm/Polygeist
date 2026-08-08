module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dyn_quant_pack_4bit_weight_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x32xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %c4_i32 = arith.constant 4 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 1.500000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    affine.for %arg4 = 0 to 48 {
      %0 = affine.load %arg0[%arg4, 0] : memref<?x64xf32>
      %1:2 = affine.for %arg5 = 1 to 64 iter_args(%arg6 = %0, %arg7 = %0) -> (f32, f32) {
        %6 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
        %7 = arith.cmpf olt, %6, %arg7 : f32
        %8 = arith.select %7, %6, %arg7 : f32
        %9 = arith.cmpf ogt, %6, %arg6 : f32
        %10 = arith.select %9, %6, %arg6 : f32
        affine.yield %10, %8 : f32, f32
      }
      %2 = arith.subf %1#0, %1#1 : f32
      %3 = arith.divf %2, %cst_0 : f32
      affine.store %3, %arg2[%arg4] : memref<?xf32>
      %4 = arith.negf %1#1 : f32
      %5 = arith.divf %4, %3 : f32
      affine.store %5, %arg3[%arg4] : memref<?xf32>
      affine.for %arg5 = 0 to 64 step 2 {
        %6 = affine.load %arg0[%arg4, %arg5] : memref<?x64xf32>
        %7 = affine.load %arg2[%arg4] : memref<?xf32>
        %8 = arith.divf %6, %7 : f32
        %9 = affine.load %arg3[%arg4] : memref<?xf32>
        %10 = arith.addf %8, %9 : f32
        %11 = arith.addf %10, %cst : f32
        %12 = arith.fptosi %11 : f32 to i32
        %13 = affine.load %arg0[%arg4, %arg5 + 1] : memref<?x64xf32>
        %14 = arith.divf %13, %7 : f32
        %15 = arith.addf %14, %9 : f32
        %16 = arith.addf %15, %cst : f32
        %17 = arith.fptosi %16 : f32 to i32
        %18 = arith.cmpi slt, %12, %c0_i32 : i32
        %19 = arith.select %18, %c0_i32, %12 : i32
        %20 = scf.if %18 -> (i1) {
          scf.yield %false : i1
        } else {
          %35 = arith.cmpi sgt, %12, %c15_i32 : i32
          scf.yield %35 : i1
        }
        %21 = arith.select %20, %c15_i32, %19 : i32
        %22 = arith.cmpi slt, %17, %c0_i32 : i32
        %23 = arith.select %22, %c0_i32, %17 : i32
        %24 = scf.if %22 -> (i1) {
          scf.yield %false : i1
        } else {
          %35 = arith.cmpi sgt, %17, %c15_i32 : i32
          scf.yield %35 : i1
        }
        %25 = arith.select %24, %c15_i32, %23 : i32
        %26 = arith.shli %25, %c4_i32 : i32
        %27 = arith.ori %21, %26 : i32
        %28 = arith.trunci %27 : i32 to i8
        %29 = arith.cmpi slt, %arg5, %c0 : index
        %30 = arith.subi %c-1, %arg5 : index
        %31 = arith.select %29, %30, %arg5 : index
        %32 = arith.divsi %31, %c2 : index
        %33 = arith.subi %c-1, %32 : index
        %34 = arith.select %29, %33, %32 : index
        memref.store %28, %arg1[%arg4, %34] : memref<?x32xi8>
      }
    }
    return
  }
}
