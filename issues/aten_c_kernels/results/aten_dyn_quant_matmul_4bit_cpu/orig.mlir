module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_dyn_quant_matmul_4bit_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x32xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x48xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c15_i32 = arith.constant 15 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    affine.for %arg5 = 0 to 32 {
      affine.for %arg6 = 0 to 48 {
        %0 = affine.load %arg3[%arg6] : memref<?xf32>
        %1 = affine.load %arg2[%arg6] : memref<?xf32>
        %2 = affine.for %arg7 = 0 to 64 iter_args(%arg8 = %cst) -> (f32) {
          %3 = arith.index_cast %arg7 : index to i32
          %4 = arith.cmpi slt, %arg7, %c0 : index
          %5 = arith.subi %c-1, %arg7 : index
          %6 = arith.select %4, %5, %arg7 : index
          %7 = arith.divsi %6, %c2 : index
          %8 = arith.subi %c-1, %7 : index
          %9 = arith.select %4, %8, %7 : index
          %10 = memref.load %arg1[%arg6, %9] : memref<?x32xi8>
          %11 = arith.extui %10 : i8 to i32
          %12 = arith.andi %3, %c1_i32 : i32
          %13 = arith.muli %12, %c4_i32 : i32
          %14 = arith.shrsi %11, %13 : i32
          %15 = arith.andi %14, %c15_i32 : i32
          %16 = affine.load %arg0[%arg5, %arg7] : memref<?x64xf32>
          %17 = arith.sitofp %15 : i32 to f32
          %18 = arith.subf %17, %0 : f32
          %19 = arith.mulf %16, %18 : f32
          %20 = arith.mulf %19, %1 : f32
          %21 = arith.addf %arg8, %20 : f32
          affine.yield %21 : f32
        }
        affine.store %2, %arg4[%arg5, %arg6] : memref<?x48xf32>
      }
    }
    return
  }
}
