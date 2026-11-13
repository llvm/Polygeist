module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_spmm_reduce_arg_cpu(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xf32>, %arg3: memref<?x24xf32>, %arg4: i32, %arg5: memref<?x24xf32>, %arg6: memref<?x24xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg4, %c0_i32 : i32
    %1 = arith.select %0, %cst_0, %cst : f32
    affine.for %arg7 = 0 to 16 {
      affine.for %arg8 = 0 to 24 {
        %2 = affine.load %arg0[%arg7] : memref<?xi32>
        %3 = affine.load %arg0[%arg7 + 1] : memref<?xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = arith.index_cast %2 : i32 to index
        %6:2 = scf.for %arg9 = %5 to %4 step %c1 iter_args(%arg10 = %c-1_i32, %arg11 = %1) -> (i32, f32) {
          %7 = arith.index_cast %arg9 : index to i32
          %8 = memref.load %arg2[%arg9] : memref<?xf32>
          %9 = memref.load %arg1[%arg9] : memref<?xi32>
          %10 = arith.index_cast %9 : i32 to index
          %11 = memref.load %arg3[%10, %arg8] : memref<?x24xf32>
          %12 = arith.mulf %8, %11 : f32
          %13 = arith.cmpf ogt, %12, %arg11 : f32
          %14 = arith.select %0, %13, %false : i1
          %15 = arith.cmpf olt, %12, %arg11 : f32
          %16 = arith.select %0, %false, %15 : i1
          %17 = arith.select %14, %true, %16 : i1
          %18 = arith.select %17, %7, %arg10 : i32
          %19 = arith.select %17, %12, %arg11 : f32
          scf.yield %18, %19 : i32, f32
        }
        affine.store %6#1, %arg5[%arg7, %arg8] : memref<?x24xf32>
        affine.store %6#0, %arg6[%arg7, %arg8] : memref<?x24xi32>
      }
    }
    return
  }
}

