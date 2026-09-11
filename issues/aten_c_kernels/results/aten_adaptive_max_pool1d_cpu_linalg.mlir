module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_adaptive_max_pool1d_cpu(%arg0: memref<?x32xf32>, %arg1: memref<?x7xf32>, %arg2: memref<?x7xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-39 = arith.constant -39 : index
    %c38 = arith.constant 38 : index
    %c-1 = arith.constant -1 : index
    %c32 = arith.constant 32 : index
    %c7 = arith.constant 7 : index
    %c32_i32 = arith.constant 32 : i32
    %c7_i32 = arith.constant 7 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 7 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.divsi %1, %c7_i32 : i32
        %3 = arith.muli %arg4, %c32 : index
        %4 = arith.cmpi slt, %3, %c0 : index
        %5 = arith.subi %c-1, %3 : index
        %6 = arith.select %4, %5, %3 : index
        %7 = arith.divsi %6, %c7 : index
        %8 = arith.subi %c-1, %7 : index
        %9 = arith.select %4, %8, %7 : index
        %10 = memref.load %arg0[%arg3, %9] : memref<?x32xf32>
        %11 = arith.addi %9, %c1 : index
        %12 = arith.addi %3, %c38 : index
        %13 = arith.cmpi slt, %12, %c0 : index
        %14 = arith.subi %c-39, %3 : index
        %15 = arith.select %13, %14, %12 : index
        %16 = arith.divsi %15, %c7 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19:2 = scf.for %arg5 = %11 to %18 step %c1 iter_args(%arg6 = %10, %arg7 = %2) -> (f32, i32) {
          %20 = arith.index_cast %arg5 : index to i32
          %21 = memref.load %arg0[%arg3, %arg5] : memref<?x32xf32>
          %22 = arith.cmpf ogt, %21, %arg6 : f32
          %23 = arith.select %22, %20, %arg7 : i32
          %24 = arith.select %22, %21, %arg6 : f32
          scf.yield %24, %23 : f32, i32
        }
        affine.store %19#0, %arg1[%arg3, %arg4] : memref<?x7xf32>
        affine.store %19#1, %arg2[%arg3, %arg4] : memref<?x7xi32>
      }
    }
    return
  }
}

