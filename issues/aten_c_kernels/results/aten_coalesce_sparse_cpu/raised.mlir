module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_coalesce_sparse_cpu(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg4[0] : memref<?xi32>
    affine.for %arg5 = 0 to 512 {
      %0 = affine.load %arg4[0] : memref<?xi32>
      %1 = arith.index_cast %arg5 : index to i32
      %2 = arith.cmpi ne, %1, %c0_i32 : i32
      %3 = affine.load %arg0[%arg5] : memref<?xi32>
      %4 = arith.addi %0, %c-1_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %arg2[%5] : memref<?xi32>
      %7 = arith.cmpi eq, %3, %6 : i32
      %8 = arith.select %2, %7, %false : i1
      %9 = scf.if %8 -> (i32) {
        %10 = arith.addi %0, %c-1_i32 : i32
        %11 = arith.index_cast %10 : i32 to index
        %12 = affine.load %arg1[%arg5] : memref<?xf32>
        %13 = memref.load %arg3[%11] : memref<?xf32>
        %14 = arith.addf %13, %12 : f32
        memref.store %14, %arg3[%11] : memref<?xf32>
        scf.yield %0 : i32
      } else {
        %10 = arith.index_cast %0 : i32 to index
        %11 = affine.load %arg0[%arg5] : memref<?xi32>
        memref.store %11, %arg2[%10] : memref<?xi32>
        %12 = arith.addi %0, %c1_i32 : i32
        %13 = affine.load %arg1[%arg5] : memref<?xf32>
        memref.store %13, %arg3[%10] : memref<?xf32>
        scf.yield %12 : i32
      }
      affine.store %9, %arg4[0] : memref<?xi32>
    }
    return
  }
}

