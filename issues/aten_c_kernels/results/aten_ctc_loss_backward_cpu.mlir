#set = affine_set<(d0) : (-d0 + 9 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 8 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_backward_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?x24x11xf32>, %arg4: memref<?xf32>, %arg5: memref<?x4x12xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-2 = arith.constant -2 : index
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c22 = arith.constant 22 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<24x11xf32>
    affine.for %arg6 = 0 to 24 {
      affine.for %arg7 = 0 to 4 {
        affine.for %arg8 = 0 to 12 {
          %0 = affine.load %arg0[%arg6, %arg7, %arg8] : memref<?x4x12xf32>
          %1 = math.exp %0 : f32
          %2 = affine.load %arg4[%arg7] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          affine.store %3, %arg5[%arg6, %arg7, %arg8] : memref<?x4x12xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 4 {
      affine.for %arg7 = 0 to 24 {
        affine.for %arg8 = 0 to 11 {
          affine.store %cst_0, %alloca[%arg7, %arg8] : memref<24x11xf32>
        }
      }
      affine.store %cst, %alloca[23, 9] : memref<24x11xf32>
      affine.store %cst, %alloca[23, 10] : memref<24x11xf32>
      affine.for %arg7 = 0 to 23 {
        %3 = arith.subi %c22, %arg7 : index
        %4 = arith.index_cast %3 : index to i32
        %5 = arith.addi %4, %c1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        affine.for %arg8 = 0 to 11 {
          %7 = arith.index_cast %arg8 : index to i32
          %8 = arith.andi %7, %c1_i32 : i32
          %9 = arith.cmpi ne, %8, %c0_i32 : i32
          %10 = scf.if %9 -> (i32) {
            %20 = arith.cmpi slt, %arg8, %c0 : index
            %21 = arith.subi %c-1, %arg8 : index
            %22 = arith.select %20, %21, %arg8 : index
            %23 = arith.divsi %22, %c2 : index
            %24 = arith.subi %c-1, %23 : index
            %25 = arith.select %20, %24, %23 : index
            %26 = memref.load %arg1[%arg6, %25] : memref<?x5xi32>
            scf.yield %26 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %11 = affine.load %alloca[-%arg7 + 23, %arg8] : memref<24x11xf32>
          %12 = arith.index_cast %10 : i32 to index
          %13 = memref.load %arg0[%6, %arg6, %12] : memref<?x4x12xf32>
          %14 = math.exp %13 : f32
          %15 = arith.mulf %11, %14 : f32
          %16 = arith.addi %7, %c1_i32 : i32
          %17 = affine.if #set(%arg8) -> f32 {
            %20 = arith.andi %16, %c1_i32 : i32
            %21 = arith.cmpi ne, %20, %c0_i32 : i32
            %22 = scf.if %21 -> (i32) {
              %29 = arith.addi %arg8, %c1 : index
              %30 = arith.cmpi slt, %29, %c0 : index
              %31 = arith.subi %c-2, %arg8 : index
              %32 = arith.select %30, %31, %29 : index
              %33 = arith.divsi %32, %c2 : index
              %34 = arith.subi %c-1, %33 : index
              %35 = arith.select %30, %34, %33 : index
              %36 = memref.load %arg1[%arg6, %35] : memref<?x5xi32>
              scf.yield %36 : i32
            } else {
              scf.yield %arg2 : i32
            }
            %23 = affine.load %alloca[-%arg7 + 23, %arg8 + 1] : memref<24x11xf32>
            %24 = arith.index_cast %22 : i32 to index
            %25 = memref.load %arg0[%6, %arg6, %24] : memref<?x4x12xf32>
            %26 = math.exp %25 : f32
            %27 = arith.mulf %23, %26 : f32
            %28 = arith.addf %15, %27 : f32
            affine.yield %28 : f32
          } else {
            affine.yield %15 : f32
          }
          %18 = arith.addi %7, %c2_i32 : i32
          %19 = affine.if #set1(%arg8) -> f32 {
            %20 = arith.andi %18, %c1_i32 : i32
            %21 = arith.cmpi ne, %20, %c0_i32 : i32
            %22 = scf.if %21 -> (i32) {
              %27 = arith.cmpi slt, %arg8, %c0 : index
              %28 = arith.subi %c-1, %arg8 : index
              %29 = arith.select %27, %28, %arg8 : index
              %30 = arith.divsi %29, %c2 : index
              %31 = arith.subi %c-1, %30 : index
              %32 = arith.select %27, %31, %30 : index
              %33 = arith.addi %32, %c1 : index
              %34 = memref.load %arg1[%arg6, %33] : memref<?x5xi32>
              scf.yield %34 : i32
            } else {
              scf.yield %arg2 : i32
            }
            %23 = arith.cmpi ne, %10, %arg2 : i32
            %24 = arith.cmpi ne, %10, %22 : i32
            %25 = arith.andi %23, %24 : i1
            %26 = scf.if %25 -> (f32) {
              %27 = affine.load %alloca[-%arg7 + 23, %arg8 + 2] : memref<24x11xf32>
              %28 = arith.index_cast %22 : i32 to index
              %29 = memref.load %arg0[%6, %arg6, %28] : memref<?x4x12xf32>
              %30 = math.exp %29 : f32
              %31 = arith.mulf %27, %30 : f32
              %32 = arith.addf %17, %31 : f32
              scf.yield %32 : f32
            } else {
              scf.yield %17 : f32
            }
            affine.yield %26 : f32
          } else {
            affine.yield %17 : f32
          }
          affine.store %19, %alloca[-%arg7 + 22, %arg8] : memref<24x11xf32>
        }
      }
      %0 = affine.load %arg3[%arg6, 23, 10] : memref<?x24x11xf32>
      %1 = affine.load %arg3[%arg6, 23, 9] : memref<?x24x11xf32>
      %2 = arith.addf %0, %1 : f32
      affine.for %arg7 = 0 to 24 {
        affine.for %arg8 = 0 to 11 {
          %3 = arith.index_cast %arg8 : index to i32
          %4 = arith.andi %3, %c1_i32 : i32
          %5 = arith.cmpi ne, %4, %c0_i32 : i32
          %6 = scf.if %5 -> (i32) {
            %16 = arith.cmpi slt, %arg8, %c0 : index
            %17 = arith.subi %c-1, %arg8 : index
            %18 = arith.select %16, %17, %arg8 : index
            %19 = arith.divsi %18, %c2 : index
            %20 = arith.subi %c-1, %19 : index
            %21 = arith.select %16, %20, %19 : index
            %22 = memref.load %arg1[%arg6, %21] : memref<?x5xi32>
            scf.yield %22 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %7 = arith.index_cast %6 : i32 to index
          %8 = affine.load %arg4[%arg6] : memref<?xf32>
          %9 = affine.load %arg3[%arg6, %arg7, %arg8] : memref<?x24x11xf32>
          %10 = arith.mulf %8, %9 : f32
          %11 = affine.load %alloca[%arg7, %arg8] : memref<24x11xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.divf %12, %2 : f32
          %14 = memref.load %arg5[%arg7, %arg6, %7] : memref<?x4x12xf32>
          %15 = arith.subf %14, %13 : f32
          memref.store %15, %arg5[%arg7, %arg6, %7] : memref<?x4x12xf32>
        }
      }
    }
    return
  }
}
