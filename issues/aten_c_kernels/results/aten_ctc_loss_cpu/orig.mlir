#set = affine_set<(d0) : (d0 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ctc_loss_cpu(%arg0: memref<?x4x12xf32>, %arg1: memref<?x5xi32>, %arg2: i32, %arg3: memref<?xf32>, %arg4: memref<?x24x11xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c-2_i32 = arith.constant -2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 24 {
        affine.for %arg7 = 0 to 11 {
          affine.store %cst, %arg4[%arg5, %arg6, %arg7] : memref<?x24x11xf32>
        }
      }
      %1 = affine.load %arg0[0, %arg5, symbol(%0)] : memref<?x4x12xf32>
      %2 = math.exp %1 : f32
      affine.store %2, %arg4[%arg5, 0, 0] : memref<?x24x11xf32>
      %3 = affine.load %arg1[%arg5, 0] : memref<?x5xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %arg0[%c0, %arg5, %4] : memref<?x4x12xf32>
      %6 = math.exp %5 : f32
      affine.store %6, %arg4[%arg5, 0, 1] : memref<?x24x11xf32>
      affine.for %arg6 = 1 to 24 {
        affine.for %arg7 = 0 to 11 {
          %12 = arith.index_cast %arg7 : index to i32
          %13 = arith.andi %12, %c1_i32 : i32
          %14 = arith.cmpi ne, %13, %c0_i32 : i32
          %15 = scf.if %14 -> (i32) {
            %26 = arith.cmpi slt, %arg7, %c0 : index
            %27 = arith.subi %c-1, %arg7 : index
            %28 = arith.select %26, %27, %arg7 : index
            %29 = arith.divsi %28, %c2 : index
            %30 = arith.subi %c-1, %29 : index
            %31 = arith.select %26, %30, %29 : index
            %32 = memref.load %arg1[%arg5, %31] : memref<?x5xi32>
            scf.yield %32 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %16 = affine.load %arg4[%arg5, %arg6 - 1, %arg7] : memref<?x24x11xf32>
          %17 = affine.if #set(%arg7) -> f32 {
            %26 = affine.load %arg4[%arg5, %arg6 - 1, %arg7 - 1] : memref<?x24x11xf32>
            %27 = arith.addf %16, %26 : f32
            affine.yield %27 : f32
          } else {
            affine.yield %16 : f32
          }
          %18 = arith.cmpi sgt, %12, %c1_i32 : i32
          %19 = arith.cmpi ne, %15, %arg2 : i32
          %20 = arith.andi %18, %19 : i1
          %21 = scf.if %20 -> (f32) {
            %26 = arith.addi %12, %c-2_i32 : i32
            %27 = arith.andi %26, %c1_i32 : i32
            %28 = arith.cmpi ne, %27, %c0_i32 : i32
            %29 = scf.if %28 -> (i32) {
              %32 = arith.cmpi slt, %arg7, %c0 : index
              %33 = arith.subi %c-1, %arg7 : index
              %34 = arith.select %32, %33, %arg7 : index
              %35 = arith.divsi %34, %c2 : index
              %36 = arith.subi %c-1, %35 : index
              %37 = arith.select %32, %36, %35 : index
              %38 = arith.addi %37, %c-1 : index
              %39 = memref.load %arg1[%arg5, %38] : memref<?x5xi32>
              scf.yield %39 : i32
            } else {
              scf.yield %arg2 : i32
            }
            %30 = arith.cmpi ne, %15, %29 : i32
            %31 = scf.if %30 -> (f32) {
              %32 = affine.load %arg4[%arg5, %arg6 - 1, %arg7 - 2] : memref<?x24x11xf32>
              %33 = arith.addf %17, %32 : f32
              scf.yield %33 : f32
            } else {
              scf.yield %17 : f32
            }
            scf.yield %31 : f32
          } else {
            scf.yield %17 : f32
          }
          %22 = arith.index_cast %15 : i32 to index
          %23 = memref.load %arg0[%arg6, %arg5, %22] : memref<?x4x12xf32>
          %24 = math.exp %23 : f32
          %25 = arith.mulf %21, %24 : f32
          affine.store %25, %arg4[%arg5, %arg6, %arg7] : memref<?x24x11xf32>
        }
      }
      %7 = affine.load %arg4[%arg5, 23, 10] : memref<?x24x11xf32>
      %8 = affine.load %arg4[%arg5, 23, 9] : memref<?x24x11xf32>
      %9 = arith.addf %7, %8 : f32
      %10 = func.call @logf(%9) : (f32) -> f32
      %11 = arith.negf %10 : f32
      affine.store %11, %arg3[%arg5] : memref<?xf32>
    }
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>}
}
