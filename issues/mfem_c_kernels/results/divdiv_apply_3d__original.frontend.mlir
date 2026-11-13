#set = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 - 1 == 0)>
#set2 = affine_set<(d0) : (d0 - 2 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_divdiv_apply_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c108_i32 = arith.constant 108 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<4xf64>
    %alloca_0 = memref.alloca() : memref<4x4xf64>
    %alloca_1 = memref.alloca() : memref<5xf64>
    %alloca_2 = memref.alloca() : memref<5x5xf64>
    %alloca_3 = memref.alloca() : memref<5x5x5xf64>
    affine.for %arg7 = 0 to 2 {
      %0 = arith.index_cast %arg7 : index to i32
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            affine.store %cst, %alloca_3[%arg8, %arg9, %arg10] : memref<5x5x5xf64>
          }
        }
      }
      %1 = arith.muli %0, %c108_i32 : i32
      %2 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %c0_i32) -> (i32) {
        %3 = arith.index_cast %arg8 : index to i32
        %4 = arith.cmpi eq, %3, %c2_i32 : i32
        %5 = arith.select %4, %c4_i32, %c3_i32 : i32
        %6 = arith.cmpi eq, %3, %c1_i32 : i32
        %7 = arith.select %6, %c4_i32, %c3_i32 : i32
        %8 = arith.cmpi eq, %3, %c0_i32 : i32
        %9 = arith.select %8, %c4_i32, %c3_i32 : i32
        %10 = arith.index_cast %5 : i32 to index
        %11 = arith.index_cast %7 : i32 to index
        %12 = arith.index_cast %9 : i32 to index
        scf.for %arg10 = %c0 to %10 step %c1 {
          %16 = arith.index_cast %arg10 : index to i32
          affine.for %arg11 = 0 to 5 {
            affine.for %arg12 = 0 to 5 {
              affine.store %cst, %alloca_2[%arg11, %arg12] : memref<5x5xf64>
            }
          }
          %17 = arith.muli %16, %7 : i32
          scf.for %arg11 = %c0 to %11 step %c1 {
            %18 = arith.index_cast %arg11 : index to i32
            affine.for %arg12 = 0 to 5 {
              affine.store %cst, %alloca_1[%arg12] : memref<5xf64>
            }
            %19 = arith.addi %18, %17 : i32
            %20 = arith.muli %19, %9 : i32
            scf.for %arg12 = %c0 to %12 step %c1 {
              %21 = arith.index_cast %arg12 : index to i32
              %22 = arith.addi %21, %20 : i32
              %23 = arith.addi %22, %arg9 : i32
              %24 = arith.addi %23, %1 : i32
              %25 = arith.index_cast %24 : i32 to index
              %26 = memref.load %arg5[%25] : memref<?xf64>
              affine.for %arg13 = 0 to 5 {
                %27 = arith.index_cast %arg13 : index to i32
                %28 = affine.if #set(%arg8) -> f64 {
                  %32 = arith.muli %27, %c4_i32 : i32
                  %33 = arith.addi %32, %21 : i32
                  %34 = arith.index_cast %33 : i32 to index
                  %35 = memref.load %arg2[%34] : memref<?xf64>
                  affine.yield %35 : f64
                } else {
                  %32 = arith.muli %27, %c3_i32 : i32
                  %33 = arith.addi %32, %21 : i32
                  %34 = arith.index_cast %33 : i32 to index
                  %35 = memref.load %arg0[%34] : memref<?xf64>
                  affine.yield %35 : f64
                }
                %29 = arith.mulf %26, %28 : f64
                %30 = affine.load %alloca_1[%arg13] : memref<5xf64>
                %31 = arith.addf %30, %29 : f64
                affine.store %31, %alloca_1[%arg13] : memref<5xf64>
              }
            }
            affine.for %arg12 = 0 to 5 {
              %21 = arith.index_cast %arg12 : index to i32
              %22 = affine.if #set1(%arg8) -> f64 {
                %23 = arith.muli %21, %c4_i32 : i32
                %24 = arith.addi %23, %18 : i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %arg2[%25] : memref<?xf64>
                affine.yield %26 : f64
              } else {
                %23 = arith.muli %21, %c3_i32 : i32
                %24 = arith.addi %23, %18 : i32
                %25 = arith.index_cast %24 : i32 to index
                %26 = memref.load %arg0[%25] : memref<?xf64>
                affine.yield %26 : f64
              }
              affine.for %arg13 = 0 to 5 {
                %23 = affine.load %alloca_1[%arg13] : memref<5xf64>
                %24 = arith.mulf %23, %22 : f64
                %25 = affine.load %alloca_2[%arg12, %arg13] : memref<5x5xf64>
                %26 = arith.addf %25, %24 : f64
                affine.store %26, %alloca_2[%arg12, %arg13] : memref<5x5xf64>
              }
            }
          }
          affine.for %arg11 = 0 to 5 {
            %18 = arith.index_cast %arg11 : index to i32
            %19 = arith.muli %18, %c4_i32 : i32
            %20 = arith.addi %19, %16 : i32
            %21 = arith.index_cast %20 : i32 to index
            %22 = arith.muli %18, %c3_i32 : i32
            %23 = arith.addi %22, %16 : i32
            %24 = arith.index_cast %23 : i32 to index
            %25 = affine.if #set2(%arg8) -> f64 {
              %26 = memref.load %arg2[%21] : memref<?xf64>
              affine.yield %26 : f64
            } else {
              %26 = memref.load %arg0[%24] : memref<?xf64>
              affine.yield %26 : f64
            }
            affine.for %arg12 = 0 to 5 {
              affine.for %arg13 = 0 to 5 {
                %26 = affine.load %alloca_2[%arg12, %arg13] : memref<5x5xf64>
                %27 = arith.mulf %26, %25 : f64
                %28 = affine.load %alloca_3[%arg11, %arg12, %arg13] : memref<5x5x5xf64>
                %29 = arith.addf %28, %27 : f64
                affine.store %29, %alloca_3[%arg11, %arg12, %arg13] : memref<5x5x5xf64>
              }
            }
          }
        }
        %13 = arith.muli %9, %7 : i32
        %14 = arith.muli %13, %5 : i32
        %15 = arith.addi %arg9, %14 : i32
        affine.yield %15 : i32
      }
      affine.for %arg8 = 0 to 5 {
        affine.for %arg9 = 0 to 5 {
          affine.for %arg10 = 0 to 5 {
            %3 = affine.load %arg4[%arg7 * 125 + %arg10 + %arg8 * 25 + %arg9 * 5] : memref<?xf64>
            %4 = affine.load %alloca_3[%arg8, %arg9, %arg10] : memref<5x5x5xf64>
            %5 = arith.mulf %4, %3 : f64
            affine.store %5, %alloca_3[%arg8, %arg9, %arg10] : memref<5x5x5xf64>
          }
        }
      }
      affine.for %arg8 = 0 to 5 {
        %3 = arith.index_cast %arg8 : index to i32
        %4 = affine.for %arg9 = 0 to 3 iter_args(%arg10 = %c0_i32) -> (i32) {
          %5 = arith.index_cast %arg9 : index to i32
          %6 = arith.cmpi eq, %5, %c2_i32 : i32
          %7 = arith.select %6, %c4_i32, %c3_i32 : i32
          %8 = arith.cmpi eq, %5, %c1_i32 : i32
          %9 = arith.select %8, %c4_i32, %c3_i32 : i32
          %10 = arith.cmpi eq, %5, %c0_i32 : i32
          %11 = arith.select %10, %c4_i32, %c3_i32 : i32
          %12 = arith.index_cast %9 : i32 to index
          %13 = arith.index_cast %11 : i32 to index
          scf.for %arg11 = %c0 to %12 step %c1 {
            scf.for %arg12 = %c0 to %13 step %c1 {
              memref.store %cst, %alloca_0[%arg11, %arg12] : memref<4x4xf64>
            }
          }
          %14 = arith.cmpi sgt, %13, %c0 : index
          affine.for %arg11 = 0 to 5 {
            %19 = arith.index_cast %arg11 : index to i32
            scf.for %arg12 = %c0 to %13 step %c1 {
              memref.store %cst, %alloca[%arg12] : memref<4xf64>
            }
            affine.for %arg12 = 0 to 5 {
              %20 = arith.index_cast %arg12 : index to i32
              %21 = affine.load %alloca_3[%arg8, %arg11, %arg12] : memref<5x5x5xf64>
              scf.for %arg13 = %c0 to %13 step %c1 {
                %22 = arith.index_cast %arg13 : index to i32
                %23 = affine.if #set(%arg9) -> f64 {
                  %27 = arith.muli %22, %c5_i32 : i32
                  %28 = arith.addi %27, %20 : i32
                  %29 = arith.index_cast %28 : i32 to index
                  %30 = memref.load %arg3[%29] : memref<?xf64>
                  affine.yield %30 : f64
                } else {
                  %27 = arith.muli %22, %c5_i32 : i32
                  %28 = arith.addi %27, %20 : i32
                  %29 = arith.index_cast %28 : i32 to index
                  %30 = memref.load %arg1[%29] : memref<?xf64>
                  affine.yield %30 : f64
                }
                %24 = arith.mulf %21, %23 : f64
                %25 = memref.load %alloca[%arg13] : memref<4xf64>
                %26 = arith.addf %25, %24 : f64
                memref.store %26, %alloca[%arg13] : memref<4xf64>
              }
            }
            scf.for %arg12 = %c0 to %12 step %c1 {
              scf.if %14 {
                %20 = arith.index_cast %arg12 : index to i32
                %21 = affine.if #set1(%arg9) -> f64 {
                  %22 = arith.muli %20, %c5_i32 : i32
                  %23 = arith.addi %22, %19 : i32
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %arg3[%24] : memref<?xf64>
                  affine.yield %25 : f64
                } else {
                  %22 = arith.muli %20, %c5_i32 : i32
                  %23 = arith.addi %22, %19 : i32
                  %24 = arith.index_cast %23 : i32 to index
                  %25 = memref.load %arg1[%24] : memref<?xf64>
                  affine.yield %25 : f64
                }
                scf.for %arg13 = %c0 to %13 step %c1 {
                  %22 = memref.load %alloca[%arg13] : memref<4xf64>
                  %23 = arith.mulf %22, %21 : f64
                  %24 = memref.load %alloca_0[%arg12, %arg13] : memref<4x4xf64>
                  %25 = arith.addf %24, %23 : f64
                  memref.store %25, %alloca_0[%arg12, %arg13] : memref<4x4xf64>
                }
              }
            }
          }
          %15 = arith.index_cast %7 : i32 to index
          scf.for %arg11 = %c0 to %15 step %c1 {
            %19 = arith.index_cast %arg11 : index to i32
            %20 = arith.muli %19, %9 : i32
            %21 = arith.muli %19, %c5_i32 : i32
            %22 = arith.addi %21, %3 : i32
            %23 = arith.index_cast %22 : i32 to index
            scf.for %arg12 = %c0 to %12 step %c1 {
              %24 = arith.index_cast %arg12 : index to i32
              %25 = arith.addi %24, %20 : i32
              %26 = arith.muli %25, %11 : i32
              scf.for %arg13 = %c0 to %13 step %c1 {
                %27 = arith.index_cast %arg13 : index to i32
                %28 = arith.addi %27, %26 : i32
                %29 = arith.addi %28, %arg10 : i32
                %30 = arith.addi %29, %1 : i32
                %31 = arith.index_cast %30 : i32 to index
                %32 = memref.load %alloca_0[%arg12, %arg13] : memref<4x4xf64>
                %33 = affine.if #set2(%arg9) -> f64 {
                  %37 = memref.load %arg3[%23] : memref<?xf64>
                  affine.yield %37 : f64
                } else {
                  %37 = memref.load %arg1[%23] : memref<?xf64>
                  affine.yield %37 : f64
                }
                %34 = arith.mulf %32, %33 : f64
                %35 = memref.load %arg6[%31] : memref<?xf64>
                %36 = arith.addf %35, %34 : f64
                memref.store %36, %arg6[%31] : memref<?xf64>
              }
            }
          }
          %16 = arith.muli %11, %9 : i32
          %17 = arith.muli %16, %7 : i32
          %18 = arith.addi %arg10, %17 : i32
          affine.yield %18 : i32
        }
      }
    }
    return
  }
}
