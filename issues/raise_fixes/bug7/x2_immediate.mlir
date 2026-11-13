module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @x2(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
    %c0 = arith.constant 0 : index
    %0 = "polygeist.undef"() : () -> i32
    memref.store %0, %cast[%c0] : memref<?xi32>
    %alloca_0 = memref.alloca() : memref<1xi32>
    %cast_1 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32>
    %c0_2 = arith.constant 0 : index
    %1 = "polygeist.undef"() : () -> i32
    memref.store %1, %cast_1[%c0_2] : memref<?xi32>
    %alloca_3 = memref.alloca() : memref<1xi32>
    %cast_4 = memref.cast %alloca_3 : memref<1xi32> to memref<?xi32>
    %c0_5 = arith.constant 0 : index
    %2 = "polygeist.undef"() : () -> i32
    memref.store %2, %cast_4[%c0_5] : memref<?xi32>
    %alloca_6 = memref.alloca() : memref<1xi32>
    %cast_7 = memref.cast %alloca_6 : memref<1xi32> to memref<?xi32>
    %c0_8 = arith.constant 0 : index
    %3 = "polygeist.undef"() : () -> i32
    memref.store %3, %cast_7[%c0_8] : memref<?xi32>
    %c0_9 = arith.constant 0 : index
    memref.store %arg0, %cast_7[%c0_9] : memref<?xi32>
    %true = arith.constant true
    %alloca_10 = memref.alloca() : memref<i1>
    %alloca_11 = memref.alloca() : memref<i1>
    memref.store %true, %alloca_11[] : memref<i1>
    memref.store %true, %alloca_10[] : memref<i1>
    %4 = memref.load %alloca_10[] : memref<i1>
    scf.if %4 {
      scf.execute_region {
        %6 = memref.load %alloca_10[] : memref<i1>
        scf.if %6 {
          scf.execute_region {
            %c1_i32 = arith.constant 1 : i32
            %c0_i32 = arith.constant 0 : i32
            %7 = arith.cmpi ne, %c1_i32, %c0_i32 : i32
            scf.if %7 {
              %8 = memref.load %alloca_10[] : memref<i1>
              scf.if %8 {
                scf.execute_region {
                  %false = arith.constant false
                  memref.store %false, %alloca_10[] : memref<i1>
                  memref.store %false, %alloca_11[] : memref<i1>
                  scf.yield
                }
              }
            }
            scf.yield
          }
        }
        scf.yield
      }
    }
    %5 = memref.load %alloca_10[] : memref<i1>
    scf.if %5 {
      scf.execute_region {
        %6 = memref.load %alloca_10[] : memref<i1>
        scf.if %6 {
          scf.execute_region {
            %7 = memref.load %alloca_10[] : memref<i1>
            scf.if %7 {
              scf.execute_region {
                %c0_i32_17 = arith.constant 0 : i32
                %c0_18 = arith.constant 0 : index
                memref.store %c0_i32_17, %cast_4[%c0_18] : memref<?xi32>
                scf.yield
              }
            }
            %true_12 = arith.constant true
            %alloca_13 = memref.alloca() : memref<i1>
            %alloca_14 = memref.alloca() : memref<i1>
            memref.store %true_12, %alloca_14[] : memref<i1>
            cf.br ^bb1
          ^bb1:  // 2 preds: ^bb0, ^bb2
            %c0_15 = arith.constant 0 : index
            %8 = memref.load %cast_4[%c0_15] : memref<?xi32>
            %c0_16 = arith.constant 0 : index
            %9 = memref.load %cast_7[%c0_16] : memref<?xi32>
            %10 = arith.cmpi slt, %8, %9 : i32
            %11 = arith.extui %10 : i1 to i32
            %c0_i32 = arith.constant 0 : i32
            %12 = arith.cmpi ne, %11, %c0_i32 : i32
            %13 = memref.load %alloca_14[] : memref<i1>
            %14 = arith.andi %12, %13 : i1
            cf.cond_br %14, ^bb2, ^bb3
          ^bb2:  // pred: ^bb1
            %15 = memref.load %alloca_14[] : memref<i1>
            memref.store %15, %alloca_13[] : memref<i1>
            %16 = memref.load %alloca_13[] : memref<i1>
            scf.if %16 {
              scf.execute_region {
                %19 = memref.load %alloca_13[] : memref<i1>
                scf.if %19 {
                  scf.execute_region {
                    %c0_i32_23 = arith.constant 0 : i32
                    %c0_24 = arith.constant 0 : index
                    memref.store %c0_i32_23, %cast_1[%c0_24] : memref<?xi32>
                    scf.yield
                  }
                }
                %true_17 = arith.constant true
                %alloca_18 = memref.alloca() : memref<i1>
                %alloca_19 = memref.alloca() : memref<i1>
                memref.store %true_17, %alloca_19[] : memref<i1>
                cf.br ^bb1
              ^bb1:  // 2 preds: ^bb0, ^bb2
                %c0_20 = arith.constant 0 : index
                %20 = memref.load %cast_1[%c0_20] : memref<?xi32>
                %c0_21 = arith.constant 0 : index
                %21 = memref.load %cast_7[%c0_21] : memref<?xi32>
                %22 = arith.cmpi slt, %20, %21 : i32
                %23 = arith.extui %22 : i1 to i32
                %c0_i32_22 = arith.constant 0 : i32
                %24 = arith.cmpi ne, %23, %c0_i32_22 : i32
                %25 = memref.load %alloca_19[] : memref<i1>
                %26 = arith.andi %24, %25 : i1
                cf.cond_br %26, ^bb2, ^bb3
              ^bb2:  // pred: ^bb1
                %27 = memref.load %alloca_19[] : memref<i1>
                memref.store %27, %alloca_18[] : memref<i1>
                %28 = memref.load %alloca_18[] : memref<i1>
                scf.if %28 {
                  scf.execute_region {
                    %31 = memref.load %alloca_18[] : memref<i1>
                    scf.if %31 {
                      scf.execute_region {
                        %c0_i32_29 = arith.constant 0 : i32
                        %c0_30 = arith.constant 0 : index
                        memref.store %c0_i32_29, %cast[%c0_30] : memref<?xi32>
                        scf.yield
                      }
                    }
                    %true_23 = arith.constant true
                    %alloca_24 = memref.alloca() : memref<i1>
                    %alloca_25 = memref.alloca() : memref<i1>
                    memref.store %true_23, %alloca_25[] : memref<i1>
                    cf.br ^bb1
                  ^bb1:  // 2 preds: ^bb0, ^bb2
                    %c0_26 = arith.constant 0 : index
                    %32 = memref.load %cast[%c0_26] : memref<?xi32>
                    %c0_27 = arith.constant 0 : index
                    %33 = memref.load %cast_7[%c0_27] : memref<?xi32>
                    %34 = arith.cmpi slt, %32, %33 : i32
                    %35 = arith.extui %34 : i1 to i32
                    %c0_i32_28 = arith.constant 0 : i32
                    %36 = arith.cmpi ne, %35, %c0_i32_28 : i32
                    %37 = memref.load %alloca_25[] : memref<i1>
                    %38 = arith.andi %36, %37 : i1
                    cf.cond_br %38, ^bb2, ^bb3
                  ^bb2:  // pred: ^bb1
                    %39 = memref.load %alloca_25[] : memref<i1>
                    memref.store %39, %alloca_24[] : memref<i1>
                    %40 = memref.load %alloca_25[] : memref<i1>
                    memref.store %40, %alloca_24[] : memref<i1>
                    %41 = memref.load %alloca_24[] : memref<i1>
                    scf.if %41 {
                      scf.execute_region {
                        %c0_29 = arith.constant 0 : index
                        %42 = memref.load %cast[%c0_29] : memref<?xi32>
                        %c1_i32 = arith.constant 1 : i32
                        %43 = arith.addi %42, %c1_i32 : i32
                        %c0_30 = arith.constant 0 : index
                        memref.store %43, %cast[%c0_30] : memref<?xi32>
                        scf.yield
                      }
                    }
                    cf.br ^bb1
                  ^bb3:  // pred: ^bb1
                    scf.yield
                  }
                }
                %29 = memref.load %alloca_19[] : memref<i1>
                memref.store %29, %alloca_18[] : memref<i1>
                %30 = memref.load %alloca_18[] : memref<i1>
                scf.if %30 {
                  scf.execute_region {
                    %c0_23 = arith.constant 0 : index
                    %31 = memref.load %cast_1[%c0_23] : memref<?xi32>
                    %c1_i32 = arith.constant 1 : i32
                    %32 = arith.addi %31, %c1_i32 : i32
                    %c0_24 = arith.constant 0 : index
                    memref.store %32, %cast_1[%c0_24] : memref<?xi32>
                    scf.yield
                  }
                }
                cf.br ^bb1
              ^bb3:  // pred: ^bb1
                scf.yield
              }
            }
            %17 = memref.load %alloca_14[] : memref<i1>
            memref.store %17, %alloca_13[] : memref<i1>
            %18 = memref.load %alloca_13[] : memref<i1>
            scf.if %18 {
              scf.execute_region {
                %c0_17 = arith.constant 0 : index
                %19 = memref.load %cast_4[%c0_17] : memref<?xi32>
                %c1_i32 = arith.constant 1 : i32
                %20 = arith.addi %19, %c1_i32 : i32
                %c0_18 = arith.constant 0 : index
                memref.store %20, %cast_4[%c0_18] : memref<?xi32>
                scf.yield
              }
            }
            cf.br ^bb1
          ^bb3:  // pred: ^bb1
            scf.yield
          }
        }
        scf.yield
      }
    }
    return
  }
}
