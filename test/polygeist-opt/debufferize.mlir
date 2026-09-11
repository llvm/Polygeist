//polygeist-opt --canonicalize --linalg-debufferize --canonicalize debufferize.mlir 

#map16 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map17 = affine_map<(d0, d1, d2, d3) -> (d1 + d3, d0 + d2)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map22 = affine_map<(d0, d1) -> (d1, d0)>

 module @in_place_add{
     func.func @in_place_add(%value: f32) {
       %c0 = arith.constant 0 : index
       %buffer = memref.alloca() : memref<128xf32> 
       linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]
       } ins(%buffer : memref<128xf32>) 
         outs(%buffer : memref<128xf32>) {
       ^bb0(%in: f32, %out: f32):
         %sum = arith.addf %in, %value : f32
         linalg.yield %sum : f32
       }
       return
     }
 }

  module @in_place_add2{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32) {
        %c0 = arith.constant 0 : index
        //%buffer = memref.alloca() : memref<128xf32> 
        linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%buffer : memref<128xf32>) 
          outs(%buffer : memref<128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %sum = arith.addf %in, %value : f32
          linalg.yield %sum : f32
        }
        return
      }
  }

  module @in_place_cond_add{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        //%buffer = memref.alloca() : memref<128xf32>
        scf.if %cond {
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buffer : memref<128xf32>) 
            outs(%buffer : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            linalg.yield %sum : f32
          }
        }
        return
      }
  }

  module @in_place_add_for{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        //%buffer = memref.alloca() : memref<128xf32>
        scf.for %i = %c0 to %c10 step %c1 {
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buffer : memref<128xf32>) 
            outs(%buffer : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            linalg.yield %sum : f32
          }
        }
        return
      }
  }

  //Case when buffer is captured
  module @in_place_add_for_loop_carried{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        //%buffer = memref.alloca() : memref<128xf32>
        %result =  scf.for %i = %c0 to %c10 step %c1 iter_args(%buf = %buffer) -> (memref<128xf32>) {
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buf : memref<128xf32>) 
            outs(%buf : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            linalg.yield %sum : f32
          }
          scf.yield %buf : memref<128xf32>
        }
        return
      }
  }
  module @cross_buffer_add{
      func.func @in_place_add(%buf:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        %buf2 = memref.alloca() : memref<128xf32>
        linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%buf : memref<128xf32>) 
          outs(%buf2 : memref<128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %sum = arith.addf %in, %value : f32
          linalg.yield %sum : f32
        }
        linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%buf2 : memref<128xf32>) 
          outs(%buf : memref<128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %sum = arith.addf %in, %value : f32
          %sum2 = arith.addf %sum, %value : f32
          linalg.yield %sum2 : f32
        }
        return
      }
  }
  
  module @in_place_add_for_loop_carried_cross_buffer{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        %buffer2 = memref.alloca() : memref<128xf32>
        %result:2 =  scf.for %i = %c0 to %c10 step %c1 iter_args(%buf = %buffer, %buf2 = %buffer2) -> (memref<128xf32>, memref<128xf32>) {
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buf : memref<128xf32>) 
            outs(%buf2 : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            linalg.yield %sum : f32
          }
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buf2 : memref<128xf32>) 
            outs(%buf : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            %sum2 = arith.addf %sum, %value : f32
            linalg.yield %sum2 : f32
          }
          scf.yield %buf, %buf2 : memref<128xf32>, memref<128xf32>
        }
        return
      }
  }

//    //TODO: Doesn't bufferize --affine loop carried iter_args doesn't canonicalizes (missing pattern?)
//    module @in_place_add_for_loop_carried3{
//       func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
//         %c0 = arith.constant 0 : index
//         %c1 = arith.constant 1 : index
//         %c10 = arith.constant 10 : index
//         %buffer2 = memref.alloca() : memref<128xf32>
//         %result:2 =  affine.for %i = %c0 to %c10 iter_args(%buf = %buffer, %buf2 = %buffer2) -> (memref<128xf32>, memref<128xf32>) {
//           linalg.generic {
//             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
//             iterator_types = ["parallel"]
//           } ins(%buf : memref<128xf32>) 
//             outs(%buf2 : memref<128xf32>) {
//           ^bb0(%in: f32, %out: f32):
//             %sum = arith.addf %in, %value : f32
//             linalg.yield %sum : f32
//           }
//           linalg.generic {
//             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
//             iterator_types = ["parallel"]
//           } ins(%buf2 : memref<128xf32>) 
//             outs(%buf : memref<128xf32>) {
//           ^bb0(%in: f32, %out: f32):
//             %sum = arith.addf %in, %value : f32
//             %sum2 = arith.addf %sum, %value : f32
//             linalg.yield %sum2 : f32
//           }
//           affine.yield %buf, %buf2 : memref<128xf32>, memref<128xf32>
//         }
//         return
//       }
//   }

//   module @in_place_add_for_loop_affine{
//       func.func @in_place_add(%buf:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
//         %c0 = arith.constant 0 : index
//         %c1 = arith.constant 1 : index
//         %c10 = arith.constant 10 : index
//         %buf2 = memref.alloca() : memref<128xf32>
//         affine.for %i = %c0 to %c10 {
//           linalg.generic {
//             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
//             iterator_types = ["parallel"]
//           } ins(%buf : memref<128xf32>) 
//             outs(%buf2 : memref<128xf32>) {
//           ^bb0(%in: f32, %out: f32):
//             %sum = arith.addf %in, %value : f32
//             linalg.yield %sum : f32
//           }
//           linalg.generic {
//             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
//             iterator_types = ["parallel"]
//           } ins(%buf2 : memref<128xf32>) 
//             outs(%buf : memref<128xf32>) {
//           ^bb0(%in: f32, %out: f32):
//             %sum = arith.addf %in, %value : f32
//             %sum2 = arith.addf %sum, %value : f32
//             linalg.yield %sum2 : f32
//           }
//         }
//         return
//       }
//   }


     module @in_place_cond_add_followed_by_add{
       func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
         %c0 = arith.constant 0 : index
         //%buffer = memref.alloca() : memref<128xf32>
         scf.if %cond {
           linalg.generic {
             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]
           } ins(%buffer : memref<128xf32>) 
             outs(%buffer : memref<128xf32>) {
           ^bb0(%in: f32, %out: f32):
             %sum = arith.addf %in, %value : f32
             %sum2 = arith.addf %sum, %value : f32
             linalg.yield %sum2 : f32
           }
         }
         linalg.generic {
           indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
           iterator_types = ["parallel"]
         } ins(%buffer : memref<128xf32>) 
           outs(%buffer : memref<128xf32>) {
         ^bb0(%in: f32, %out: f32):
           %sum = arith.addf %in, %value : f32
           linalg.yield %sum : f32
         }
         return
       }
   }

     module @in_place_cond_add_followed_by_add2{
       func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1, %cond2: i1) {
         %c0 = arith.constant 0 : index
         //%buffer = memref.alloca() : memref<128xf32>
         scf.if %cond2 {
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buffer : memref<128xf32>) 
            outs(%buffer : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            %sum2 = arith.addf %sum, %value : f32
            %sum3 = arith.addf %sum2, %value : f32
            linalg.yield %sum3 : f32
          }
          scf.if %cond {
            linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
              iterator_types = ["parallel"]
            } ins(%buffer : memref<128xf32>) 
              outs(%buffer : memref<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %sum = arith.addf %in, %value : f32
              %sum2 = arith.addf %sum, %value : f32
              linalg.yield %sum2 : f32
            }
          }
         }
          scf.if %cond {
            linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
              iterator_types = ["parallel"]
            } ins(%buffer : memref<128xf32>) 
              outs(%buffer : memref<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %sum = arith.addf %in, %value : f32
              %sum2 = arith.addf %sum, %value : f32
              linalg.yield %sum2 : f32
            }
          }
         linalg.generic {
           indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
           iterator_types = ["parallel"]
         } ins(%buffer : memref<128xf32>) 
           outs(%buffer : memref<128xf32>) {
         ^bb0(%in: f32, %out: f32):
           %sum = arith.addf %in, %value : f32
           linalg.yield %sum : f32
         }
         return
       }
   }

    module @in_place_cond_add_followed_by_add3{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1, %cond2: i1) {
        %c0 = arith.constant 0 : index
        //%buffer = memref.alloca() : memref<128xf32>
        scf.if %cond2 {
         scf.if %cond {
           linalg.generic {
             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]
           } ins(%buffer : memref<128xf32>) 
             outs(%buffer : memref<128xf32>) {
           ^bb0(%in: f32, %out: f32):
             %sum = arith.addf %in, %value : f32
             %sum2 = arith.addf %sum, %value : f32
             linalg.yield %sum2 : f32
           }
         }
        }
        scf.if %cond2 {
           linalg.generic {
             indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]
           } ins(%buffer : memref<128xf32>) 
             outs(%buffer : memref<128xf32>) {
           ^bb0(%in: f32, %out: f32):
             %sum = arith.addf %in, %value : f32
             %sum2 = arith.addf %sum, %value : f32
             %sum3 = arith.addf %sum2, %value : f32
             linalg.yield %sum3 : f32
           }
        }
        linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%buffer : memref<128xf32>) 
          outs(%buffer : memref<128xf32>) {
        ^bb0(%in: f32, %out: f32):
          %sum = arith.addf %in, %value : f32
          linalg.yield %sum : f32
        }
        return
      }
    }

  module @conv_2 {
   func.func @main(%0: memref<515x67xi32> {llvm.noalias}, %1: memref<4x4xi32> {llvm.noalias}, %2: memref<512x64xi32> {llvm.noalias}) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
     %c0_i32 = arith.constant 0 : i32
     linalg.generic {indexing_maps = [#map17, #map18, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : memref<515x67xi32>, memref<4x4xi32>) outs(%2 : memref<512x64xi32>) {
     ^bb0(%in: i32, %in_0: i32, %out: i32):
       %3 = arith.muli %in, %in_0 : i32
       %4 = arith.addi %out, %3 : i32
       linalg.yield %4 : i32
     }
     return %c0_i32 : i32
   }
  }

 module @harris_score_with_gradient_extra_kernel {
     //memref.global "private" @_ZL8coeffs_1 : memref<5x5xi32> = dense<1>
     //memref.global "private" @_ZL8coeffs_y : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
     //memref.global "private" @_ZL8coeffs_x : memref<3x3xi32> = dense<[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]>
     func.func @main(%input: memref<518x518xi32>, %0: memref<3x3xi32> {llvm.noalias}, %1: memref<3x3xi32> {llvm.noalias}, %2: memref<5x5xi32> {llvm.noalias}, %score: memref<512x512xi32> {llvm.noalias}) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
       %c4_i32 = arith.constant 4 : i32
       %c0_i32 = arith.constant 0 : i32
       %alloca = memref.alloca() : memref<512x512xi32>
       %alloca_0 = memref.alloca() : memref<512x512xi32>
       %alloca_1 = memref.alloca() : memref<512x512xi32>
       %alloca_2 = memref.alloca() : memref<516x516xi32>
       %alloca_3 = memref.alloca() : memref<516x516xi32>
       //%score    = memref.alloca() : memref<512x512xi32>
       //%0 = memref.get_global @_ZL8coeffs_x : memref<3x3xi32>
       //%1 = memref.get_global @_ZL8coeffs_y : memref<3x3xi32>
       //%2 = memref.get_global @_ZL8coeffs_1 : memref<5x5xi32>
       linalg.generic {indexing_maps = [#map17, #map18, #map18, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%input, %0, %1 : memref<518x518xi32>, memref<3x3xi32>, memref<3x3xi32>) outs(%alloca_2, %alloca_3 : memref<516x516xi32>, memref<516x516xi32>) {
       ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32):
         %4 = arith.muli %in, %in_5 : i32
         %5 = arith.addi %out_7, %4 : i32
         %6 = arith.muli %in, %in_6 : i32
         %7 = arith.addi %out, %6 : i32
         linalg.yield %7, %5 : i32, i32
       }
       linalg.generic {indexing_maps = [#map17, #map17, #map18, #map19, #map19, #map19], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%alloca_3, %alloca_2, %2 : memref<516x516xi32>, memref<516x516xi32>, memref<5x5xi32>) outs(%alloca, %alloca_0, %alloca_1 : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) {
       ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32, %out_7: i32, %out_8: i32):
         %4 = arith.muli %in, %in : i32
         %5 = arith.muli %4, %in_6 : i32
         %6 = arith.addi %out_8, %5 : i32
         %7 = arith.muli %in_5, %in_5 : i32
         %8 = arith.muli %7, %in_6 : i32
         %9 = arith.addi %out_7, %8 : i32
         %10 = arith.muli %in, %in_5 : i32
         %11 = arith.muli %10, %in_6 : i32
         %12 = arith.addi %out, %11 : i32
         linalg.yield %12, %9, %6 : i32, i32, i32
       }
       linalg.generic {indexing_maps = [#map22, #map22, #map22, #map22], iterator_types = ["parallel", "parallel"]} ins(%alloca_1, %alloca_0, %alloca : memref<512x512xi32>, memref<512x512xi32>, memref<512x512xi32>) outs(%score : memref<512x512xi32>) {
       ^bb0(%in: i32, %in_5: i32, %in_6: i32, %out: i32):
         %4 = arith.muli %in, %in_5 : i32
         %5 = arith.muli %in_6, %in_6 : i32
         %6 = arith.subi %4, %5 : i32
         %7 = arith.addi %in, %in_5 : i32
         %8 = arith.muli %7, %c4_i32 : i32
         %9 = arith.muli %8, %7 : i32
         %10 = arith.subi %6, %9 : i32
         linalg.yield %10 : i32
       }
       return %c0_i32 : i32
     }
   }
   
     module @for_loop_within_for_loop{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        //%buffer = memref.alloca() : memref<128xf32>
        scf.for %i = %c0 to %c10 step %c1 {
          scf.for %j = %c0 to %c10 step %c1 {
            linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
              iterator_types = ["parallel"]
            } ins(%buffer : memref<128xf32>) 
              outs(%buffer : memref<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %sum = arith.addf %in, %value : f32
              linalg.yield %sum : f32
            }
          }
          linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
          } ins(%buffer : memref<128xf32>) 
            outs(%buffer : memref<128xf32>) {
          ^bb0(%in: f32, %out: f32):
            %sum = arith.addf %in, %value : f32
            linalg.yield %sum : f32
          }
        }
        return
      }
  }
   
     module @for_loop_with_if_with_for{
      func.func @in_place_add(%buffer:  memref<128xf32> {llvm.noalias}, %value: f32, %cond: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        //%buffer = memref.alloca() : memref<128xf32>
        scf.for %i = %c0 to %c10 step %c1 {
          scf.if %cond {
            linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
              iterator_types = ["parallel"]
            } ins(%buffer : memref<128xf32>) 
              outs(%buffer : memref<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %sum = arith.addf %in, %value : f32
              linalg.yield %sum : f32
            }
            scf.for %j = %c0 to %c10 step %c1 {
              linalg.generic {
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]
              } ins(%buffer : memref<128xf32>) 
                outs(%buffer : memref<128xf32>) {
              ^bb0(%in: f32, %out: f32):
                %sum = arith.addf %in, %value : f32
                linalg.yield %sum : f32
              }
            }
            linalg.generic {
              indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
              iterator_types = ["parallel"]
            } ins(%buffer : memref<128xf32>) 
              outs(%buffer : memref<128xf32>) {
            ^bb0(%in: f32, %out: f32):
              %sum = arith.addf %in, %value : f32
              linalg.yield %sum : f32
            }
          }
        }
        return
      }
  }
