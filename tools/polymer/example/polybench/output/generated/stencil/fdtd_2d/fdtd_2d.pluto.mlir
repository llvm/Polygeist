#map0 = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<() -> (1)>
#map3 = affine_map<()[s0] -> (s0 - 1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1] -> (s0, s1)>
#map6 = affine_map<()[s0, s1] -> (s0 - 1, s1)>
#map7 = affine_map<()[s0, s1] -> (s0, s1 - 1)>
#map8 = affine_map<()[s0, s1] -> (s0, s1 + 1)>
#map9 = affine_map<()[s0, s1] -> (s0 + 1, s1)>
#map10 = affine_map<()[s0] -> (32, s0)>
#map11 = affine_map<(d0) -> (d0 - 1)>
#map12 = affine_map<()[s0, s1] -> (3, s0, s1)>
#map13 = affine_map<() -> (4)>
#map14 = affine_map<()[s0, s1] -> (32, s0, s1)>
#map15 = affine_map<()[s0] -> (4, s0)>
#map16 = affine_map<(d0) -> (d0 * 32)>
#map17 = affine_map<(d0)[s0] -> (s0, d0 * 32 + 32)>
#map18 = affine_map<()[s0] -> ((s0 - 1) floordiv 32 + 1)>
#map19 = affine_map<() -> (32)>
#map20 = affine_map<(d0)[s0, s1] -> (s0, s1, d0 * 32 + 32)>
#map21 = affine_map<()[s0, s1] -> ((s0 - 1) floordiv 32 + 1, (s1 - 1) floordiv 32 + 1)>
#map22 = affine_map<()[s0] -> (s0 ceildiv 32)>
#map23 = affine_map<(d0) -> (1, d0 * 32)>

#set0 = affine_set<()[s0, s1] : (s0 - 3 >= 0, s1 - 2 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 - 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 4 >= 0, s1 - 4 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 4 >= 0, -s1 + 3 >= 0)>
#set4 = affine_set<()[s0, s1] : (-s0 + 3 >= 0, s1 - 4 >= 0)>
#set5 = affine_set<()[s0, s1] : (-s0 + 3 >= 0, -s1 + 3 >= 0)>
#set6 = affine_set<(d0) : (d0 == 0)>

module {
  func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %3 = alloca() : memref<1xf64>
      call @S0(%3, %arg6, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
      affine.for %arg8 = 0 to %1 {
        call @S1(%arg4, %arg8, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      }
      affine.for %arg8 = 1 to %2 {
        affine.for %arg9 = 0 to %1 {
          call @S2(%arg4, %arg8, %arg9, %arg5) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = 0 to %2 {
        affine.for %arg9 = 1 to %1 {
          call @S3(%arg3, %arg8, %arg9, %arg5) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = 0 to #map3()[%2] {
        affine.for %arg9 = 0 to #map3()[%1] {
          call @S4(%arg5, %arg8, %arg9, %arg4, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
        }
      }
    }
    return
  }
  func @S0(%arg0: memref<1xf64>, %arg1: memref<500xf64>, %arg2: index) attributes {scop.stmt} {
    %0 = affine.load %arg1[symbol(%arg2)] : memref<500xf64>
    affine.store %0, %arg0[0] : memref<1xf64>
    return
  }
  func @S1(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: memref<1xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<1xf64>
    affine.store %0, %arg0[0, symbol(%arg1)] : memref<1000x1200xf64>
    return
  }
  func @S2(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>) attributes {scop.stmt} {
    %cst = constant 5.000000e-01 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %2 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<1000x1200xf64>
    %3 = subf %1, %2 : f64
    %4 = mulf %cst, %3 : f64
    %5 = subf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    return
  }
  func @S3(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>) attributes {scop.stmt} {
    %cst = constant 5.000000e-01 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %2 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<1000x1200xf64>
    %3 = subf %1, %2 : f64
    %4 = mulf %cst, %3 : f64
    %5 = subf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    return
  }
  func @S4(%arg0: memref<1000x1200xf64>, %arg1: index, %arg2: index, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>) attributes {scop.stmt} {
    %cst = constant 0.69999999999999996 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1] : memref<1000x1200xf64>
    %2 = affine.load %arg4[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %3 = subf %1, %2 : f64
    %4 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<1000x1200xf64>
    %5 = addf %3, %4 : f64
    %6 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    %7 = subf %5, %6 : f64
    %8 = mulf %cst, %7 : f64
    %9 = subf %0, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<1000x1200xf64>
    return
  }
  func @kernel_fdtd_2d_new(%arg0: i32, %arg1: memref<500xf64>, %arg2: memref<1000x1200xf64>, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: i32, %arg6: i32) {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = index_cast %arg6 : i32 to index
    %1 = index_cast %arg5 : i32 to index
    %2 = index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      %3 = alloca() : memref<1xf64>
      call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      affine.for %arg8 = 1 to min #map10()[%1] {
        call @S2(%arg2, %arg7, %arg8, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
      }
      affine.for %arg8 = 1 to min #map12()[%1, %0] {
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        affine.for %arg9 = 1 to min #map10()[%1] {
          %4 = affine.apply #map11(%arg9)
          call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
          call @S3(%arg4, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S2(%arg2, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.if #set0()[%1, %0] {
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      }
      affine.if #set1()[%1, %0] {
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        call @S4(%arg3, %arg7, %c0, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
        call @S3(%arg4, %arg7, %c1, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        call @S2(%arg2, %arg7, %c1, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
      }
      affine.if #set2()[%1, %0] {
        call @S0(%3, %arg1, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        affine.for %arg8 = 1 to min #map10()[%1] {
          %4 = affine.apply #map11(%arg8)
          call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
          call @S3(%arg4, %arg7, %arg8, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S2(%arg2, %arg7, %arg8, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.if #set3()[%1, %0] {
        call @S0(%3, %arg1, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      }
      affine.if #set4()[%1, %0] {
        call @S0(%3, %arg1, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        affine.for %arg8 = 1 to %1 {
          %4 = affine.apply #map11(%arg8)
          call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
          call @S3(%arg4, %arg7, %arg8, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S2(%arg2, %arg7, %arg8, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.if #set5()[%1, %0] {
        call @S0(%3, %arg1, %arg7) : (memref<1xf64>, memref<500xf64>, index) -> ()
      }
      affine.for %arg8 = 4 to min #map14()[%1, %0] {
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        affine.for %arg9 = 1 to min #map10()[%1] {
          %4 = affine.apply #map11(%arg9)
          call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
          call @S3(%arg4, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S2(%arg2, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = max #map15()[%0] to min #map10()[%1] {
        call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
      }
      affine.for %arg8 = max #map15()[%1] to min #map10()[%0] {
        call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        affine.for %arg9 = 1 to %1 {
          %4 = affine.apply #map11(%arg9)
          call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
          call @S3(%arg4, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S2(%arg2, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
      }
      affine.for %arg8 = 1 to #map18()[%1] {
        affine.for %arg9 = #map16(%arg8) to min #map17(%arg8)[%1] {
          call @S2(%arg2, %arg7, %arg9, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
        }
        affine.for %arg9 = 1 to min #map10()[%0] {
          affine.for %arg10 = #map16(%arg8) to min #map17(%arg8)[%1] {
            %4 = affine.apply #map11(%arg10)
            call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
            call @S3(%arg4, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            call @S2(%arg2, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          }
        }
      }
      affine.for %arg8 = 1 to min #map21()[%1, %0] {
        affine.for %arg9 = #map16(%arg8) to min #map20(%arg8)[%1, %0] {
          call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
          affine.for %arg10 = 1 to 32 {
            %4 = affine.apply #map11(%arg10)
            call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
            call @S3(%arg4, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            call @S2(%arg2, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          }
        }
        affine.for %arg9 = %0 to min #map17(%arg8)[%1] {
          call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        }
        affine.for %arg9 = %1 to min #map17(%arg8)[%0] {
          call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          affine.for %arg10 = 1 to 32 {
            %4 = affine.apply #map11(%arg10)
            call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
            call @S3(%arg4, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            call @S2(%arg2, %arg7, %arg10, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
          }
        }
        affine.for %arg9 = 1 to #map18()[%1] {
          affine.for %arg10 = #map16(%arg8) to min #map17(%arg8)[%0] {
            affine.for %arg11 = #map16(%arg9) to min #map17(%arg9)[%1] {
              %4 = affine.apply #map11(%arg11)
              call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
              call @S3(%arg4, %arg7, %arg11, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
              call @S2(%arg2, %arg7, %arg11, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            }
          }
        }
      }
      affine.for %arg8 = #map22()[%0] to #map18()[%1] {
        affine.for %arg9 = #map16(%arg8) to min #map17(%arg8)[%1] {
          call @S1(%arg2, %arg7, %3) : (memref<1000x1200xf64>, index, memref<1xf64>) -> ()
        }
      }
      affine.for %arg8 = #map22()[%1] to #map18()[%0] {
        affine.for %arg9 = 0 to #map18()[%1] {
          affine.for %arg10 = #map16(%arg8) to min #map17(%arg8)[%0] {
            affine.if #set6(%arg9) {
              call @S3(%arg4, %arg7, %c0, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            }
            affine.for %arg11 = max #map23(%arg9) to min #map17(%arg9)[%1] {
              %4 = affine.apply #map11(%arg11)
              call @S4(%arg3, %arg7, %4, %arg2, %arg4) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>, memref<1000x1200xf64>) -> ()
              call @S3(%arg4, %arg7, %arg11, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
              call @S2(%arg2, %arg7, %arg11, %arg3) : (memref<1000x1200xf64>, index, index, memref<1000x1200xf64>) -> ()
            }
          }
        }
      }
    }
    return
  }
}
