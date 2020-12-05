  func @kernel_3mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<16x18xf64>, %arg6: memref<16x20xf64>, %arg7: memref<20x18xf64>, %arg8: memref<18x22xf64>, %arg9: memref<18x24xf64>, %arg10: memref<24x22xf64>, %arg11: memref<16x22xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    %2 = index_cast %arg3 : i32 to index
    %3 = index_cast %arg4 : i32 to index
    %4 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %0 {
        call @S0(%arg5, %arg12, %arg13) : (memref<16x18xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %1 {
          call @S1(%arg5, %arg12, %arg13, %arg7, %arg14, %arg6) : (memref<16x18xf64>, index, index, memref<20x18xf64>, index, memref<16x20xf64>) -> ()
        }
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %2 {
        call @S2(%arg8, %arg12, %arg13) : (memref<18x22xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %3 {
          call @S3(%arg8, %arg12, %arg13, %arg10, %arg14, %arg9) : (memref<18x22xf64>, index, index, memref<24x22xf64>, index, memref<18x24xf64>) -> ()
        }
      }
    }
    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %2 {
        call @S4(%arg11, %arg12, %arg13) : (memref<16x22xf64>, index, index) -> ()
        affine.for %arg14 = 0 to %0 {
          call @S5(%arg11, %arg12, %arg13, %arg8, %arg14, %arg5) : (memref<16x22xf64>, index, index, memref<18x22xf64>, index, memref<16x18xf64>) -> ()
        }
      }
    }
    return
  }
