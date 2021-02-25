// RUN: mlir-clang %s %stdinclude --function=init_array | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>

/* Include polybench common header. */
#include <polybench.h>

void use(double A[20]);
/* Array initialization. */

void init_array (int n)
{
  double (*B)[20] = (double(*)[20])polybench_alloc_data (20, sizeof(double)) ;
  (*B)[2] = 3.0;
  use(*B);
}


// CHECK:  func @init_array(%arg0: i32)
// CHECK-NEXT:    %cst = constant 3.000000e+00 : f64
// CHECK-NEXT:    %c2 = constant 2 : index
// CHECK-NEXT:    %0 = alloc() : memref<20xf64>
// CHECK-NEXT:    store %cst, %0[%c2] : memref<20xf64>
// CHECK-NEXT:    %1 = memref_cast %0 : memref<20xf64> to memref<?xf64>
// CHECK-NEXT:    call @use(%1) : (memref<?xf64>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// EXEC: {{[0-9]\.[0-9]+}}
