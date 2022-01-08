// RUN: mlir-clang %s --function=* -S | FileCheck %s

void use(float);

int solver(	float** y,
					int xmax,
                    float tolerance,
					float err) {
		for (int j = 0; j < 1; j++) {
			if(err > 0){
				break;
			}
    		float scale_fina = (float)xmax;

			if ( err > tolerance ) {
				    break;
			}
			use(scale_fina);
		}
	return 0;
} 

// CHECK:   func @solver(%arg0: memref<?xmemref<?xf32>>, %arg1: i32, %arg2: f32, %arg3: f32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = scf.while (%arg4 = %true, %arg5 = %c0_i32) : (i1, i32) -> i32 {
// CHECK-NEXT:       %1 = arith.cmpi slt, %arg5, %c1_i32 : i32
// CHECK-NEXT:       %2 = arith.andi %1, %arg4 : i1
// CHECK-NEXT:       scf.condition(%2) %arg5 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg4: i32):  // no predecessors
// CHECK-NEXT:       %1 = arith.cmpf ugt, %arg3, %cst : f32
// CHECK-NEXT:       %2 = arith.xori %1, %true : i1
// CHECK-NEXT:       %3 = scf.if %2 -> (i1) {
// CHECK-NEXT:         %5 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:         %6 = arith.cmpf ugt, %arg3, %arg2 : f32
// CHECK-NEXT:         %7 = arith.xori %6, %true : i1
// CHECK-NEXT:         %8 = arith.xori %6, %true : i1
// CHECK-NEXT:         scf.if %7 {
// CHECK-NEXT:           call @use(%5) : (f32) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %8 : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %false : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       %4 = arith.addi %arg4, %c1_i32 : i32
// CHECK-NEXT:       scf.yield %3, %4 : i1, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
