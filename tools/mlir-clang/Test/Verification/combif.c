// RUN: mlir-clang %s --function=* -S | FileCheck %s

// TODO handle negation on if combine
// TODO remove unused cyclic phi
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
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-DAG:     %0 = llvm.mlir.undef : f32
// CHECK-DAG:     %1:2 = scf.while (%arg4 = %0, %arg5 = %c0_i32, %arg6 = %true) : (f32, i32, i1) -> (f32, i32) {
// CHECK-NEXT:       %2 = arith.cmpi slt, %arg5, %c1_i32 : i32
// CHECK-NEXT:       %3 = arith.andi %2, %arg6 : i1
// CHECK-NEXT:       scf.condition(%3) %arg4, %arg5 : f32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg4: f32, %arg5: i32):
// CHECK-NEXT:       %2 = arith.cmpf ugt, %arg3, %cst : f32
// CHECK-NEXT:       %3:3 = scf.if %2 -> (f32, i1, i32) {
// CHECK-NEXT:         scf.yield %arg4, %false, %arg5 : f32, i1, i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %4 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:         %5 = arith.cmpf ugt, %arg3, %arg2 : f32
// CHECK-NEXT:         %6 = arith.xori %5, %true : i1
// CHECK-NEXT:         %7 = scf.if %5 -> (i32) {
// CHECK-NEXT:           scf.yield %arg5 : i32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           call @use(%4) : (f32) -> ()
// CHECK-NEXT:           %8 = arith.addi %arg5, %c1_i32 : i32
// CHECK-NEXT:           scf.yield %8 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %4, %6, %7 : f32, i1, i32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %3#0, %3#2, %3#1 : f32, i32, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
