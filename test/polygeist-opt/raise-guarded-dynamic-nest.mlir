// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

#minus_one = affine_map<()[s0] -> (s0 - 1)>

module {
  memref.global @bounds : memref<3xi32>
  memref.global @array : memref<16x16x16x5xf64>

  // Model the mixed scf/affine nest produced by cgeist for NPB BT add.c.
  // The non-empty guards are redundant, and the innermost bound load is
  // invariant because the loop writes only A.
  func.func @guarded_dynamic_nest() {
    %one = arith.constant 1 : index
    %zero = arith.constant 0.0 : f64
    %bounds = memref.get_global @bounds : memref<3xi32>
    %A = memref.get_global @array : memref<16x16x16x5xf64>
    %z_i32 = affine.load %bounds[2] : memref<3xi32>
    %z = arith.index_cast %z_i32 : i32 to index
    %z_last = affine.apply #minus_one()[%z]
    %z_nonempty = arith.cmpi sgt, %z_last, %one : index
    scf.if %z_nonempty {
      %y_i32 = affine.load %bounds[1] : memref<3xi32>
      %y = arith.index_cast %y_i32 : i32 to index
      affine.for %k = 1 to #minus_one()[%z] {
        %x_i32 = affine.load %bounds[0] : memref<3xi32>
        %x = arith.index_cast %x_i32 : i32 to index
        %y_last = affine.apply #minus_one()[%y]
        %y_nonempty = arith.cmpi sgt, %y_last, %one : index
        scf.if %y_nonempty {
          scf.for %j = %one to %y_last step %one {
            %x_last = affine.apply #minus_one()[%x]
            scf.for %i = %one to %x_last step %one {
              affine.for %m = 0 to 5 {
                memref.store %zero, %A[%k, %j, %i, %m]
                    : memref<16x16x16x5xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @guarded_dynamic_nest
// CHECK-NOT: scf.if
// CHECK-NOT: scf.for
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK: linalg.yield
