// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

#set = affine_set<(d0) : (-d0 + 31 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0)>
#set2 = affine_set<(d0)[s0] : (-d0 + s0 -1 >= 0, s0-10 >= 0)>
#set3 = affine_set<(d0, d1)[s0] : (-d0 - d1 * 100 + s0 -1 >= 0)>
module {
  func.func @f1(%714: memref<512xf64>, %703: memref<512xf64>) {
        affine.parallel (%arg8) = (0) to (512) {
          %716 = affine.load %714[%arg8] : memref<512xf64>
          affine.if #set(%arg8) {
            %717 = affine.load %703[%arg8 + 1] : memref<512xf64>
            %718 = arith.addf %716, %717 : f64
            affine.store %718, %703[%arg8] : memref<512xf64>
          }
        }
    return
  }
  func.func @f2(%714: memref<512xf64>, %703: memref<512xf64>, %t: index) {
        affine.parallel (%arg8) = (0) to (512) {
          %716 = affine.load %714[%arg8] : memref<512xf64>
          affine.if #set1(%arg8)[%t] {
            %717 = affine.load %703[%arg8 + 1] : memref<512xf64>
            %718 = arith.addf %716, %717 : f64
            affine.store %718, %703[%arg8] : memref<512xf64>
          }
        }
    return
  }
  func.func @f3(%714: memref<512xf64>, %703: memref<512xf64>, %t: index) {
        affine.parallel (%arg8) = (0) to (512) {
          %716 = affine.load %714[%arg8] : memref<512xf64>
          affine.if #set2(%arg8)[%t] {
            %717 = affine.load %703[%arg8 + 1] : memref<512xf64>
            %718 = arith.addf %716, %717 : f64
            affine.store %718, %703[%arg8] : memref<512xf64>
          }
        }
    return
  }
  func.func @f4(%arg0: memref<512xf64>, %arg1: memref<512xf64>, %arg2: index, %arg3: index) {
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), 100) {
      %0 = affine.load %arg0[%arg4] : memref<512xf64>
      affine.if #set3(%arg5, %arg4)[%arg2] {
        %1 = affine.load %arg1[%arg5 + %arg4 * 100 + 1] : memref<512xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %arg1[%arg5 + %arg4 * 100] : memref<512xf64>
      }
    }
    return
  }
  func.func @f5(%arg0: memref<512xf64>, %arg1: memref<512xf64>, %arg2: index, %arg3: index) {
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), 100) {
      %0 = affine.load %arg0[0] : memref<512xf64>
      affine.if #set3(%arg5, %arg4)[%arg2] {
        %1 = affine.load %arg1[%arg5 + %arg4 * 100 + 1] : memref<512xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %arg1[%arg5 + %arg4 * 100] : memref<512xf64>
      }
    }
    return
  }
  func.func @f6(%arg0: memref<512xf64>, %arg1: memref<512xf64>, %arg2: index, %arg3: index) {
    affine.parallel (%arg4) = (10) to (11) {
      %0 = affine.load %arg0[%arg4] : memref<512xf64>
      %1 = affine.load %arg1[%arg4] : memref<512xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg1[%arg4] : memref<512xf64>
    }
    return
  }
}

// CHECK: #set = affine_set<()[s0] : (s0 - 10 >= 0)>
// CHECK: #set1 = affine_set<(d0, d1)[s0] : (-d0 - d1 * 100 + s0 - 1 >= 0)>

// CHECK:   func.func @f1(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>) {
// CHECK-NEXT:     affine.parallel (%[[arg2:.+]]) = (0) to (32) {
// CHECK-NEXT:       %[[V0:.+]] = affine.load %[[arg0]][%[[arg2]]] : memref<512xf64>
// CHECK-NEXT:       %[[V1:.+]] = affine.load %[[arg1]][%[[arg2]] + 1] : memref<512xf64>
// CHECK-NEXT:       %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:       affine.store %[[V2]], %[[arg1]][%[[arg2]]] : memref<512xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHEK-NEXT:   }

// CHECK:  func.func @f2(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>, %[[arg2:.+]]: index) {
// CHECK-NEXT:    affine.parallel (%[[arg3:.+]]) = (0) to (min(symbol(%[[arg2]]), 512)) {
// CHECK-NEXT:      %[[V0:.+]] = affine.load %[[arg0]][%[[arg3]]] : memref<512xf64>
// CHECK-NEXT:      %[[V1:.+]] = affine.load %[[arg1]][%[[arg3]] + 1] : memref<512xf64>
// CHECK-NEXT:      %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:      affine.store %[[V2]], %[[arg1]][%[[arg3]]] : memref<512xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:   func.func @f3(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>, %[[arg2:.+]]: index) {
// CHECK-NEXT:     affine.parallel (%[[arg3:.+]]) = (0) to (min(symbol(%[[arg2]]), 512)) {
// CHECK-NEXT:       %[[V0:.+]] = affine.load %[[arg0]][%[[arg3]]] : memref<512xf64>
// CHECK-NEXT:       affine.if #set()[%[[arg2]]] {
// CHECK-NEXT:         %[[V1:.+]] = affine.load %[[arg1]][%[[arg3]] + 1] : memref<512xf64>
// CHECK-NEXT:         %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:         affine.store %[[V2]], %[[arg1]][%[[arg3]]] : memref<512xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @f4(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
// CHECK-NEXT:     affine.parallel (%[[arg4:.+]], %[[arg5:.+]]) = (0, 0) to (symbol(%[[arg3]]), 100) {
// CHECK-NEXT:       %[[V0:.+]] = affine.load %[[arg0]][%[[arg4]]] : memref<512xf64>
// CHECK-NEXT:       affine.if #set1(%[[arg5]], %[[arg4]])[%[[arg2]]] {
// CHECK-NEXT:         %[[V1:.+]] = affine.load %[[arg1]][%[[arg5]] + %[[arg4]] * 100 + 1] : memref<512xf64>
// CHECK-NEXT:         %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:         affine.store %[[V2]], %[[arg1]][%[[arg5]] + %[[arg4]] * 100] : memref<512xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func.func @f5(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
// CHECK-NEXT:     affine.parallel (%[[arg4:.+]]) = (0) to (min(symbol(%[[arg2]]), symbol(%[[arg3]]) * 100)) {
// CHECK-NEXT:       %[[V0:.+]] = affine.load %[[arg0]][0] : memref<512xf64>
// CHECK-NEXT:       %[[V1:.+]] = affine.load %[[arg1]][%[[arg4]] + 1] : memref<512xf64>
// CHECK-NEXT:       %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:       affine.store %[[V2]], %[[arg1]][%[[arg4]]] : memref<512xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:  func.func @f6(%[[arg0:.+]]: memref<512xf64>, %[[arg1:.+]]: memref<512xf64>, %[[arg2:.+]]: index, %[[arg3:.+]]: index) {
// CHECK-NEXT:    %[[V0:.+]] = affine.load %[[arg0]][10] : memref<512xf64>
// CHECK-NEXT:    %[[V1:.+]] = affine.load %[[arg1]][10] : memref<512xf64>
// CHECK-NEXT:    %[[V2:.+]] = arith.addf %[[V0]], %[[V1]] : f64
// CHECK-NEXT:    affine.store %[[V2]], %[[arg1]][10] : memref<512xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
