// RUN: polymer-opt %s -pluto-opt | FileCheck %s
func.func private @S0() attributes {scop.stmt}
func.func private @S1() attributes {scop.stmt}

func.func @foo(%N: index, %M: index, %L: index) {
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %L {
      func.call @S0() : () -> ()
    }
    affine.for %j = 0 to %M {
      affine.for %k = 0 to %L {
        func.call @S1() : () -> ()
      }
    }
  }
  return
}


// Just need to check this thing can be transformed.
// CHECK: func.func @foo
