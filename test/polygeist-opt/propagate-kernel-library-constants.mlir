// RUN: polygeist-opt --split-input-file --propagate-kernel-library-constants %s | FileCheck %s

// Uniform values are general typed facts, not a special representation of
// zero. The producer and consumer names are deliberately arbitrary.
module {
  kernel.defn @fill_two(%out: tensor<?xf64>) -> tensor<?xf64> {
    %two = arith.constant 2.000000e+00 : f64
    %filled = linalg.fill ins(%two : f64) outs(%out : tensor<?xf64>)
        -> tensor<?xf64>
    kernel.yield %filled : tensor<?xf64>
  }
  kernel.defn @consume(%out: tensor<?xf64>) -> tensor<?xf64> {
    kernel.yield %out : tensor<?xf64>
  }

  func.func @nonzero_through_slice(%out: tensor<?xf64>, %size: index)
      -> tensor<?xf64> {
    %filled = kernel.launch @fill_two(%out)
        : (tensor<?xf64>) -> tensor<?xf64>
    %slice = tensor.extract_slice %filled[0] [%size] [1]
        : tensor<?xf64> to tensor<?xf64>
    %result = kernel.launch @consume(%slice)
        : (tensor<?xf64>) -> tensor<?xf64>
    return %result : tensor<?xf64>
  }
}

// CHECK-LABEL: func.func @nonzero_through_slice
// CHECK: kernel.launch @consume
// CHECK-SAME: polygeist.uniform_operand_constants = {operand_0 = 2.000000e+00 : f64}

// -----

// Equal facts from both branches merge at an scf.if result.
module {
  kernel.defn @fill_zero(%out: tensor<?xf32>) -> tensor<?xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<?xf32>)
        -> tensor<?xf32>
    kernel.yield %filled : tensor<?xf32>
  }
  kernel.defn @consume(%out: tensor<?xf32>) -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }

  func.func @agreeing_branches(%condition: i1, %a: tensor<?xf32>,
                               %b: tensor<?xf32>) -> tensor<?xf32> {
    %selected = scf.if %condition -> tensor<?xf32> {
      %x = kernel.launch @fill_zero(%a) : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %x : tensor<?xf32>
    } else {
      %y = kernel.launch @fill_zero(%b) : (tensor<?xf32>) -> tensor<?xf32>
      scf.yield %y : tensor<?xf32>
    }
    %result = kernel.launch @consume(%selected)
        : (tensor<?xf32>) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @agreeing_branches
// CHECK: kernel.launch @consume
// CHECK-SAME: polygeist.uniform_operand_constants = {operand_0 = 0.000000e+00 : f32}

// -----

// A diagonal-only constant write does not establish a uniform whole-tensor
// fact because its output indexing map is not the identity.
#diag = affine_map<(d0) -> (d0, d0)>
module {
  kernel.defn @fill_diagonal(%out: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %one = arith.constant 1.000000e+00 : f64
    %result = linalg.generic {
      indexing_maps = [#diag], iterator_types = ["parallel"]
    } outs(%out : tensor<?x?xf64>) {
    ^bb0(%old: f64):
      linalg.yield %one : f64
    } -> tensor<?x?xf64>
    kernel.yield %result : tensor<?x?xf64>
  }
  kernel.defn @consume(%out: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %out : tensor<?x?xf64>
  }

  func.func @partial_write_is_unknown(%out: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    %diagonal = kernel.launch @fill_diagonal(%out)
        : (tensor<?x?xf64>) -> tensor<?x?xf64>
    %result = kernel.launch @consume(%diagonal)
        : (tensor<?x?xf64>) -> tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @partial_write_is_unknown
// CHECK: kernel.launch @consume(%{{.*}}) :
// CHECK-NOT: uniform_operand_constants

// -----

// An unchanged loop-carried tensor preserves its incoming uniform fact.
module {
  kernel.defn @fill_minus_one(%out: tensor<8xi32>) -> tensor<8xi32> {
    %minus_one = arith.constant -1 : i32
    %filled = linalg.fill ins(%minus_one : i32) outs(%out : tensor<8xi32>)
        -> tensor<8xi32>
    kernel.yield %filled : tensor<8xi32>
  }
  kernel.defn @consume(%out: tensor<8xi32>) -> tensor<8xi32> {
    kernel.yield %out : tensor<8xi32>
  }

  func.func @loop_carried_constant(%out: tensor<8xi32>, %count: index)
      -> tensor<8xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %filled = kernel.launch @fill_minus_one(%out)
        : (tensor<8xi32>) -> tensor<8xi32>
    %carried = scf.for %i = %c0 to %count step %c1
        iter_args(%arg = %filled) -> tensor<8xi32> {
      scf.yield %arg : tensor<8xi32>
    }
    %result = kernel.launch @consume(%carried)
        : (tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}

// CHECK-LABEL: func.func @loop_carried_constant
// CHECK: kernel.launch @consume
// CHECK-SAME: polygeist.uniform_operand_constants = {operand_0 = -1 : i32}
