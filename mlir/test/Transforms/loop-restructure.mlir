// RUN: mlir-opt %s -pass-pipeline="func(loop-restructure)" -split-input-file | FileCheck %s

func @simple() {
  %ci0 = constant 0 : index
  %ci1 = constant 1 : index
  %ci10 = constant 10 : index
  br ^bb0(%ci0 : index)

  ^bb0(%i: index):
    %i1 = addi %i, %ci1: index
    %cmp = cmpi "eq", %i1, %ci10 : index
    cond_br %cmp, ^exit, ^bb1

  ^bb1:
    br ^bb0(%i1 : index)

  ^exit:
     return

  // CHECK loop.natural (%i)
  // CHECK loop.natural.next {{.*}}: index
  // CHECK loop.natural.return

}
