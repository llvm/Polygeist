// RUN: polygeist-opt --lower-affine --canonicalize --convert-parallel-to-gpu1 --canonicalize %s | FileCheck %s

module {
  func.func private @use(%arg0: index)
  func.func private @wow()
  func.func @foo(%i : i1, %129 : index) -> () {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    "polygeist.gpu_wrapper"(%c512, %c1, %c1) ({
      scf.if %i {
        %c0_111 = arith.constant 0 : index
        %c1_112 = arith.constant 1 : index
        scf.parallel (%arg7) = (%c0_111) to (%129) step (%c1_112) {
          func.call @use(%arg7) : (index) -> ()
          scf.yield
        }
      }
      "polygeist.polygeist_yield"() : () -> ()
    }) : (index, index, index) -> ()
    return
  }
}

// TODO we need versions that need gpu cache to split wrapper (from particlefilter), lud or sradv1 had an alloca in wrapper case
