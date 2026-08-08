#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_xlog1py(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg2 : memref<?xf32>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = memref.load %arg1[%0] : memref<?xf32>
      %2 = arith.cmpf une, %1, %1 : f32
      %3 = scf.if %2 -> (f32) {
        %4 = memref.load %arg1[%0] : memref<?xf32>
        scf.yield %4 : f32
      } else {
        %4 = memref.load %arg0[%0] : memref<?xf32>
        %5 = arith.cmpf oeq, %4, %cst : f32
        %6 = scf.if %5 -> (f32) {
          scf.yield %cst : f32
        } else {
          %7 = memref.load %arg0[%0] : memref<?xf32>
          %8 = memref.load %arg1[%0] : memref<?xf32>
          %9 = math.log1p %8 : f32
          %10 = arith.mulf %7, %9 : f32
          scf.yield %10 : f32
        }
        scf.yield %6 : f32
      }
      linalg.yield %3 : f32
    }
    return
  }
  func.func private @log1pf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>, polygeist.pure}
}

