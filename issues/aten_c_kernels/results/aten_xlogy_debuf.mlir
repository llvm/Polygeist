#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_xlogy(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = memref.load %arg1[%3] : memref<?xf32>
      %5 = arith.cmpf une, %4, %4 : f32
      %6 = scf.if %5 -> (f32) {
        %7 = memref.load %arg1[%3] : memref<?xf32>
        scf.yield %7 : f32
      } else {
        %7 = memref.load %arg0[%3] : memref<?xf32>
        %8 = arith.cmpf oeq, %7, %cst : f32
        %9 = scf.if %8 -> (f32) {
          scf.yield %cst : f32
        } else {
          %10 = memref.load %arg0[%3] : memref<?xf32>
          %11 = memref.load %arg1[%3] : memref<?xf32>
          %12 = math.log %11 : f32
          %13 = arith.mulf %10, %12 : f32
          scf.yield %13 : f32
        }
        scf.yield %9 : f32
      }
      linalg.yield %6 : f32
    } -> tensor<?xf32>
    %2 = bufferization.to_memref %1 : memref<?xf32>
    memref.copy %2, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @logf(f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>, polygeist.pure}
}

