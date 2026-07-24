#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_upsample_nearest2d(%arg0: memref<?x4x8x8xf32>, %arg1: memref<?x4x16x16xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x4x16x16xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%0 : tensor<?x4x16x16xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = linalg.index 2 : index
      %6 = arith.cmpi slt, %5, %c0 : index
      %7 = arith.subi %c-1, %5 : index
      %8 = arith.select %6, %7, %5 : index
      %9 = arith.divsi %8, %c2 : index
      %10 = arith.subi %c-1, %9 : index
      %11 = arith.select %6, %10, %9 : index
      %12 = linalg.index 3 : index
      %13 = arith.cmpi slt, %12, %c0 : index
      %14 = arith.subi %c-1, %12 : index
      %15 = arith.select %13, %14, %12 : index
      %16 = arith.divsi %15, %c2 : index
      %17 = arith.subi %c-1, %16 : index
      %18 = arith.select %13, %17, %16 : index
      %19 = memref.load %arg0[%3, %4, %11, %18] : memref<?x4x8x8xf32>
      linalg.yield %19 : f32
    } -> tensor<?x4x16x16xf32>
    %2 = bufferization.to_memref %1 : memref<?x4x16x16xf32>
    memref.copy %2, %arg1 : memref<?x4x16x16xf32> to memref<?x4x16x16xf32>
    return
  }
}

