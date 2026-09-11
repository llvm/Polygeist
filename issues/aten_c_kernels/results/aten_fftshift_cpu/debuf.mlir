#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_fftshift_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %c-256 = arith.constant -256 : index
    %c-129 = arith.constant -129 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = arith.addi %3, %c128 : index
      %5 = arith.cmpi slt, %4, %c0 : index
      %6 = arith.subi %c-129, %3 : index
      %7 = arith.select %5, %6, %4 : index
      %8 = arith.divsi %7, %c256 : index
      %9 = arith.subi %c-1, %8 : index
      %10 = arith.select %5, %9, %8 : index
      %11 = arith.muli %10, %c-256 : index
      %12 = arith.addi %3, %11 : index
      %13 = arith.addi %12, %c128 : index
      %14 = memref.load %arg0[%13] : memref<?xf32>
      linalg.yield %14 : f32
    } -> tensor<?xf32>
    %2 = bufferization.to_memref %1 : memref<?xf32>
    memref.copy %2, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

