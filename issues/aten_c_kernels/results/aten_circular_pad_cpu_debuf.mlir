#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_circular_pad_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32_i32 = arith.constant 32 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-1 = arith.constant -1 : index
    %c-3 = arith.constant -3 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xf32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%0 : tensor<?xf32>) {
    ^bb0(%out: f32):
      %3 = linalg.index 0 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.addi %4, %c-3_i32 : i32
      %6 = arith.remsi %5, %c32_i32 : i32
      %7 = arith.addi %3, %c-3 : index
      %8 = arith.cmpi slt, %7, %c0 : index
      %9 = arith.subi %c2, %3 : index
      %10 = arith.select %8, %9, %7 : index
      %11 = arith.divsi %10, %c32 : index
      %12 = arith.subi %c-1, %11 : index
      %13 = arith.select %8, %12, %11 : index
      %14 = arith.muli %13, %c32 : index
      %15 = arith.subi %14, %3 : index
      %16 = arith.addi %15, %c2 : index
      %17 = arith.cmpi sge, %16, %c0 : index
      %18 = arith.addi %6, %c32_i32 : i32
      %19 = arith.select %17, %18, %6 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = memref.load %arg0[%20] : memref<?xf32>
      linalg.yield %21 : f32
    } -> tensor<?xf32>
    %2 = bufferization.to_memref %1 : memref<?xf32>
    memref.copy %2, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

