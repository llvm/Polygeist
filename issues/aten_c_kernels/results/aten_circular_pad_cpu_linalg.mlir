#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_circular_pad_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c-3 = arith.constant -3 : index
    %c-1 = arith.constant -1 : index
    %c-3_i32 = arith.constant -3 : i32
    %c32_i32 = arith.constant 32 : i32
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.addi %1, %c-3_i32 : i32
      %3 = arith.remsi %2, %c32_i32 : i32
      %4 = arith.addi %0, %c-3 : index
      %5 = arith.cmpi slt, %4, %c0 : index
      %6 = arith.subi %c2, %0 : index
      %7 = arith.select %5, %6, %4 : index
      %8 = arith.divsi %7, %c32 : index
      %9 = arith.subi %c-1, %8 : index
      %10 = arith.select %5, %9, %8 : index
      %11 = arith.muli %10, %c32 : index
      %12 = arith.subi %11, %0 : index
      %13 = arith.addi %12, %c2 : index
      %14 = arith.cmpi sge, %13, %c0 : index
      %15 = arith.addi %3, %c32_i32 : i32
      %16 = arith.select %14, %15, %3 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %arg0[%17] : memref<?xf32>
      linalg.yield %18 : f32
    }
    return
  }
}

