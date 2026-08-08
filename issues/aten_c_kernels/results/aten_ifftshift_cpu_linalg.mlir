#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_ifftshift_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c255 = arith.constant 255 : index
    %c0 = arith.constant 0 : index
    %c-129 = arith.constant -129 : index
    %c-255 = arith.constant -255 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%arg1 : memref<?xf32>) {
    ^bb0(%out: f32):
      %0 = linalg.index 0 : index
      %1 = arith.addi %0, %c128 : index
      %2 = arith.cmpi slt, %1, %c0 : index
      %3 = arith.subi %c-129, %0 : index
      %4 = arith.select %2, %3, %1 : index
      %5 = arith.divsi %4, %c255 : index
      %6 = arith.subi %c-1, %5 : index
      %7 = arith.select %2, %6, %5 : index
      %8 = arith.muli %7, %c-255 : index
      %9 = arith.addi %0, %8 : index
      %10 = arith.addi %9, %c128 : index
      %11 = memref.load %arg0[%10] : memref<?xf32>
      linalg.yield %11 : f32
    }
    return
  }
}

