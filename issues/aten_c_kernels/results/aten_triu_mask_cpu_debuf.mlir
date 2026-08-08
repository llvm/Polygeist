#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_triu_mask_cpu(%arg0: memref<?x32xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x32xi32>
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c32, %c32] [1, 1] : tensor<?x32xi32> to tensor<?x?xi32>
    %1 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %3 = linalg.index 0 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.subi %6, %4 : i32
      %8 = arith.cmpi sge, %7, %arg1 : i32
      %9 = arith.extui %8 : i1 to i32
      linalg.yield %9 : i32
    } -> tensor<?x?xi32>
    %inserted_slice = tensor.insert_slice %1 into %0[0, 0] [%c32, %c32] [1, 1] : tensor<?x?xi32> into tensor<?x32xi32>
    %2 = bufferization.to_memref %inserted_slice : memref<?x32xi32>
    memref.copy %2, %arg0 : memref<?x32xi32> to memref<?x32xi32>
    return
  }
}

