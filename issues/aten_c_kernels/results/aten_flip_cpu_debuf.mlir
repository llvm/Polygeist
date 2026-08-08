#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_flip_cpu(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %1 = arith.cmpi ne, %arg2, %c0_i32 : i32
    %2 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %extracted_slice = tensor.extract_slice %0[0, 0] [%c32, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.subi %c31_i32, %6 : i32
      %8 = arith.select %1, %7, %6 : i32
      %9 = arith.index_cast %8 : i32 to index
      %10 = linalg.index 1 : index
      %11 = arith.index_cast %10 : index to i32
      %12 = arith.subi %c63_i32, %11 : i32
      %13 = arith.select %2, %12, %11 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = memref.load %arg0[%9, %14] : memref<?x64xf32>
      linalg.yield %15 : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %0[0, 0] [%c32, %c64] [1, 1] : tensor<?x?xf32> into tensor<?x64xf32>
    %4 = bufferization.to_memref %inserted_slice : memref<?x64xf32>
    memref.copy %4, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

