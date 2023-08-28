module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z3fooR7__half2(%arg0: memref<?x!llvm.struct<(struct<(i16)>, struct<(i16)>)>>, %arg1: memref<?x2xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<1x2xi16>
    %cast = memref.cast %alloca : memref<1x2xi16> to memref<?x2xi16>
    call @_ZNK7__half2cv11__half2_rawEv(%arg0, %cast) : (memref<?x!llvm.struct<(struct<(i16)>, struct<(i16)>)>>, memref<?x2xi16>) -> ()
    %0 = affine.load %alloca[0, 0] : memref<1x2xi16>
    %1 = affine.load %alloca[0, 1] : memref<1x2xi16>
    affine.store %0, %arg1[0, 0] : memref<?x2xi16>
    affine.store %1, %arg1[0, 1] : memref<?x2xi16>
    return
  }
  func.func @_ZNK7__half2cv11__half2_rawEv(%arg0: memref<?x!llvm.struct<(struct<(i16)>, struct<(i16)>)>>, %arg1: memref<?x2xi16>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0_i16 = arith.constant 0 : i16
    %alloca = memref.alloca() : memref<1x2xi16>
    %alloca_0 = memref.alloca() : memref<1x2xi16>
    affine.store %c0_i16, %alloca_0[0, 0] : memref<1x2xi16>
    affine.store %c0_i16, %alloca_0[0, 1] : memref<1x2xi16>
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i16)>, struct<(i16)>)>>) -> !llvm.ptr<i32>
    %1 = llvm.load %0 : !llvm.ptr<i32>
    %2 = "polygeist.memref2pointer"(%alloca_0) : (memref<1x2xi16>) -> !llvm.ptr<i32>
    llvm.store %1, %2 : !llvm.ptr<i32>
    %3 = affine.load %alloca_0[0, 0] : memref<1x2xi16>
    affine.store %3, %alloca[0, 0] : memref<1x2xi16>
    %4 = affine.load %alloca_0[0, 1] : memref<1x2xi16>
    affine.store %4, %alloca[0, 1] : memref<1x2xi16>
    %5 = affine.load %alloca[0, 0] : memref<1x2xi16>
    affine.store %5, %arg1[0, 0] : memref<?x2xi16>
    %6 = affine.load %alloca[0, 1] : memref<1x2xi16>
    affine.store %6, %arg1[0, 1] : memref<?x2xi16>
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}
