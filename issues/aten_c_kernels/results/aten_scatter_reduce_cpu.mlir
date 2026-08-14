#set = affine_set<()[s0] : (s0 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 == 0)>
#set2 = affine_set<()[s0] : (s0 - 2 == 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_scatter_reduce_cpu(%arg0: memref<?x128xf32>, %arg1: memref<?x64xi32>, %arg2: memref<?x64xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg3 : i32 to index
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 64 {
        %1 = affine.load %arg1[%arg4, %arg5] : memref<?x64xi32>
        %2 = affine.load %arg2[%arg4, %arg5] : memref<?x64xf32>
        affine.if #set()[%0] {
          %3 = arith.index_cast %1 : i32 to index
          %4 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
          %5 = arith.addf %4, %2 : f32
          memref.store %5, %arg0[%arg4, %3] : memref<?x128xf32>
        } else {
          affine.if #set1()[%0] {
            %3 = arith.index_cast %1 : i32 to index
            %4 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
            %5 = arith.mulf %4, %2 : f32
            memref.store %5, %arg0[%arg4, %3] : memref<?x128xf32>
          } else {
            affine.if #set2()[%0] {
              %3 = arith.index_cast %1 : i32 to index
              %4 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
              %5 = arith.cmpf ogt, %4, %2 : f32
              %6 = scf.if %5 -> (f32) {
                %7 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
                scf.yield %7 : f32
              } else {
                scf.yield %2 : f32
              }
              memref.store %6, %arg0[%arg4, %3] : memref<?x128xf32>
            } else {
              %3 = arith.index_cast %1 : i32 to index
              %4 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
              %5 = arith.cmpf olt, %4, %2 : f32
              %6 = scf.if %5 -> (f32) {
                %7 = memref.load %arg0[%arg4, %3] : memref<?x128xf32>
                scf.yield %7 : f32
              } else {
                scf.yield %2 : f32
              }
              memref.store %6, %arg0[%arg4, %3] : memref<?x128xf32>
            }
          }
        }
      }
    }
    return
  }
}
