#map = affine_map<(d0, d1) -> (d1 + d0 * 1125)>
#map1 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 125)>
#map2 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 250)>
#map3 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 375)>
#map4 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 500)>
#map5 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 625)>
#map6 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 750)>
#map7 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 875)>
#map8 = affine_map<(d0, d1) -> (d1 + d0 * 1125 + 1000)>
#map9 = affine_map<(d0, d1) -> (d1 + d0 * 125)>
#map10 = affine_map<(d0, d1) -> (d0, d1)>
#map11 = affine_map<(d0, d1) -> (d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_elasticity_qpoint_3d_scalarized(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c125 = arith.constant 125 : index
    %0 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %1 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %2 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %3 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %4 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %5 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %6 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %7 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %8 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %9 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %10 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %11 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %12 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %13 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %14 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %15 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %16 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %17 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %18 = polygeist.submap(%arg5, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %arg3, %16, %17 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%18 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %173, %184 : f64
      %186 = arith.mulf %in_1, %in_6 : f64
      %187 = arith.mulf %in_0, %in_7 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in_0, %in_4 : f64
      %191 = arith.mulf %in_1, %in_3 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_12 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %185 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_8, %in_8 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_9, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_10, %in_13 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    %19 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %20 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %21 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %22 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %23 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %24 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %25 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %26 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %27 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %28 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %29 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %30 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %31 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %32 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %33 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %34 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %35 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %36 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %37 = polygeist.submap(%arg5, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %arg3, %35, %36 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%37 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %173, %184 : f64
      %186 = arith.mulf %in_1, %in_6 : f64
      %187 = arith.mulf %in_0, %in_7 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in_0, %in_4 : f64
      %191 = arith.mulf %in_1, %in_3 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_11 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %189 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_10, %in_9 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_11, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_12, %in_13 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    %38 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %39 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %40 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %41 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %42 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %43 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %44 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %45 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %46 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %47 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %48 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %49 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %50 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %51 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %52 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %53 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %54 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %55 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %56 = polygeist.submap(%arg5, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %arg3, %54, %55 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%56 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %173, %184 : f64
      %186 = arith.mulf %in_1, %in_6 : f64
      %187 = arith.mulf %in_0, %in_7 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in_0, %in_4 : f64
      %191 = arith.mulf %in_1, %in_3 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_10 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %193 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_12, %in_9 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_13, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_14, %in_14 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    %57 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %58 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %59 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %60 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %61 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %62 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %63 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %64 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %65 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %66 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %67 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %68 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %69 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %70 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %71 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %72 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %73 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %74 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %75 = polygeist.submap(%arg5, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %arg3, %73, %74 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%75 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.subf %176, %175 : f64
      %186 = arith.divf %185, %184 : f64
      %187 = arith.mulf %in, %in_7 : f64
      %188 = arith.mulf %in_1, %in_5 : f64
      %189 = arith.subf %187, %188 : f64
      %190 = arith.divf %189, %184 : f64
      %191 = arith.mulf %in_1, %in_2 : f64
      %192 = arith.mulf %in, %in_4 : f64
      %193 = arith.subf %191, %192 : f64
      %194 = arith.divf %193, %184 : f64
      %195 = arith.addf %in_8, %in_12 : f64
      %196 = arith.addf %195, %in_14 : f64
      %197 = arith.mulf %in_15, %184 : f64
      %198 = arith.mulf %in_16, %186 : f64
      %199 = arith.mulf %198, %196 : f64
      %200 = arith.addf %in_8, %in_8 : f64
      %201 = arith.mulf %186, %200 : f64
      %202 = arith.addf %in_9, %in_11 : f64
      %203 = arith.mulf %190, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.addf %in_10, %in_13 : f64
      %206 = arith.mulf %194, %205 : f64
      %207 = arith.addf %204, %206 : f64
      %208 = arith.mulf %in_17, %207 : f64
      %209 = arith.addf %199, %208 : f64
      %210 = arith.mulf %197, %209 : f64
      linalg.yield %210 : f64
    }
    %76 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %77 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %78 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %79 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %80 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %81 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %82 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %83 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %84 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %85 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %86 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %87 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %88 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %89 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %90 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %91 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %92 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %93 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %94 = polygeist.submap(%arg5, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %arg3, %92, %93 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%94 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.subf %176, %175 : f64
      %186 = arith.divf %185, %184 : f64
      %187 = arith.mulf %in, %in_7 : f64
      %188 = arith.mulf %in_1, %in_5 : f64
      %189 = arith.subf %187, %188 : f64
      %190 = arith.divf %189, %184 : f64
      %191 = arith.mulf %in_1, %in_2 : f64
      %192 = arith.mulf %in, %in_4 : f64
      %193 = arith.subf %191, %192 : f64
      %194 = arith.divf %193, %184 : f64
      %195 = arith.addf %in_8, %in_11 : f64
      %196 = arith.addf %195, %in_14 : f64
      %197 = arith.mulf %in_15, %184 : f64
      %198 = arith.mulf %in_16, %190 : f64
      %199 = arith.mulf %198, %196 : f64
      %200 = arith.addf %in_10, %in_9 : f64
      %201 = arith.mulf %186, %200 : f64
      %202 = arith.addf %in_11, %in_11 : f64
      %203 = arith.mulf %190, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.addf %in_12, %in_13 : f64
      %206 = arith.mulf %194, %205 : f64
      %207 = arith.addf %204, %206 : f64
      %208 = arith.mulf %in_17, %207 : f64
      %209 = arith.addf %199, %208 : f64
      %210 = arith.mulf %197, %209 : f64
      linalg.yield %210 : f64
    }
    %95 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %96 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %97 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %98 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %99 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %100 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %101 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %102 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %103 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %104 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %105 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %106 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %107 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %108 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %109 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %110 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %111 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %112 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %113 = polygeist.submap(%arg5, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %arg3, %111, %112 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%113 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.subf %176, %175 : f64
      %186 = arith.divf %185, %184 : f64
      %187 = arith.mulf %in, %in_7 : f64
      %188 = arith.mulf %in_1, %in_5 : f64
      %189 = arith.subf %187, %188 : f64
      %190 = arith.divf %189, %184 : f64
      %191 = arith.mulf %in_1, %in_2 : f64
      %192 = arith.mulf %in, %in_4 : f64
      %193 = arith.subf %191, %192 : f64
      %194 = arith.divf %193, %184 : f64
      %195 = arith.addf %in_8, %in_10 : f64
      %196 = arith.addf %195, %in_14 : f64
      %197 = arith.mulf %in_15, %184 : f64
      %198 = arith.mulf %in_16, %194 : f64
      %199 = arith.mulf %198, %196 : f64
      %200 = arith.addf %in_12, %in_9 : f64
      %201 = arith.mulf %186, %200 : f64
      %202 = arith.addf %in_13, %in_11 : f64
      %203 = arith.mulf %190, %202 : f64
      %204 = arith.addf %201, %203 : f64
      %205 = arith.addf %in_14, %in_14 : f64
      %206 = arith.mulf %194, %205 : f64
      %207 = arith.addf %204, %206 : f64
      %208 = arith.mulf %in_17, %207 : f64
      %209 = arith.addf %199, %208 : f64
      %210 = arith.mulf %197, %209 : f64
      linalg.yield %210 : f64
    }
    %114 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %115 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %116 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %117 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %118 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %119 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %120 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %121 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %122 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %123 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %124 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %125 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %126 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %127 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %128 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %129 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %130 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %131 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %132 = polygeist.submap(%arg5, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128, %129, %arg3, %130, %131 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%132 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %182, %184 : f64
      %186 = arith.mulf %in_0, %in_5 : f64
      %187 = arith.mulf %in, %in_6 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in, %in_3 : f64
      %191 = arith.mulf %in_0, %in_2 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_12 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %185 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_8, %in_8 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_9, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_10, %in_13 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    %133 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %134 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %135 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %136 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %137 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %138 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %139 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %140 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %141 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %142 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %143 = polygeist.submap(%arg4, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %144 = polygeist.submap(%arg4, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %145 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %146 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %147 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %148 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %149 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %150 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %151 = polygeist.submap(%arg5, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %arg3, %149, %150 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%151 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %182, %184 : f64
      %186 = arith.mulf %in_0, %in_5 : f64
      %187 = arith.mulf %in, %in_6 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in, %in_3 : f64
      %191 = arith.mulf %in_0, %in_2 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_11 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %189 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_10, %in_9 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_11, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_12, %in_13 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    %152 = polygeist.submap(%arg2, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %153 = polygeist.submap(%arg2, %c2, %c125) {map = #map1} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %154 = polygeist.submap(%arg2, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %155 = polygeist.submap(%arg2, %c2, %c125) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %156 = polygeist.submap(%arg2, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %157 = polygeist.submap(%arg2, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %158 = polygeist.submap(%arg2, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %159 = polygeist.submap(%arg2, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %160 = polygeist.submap(%arg2, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %161 = polygeist.submap(%arg4, %c2, %c125) {map = #map} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %162 = polygeist.submap(%arg4, %c2, %c125) {map = #map2} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %163 = polygeist.submap(%arg4, %c2, %c125) {map = #map4} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %164 = polygeist.submap(%arg4, %c2, %c125) {map = #map5} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %165 = polygeist.submap(%arg4, %c2, %c125) {map = #map6} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %166 = polygeist.submap(%arg4, %c2, %c125) {map = #map7} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %167 = polygeist.submap(%arg4, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %168 = polygeist.submap(%arg0, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %169 = polygeist.submap(%arg1, %c2, %c125) {map = #map9} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    %170 = polygeist.submap(%arg5, %c2, %c125) {map = #map8} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map10, #map11, #map10, #map10, #map10], iterator_types = ["parallel", "parallel"]} ins(%152, %153, %154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165, %166, %167, %arg3, %168, %169 : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?x?xf64>, memref<?x?xf64>) outs(%170 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %in_9: f64, %in_10: f64, %in_11: f64, %in_12: f64, %in_13: f64, %in_14: f64, %in_15: f64, %in_16: f64, %in_17: f64, %out: f64):
      %171 = arith.mulf %in_3, %in_7 : f64
      %172 = arith.mulf %in_4, %in_6 : f64
      %173 = arith.subf %171, %172 : f64
      %174 = arith.mulf %in, %173 : f64
      %175 = arith.mulf %in_2, %in_7 : f64
      %176 = arith.mulf %in_4, %in_5 : f64
      %177 = arith.subf %175, %176 : f64
      %178 = arith.mulf %in_0, %177 : f64
      %179 = arith.subf %174, %178 : f64
      %180 = arith.mulf %in_2, %in_6 : f64
      %181 = arith.mulf %in_3, %in_5 : f64
      %182 = arith.subf %180, %181 : f64
      %183 = arith.mulf %in_1, %182 : f64
      %184 = arith.addf %179, %183 : f64
      %185 = arith.divf %182, %184 : f64
      %186 = arith.mulf %in_0, %in_5 : f64
      %187 = arith.mulf %in, %in_6 : f64
      %188 = arith.subf %186, %187 : f64
      %189 = arith.divf %188, %184 : f64
      %190 = arith.mulf %in, %in_3 : f64
      %191 = arith.mulf %in_0, %in_2 : f64
      %192 = arith.subf %190, %191 : f64
      %193 = arith.divf %192, %184 : f64
      %194 = arith.addf %in_8, %in_10 : f64
      %195 = arith.addf %194, %in_14 : f64
      %196 = arith.mulf %in_15, %184 : f64
      %197 = arith.mulf %in_16, %193 : f64
      %198 = arith.mulf %197, %195 : f64
      %199 = arith.addf %in_12, %in_9 : f64
      %200 = arith.mulf %185, %199 : f64
      %201 = arith.addf %in_13, %in_11 : f64
      %202 = arith.mulf %189, %201 : f64
      %203 = arith.addf %200, %202 : f64
      %204 = arith.addf %in_14, %in_14 : f64
      %205 = arith.mulf %193, %204 : f64
      %206 = arith.addf %203, %205 : f64
      %207 = arith.mulf %in_17, %206 : f64
      %208 = arith.addf %198, %207 : f64
      %209 = arith.mulf %196, %208 : f64
      linalg.yield %209 : f64
    }
    return
  }
}
