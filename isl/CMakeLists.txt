# ISL library.
set(ISL_MAIN_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR})
set(ISL_MAIN_INCLUDE_DIR ${ISL_MAIN_SRC_DIR}/include)

set(ISL_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(ISL_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(ISL_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

find_library(GMP_LIB gmp REQUIRED)

include_directories("isl")
include_directories("isl/include")
include_directories(${ISL_BINARY_DIR})
include_directories(${ISL_INCLUDE_DIR})

# We have a patched version of isl from
# https://github.com/chelini/isl
set(USE_GMP_FOR_MP ON)
set(ISL_GIT_HEAD_ID "7f2bda3")
include(polly/configure_isl.cmake)

# ISL files to compile.
set (ISL_FILES
  isl/basis_reduction_tab.c
  isl/isl_aff.c
  isl/isl_affine_hull.c
  isl/isl_aff_map.c
  isl/isl_arg.c
  isl/isl_ast_build.c
  isl/isl_ast_build_expr.c
  isl/isl_ast.c
  isl/isl_ast_codegen.c
  isl/isl_ast_graft.c
  isl/isl_bernstein.c
  isl/isl_blk.c
  isl/isl_bound.c
  isl/isl_box.c
  isl/isl_coalesce.c
  isl/isl_constraint.c
  isl/isl_convex_hull.c
  isl/isl_ctx.c
  isl/isl_deprecated.c
  isl/isl_dim_map.c
  isl/isl_equalities.c
  isl/isl_factorization.c
  isl/isl_farkas.c
  isl/isl_ffs.c
  isl/isl_flow.c
  isl/isl_fold.c
  isl/isl_gmp.c
  isl/isl_hash.c
  isl/isl_id.c
  isl/isl_id_to_ast_expr.c
  isl/isl_id_to_id.c
  isl/isl_id_to_pw_aff.c
  isl/isl_ilp.c
  #isl/isl_imath.c
  isl/isl_input.c
  #isl/isl_int_sioimath.c
  isl/isl_local.c
  isl/isl_local_space.c
  isl/isl_lp.c
  isl/isl_map.c
  isl/isl_map_list.c
  isl/isl_map_simplify.c
  isl/isl_map_subtract.c
  isl/isl_map_to_basic_set.c
  isl/isl_mat.c
  isl/isl_morph.c
  isl/isl_obj.c
  isl/isl_options.c
  isl/isl_output.c
  isl/isl_point.c
  isl/isl_polynomial.c
  isl/isl_printer.c
  isl/isl_range.c
  isl/isl_reordering.c
  isl/isl_sample.c
  isl/isl_scan.c
  isl/isl_schedule_band.c
  isl/isl_schedule.c
  isl/isl_schedule_constraints.c
  isl/isl_schedule_node.c
  isl/isl_scheduler.c
  isl/isl_schedule_read.c
  isl/isl_schedule_tree.c
  isl/isl_seq.c
  isl/isl_set_list.c
  isl/isl_set_to_ast_graft_list.c
  isl/isl_sort.c
  isl/isl_space.c
  isl/isl_stream.c
  isl/isl_stride.c
  isl/isl_tab.c
  isl/isl_tab_pip.c
  isl/isl_tarjan.c
  isl/isl_transitive_closure.c
  isl/isl_union_map.c
  isl/isl_val.c
  isl/isl_val_gmp.c
  #isl/isl_val_sioimath.c
  isl/isl_vec.c
  isl/isl_version.c
  isl/isl_vertices.c
  isl/print.c
  isl/set_from_map.c
  isl/set_to_map.c
  isl/uset_from_umap.c
  isl/uset_to_umap.c
  )

add_library(ISL STATIC ${ISL_FILES})
target_link_libraries(ISL PUBLIC ${GMP_LIB})
target_compile_options(ISL PRIVATE "-w")
