configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
)

if(POLYGEIST_POLYMER_ENABLE_PLUTO)
  set(POLYMER_TEST_DEPENDS
    FileCheck count not
    polymer-translate
    polymer-opt
  )
elseif(POLYGEIST_POLYMER_ENABLE_ISL)
  set(POLYMER_TEST_DEPENDS
    FileCheck count not
    polymer-opt
  )
endif()

add_lit_testsuite(check-polymer "Running the Polymer regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${POLYMER_TEST_DEPENDS}
)
set_target_properties(check-polymer PROPERTIES FOLDER "Tests")

add_lit_testsuites(POLYMER ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${POLYMER_TEST_DEPS}
)
