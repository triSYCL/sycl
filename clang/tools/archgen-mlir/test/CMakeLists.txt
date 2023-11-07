configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ARCHGENMLIR_TEST_DEPENDS
        FileCheck count not
        archgen opt llc mlir-translate
        )

add_lit_testsuite(check-archgen "Running ArchGenMLIR tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ARCHGENMLIR_TEST_DEPENDS}
        )
set_target_properties(check-archgen PROPERTIES FOLDER "Tests")

add_lit_testsuites(archgen-mlir ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ARCHGENMLIR_TEST_DEPENDS})
