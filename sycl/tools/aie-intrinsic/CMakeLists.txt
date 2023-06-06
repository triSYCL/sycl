# Intrisics are only know to the chess compiler. but not our compiler.
# So to be able to use them we build this librairy with the chess compiler
#  containing all the intrinsics we use and link it with our device code in sycl-chess.

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o
                   COMMAND xchesscc -p me -C Release_LLVM -d -f -c -o ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.cpp ${CMAKE_CURRENT_SOURCE_DIR}/aie-intrinsic.h
                   COMMENT "Built aie intrisics lib")

add_custom_target(aie-intrinsic ALL DEPENDS ${CMAKE_BINARY_DIR}/include/sycl/aie-intrinsic.h ${CMAKE_BINARY_DIR}/lib/aie-intrinsic.o)
