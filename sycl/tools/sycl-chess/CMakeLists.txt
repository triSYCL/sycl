# \TODO: Perhaps this should be walled behind a build option, it may also
# require a windows compliant variation of the shell script that will be
# selected based on the compile target similar to scan-build.
set(BinFiles
      sycl-chess)

foreach(BinFile ${BinFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${BinFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/bin
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile}
                       ${CMAKE_BINARY_DIR}/bin/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${BinFile})
  install(PROGRAMS bin/${BinFile} DESTINATION bin)
endforeach()

# \TODO: Might be able to refactor this into a single foreach
set(ChessWrapperFiles
      chesswrappers/chess-clang
      chesswrappers/linker_script.bcf
  )

foreach(ChessWrapper ${ChessWrapperFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${ChessWrapper}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/bin/chesswrappers
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/bin/${ChessWrapper}
                       ${CMAKE_BINARY_DIR}/bin/chesswrappers/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${ChessWrapper})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${ChessWrapper})
  install(PROGRAMS bin/${ChessWrapper} DESTINATION bin/chesswrappers)
endforeach()

add_custom_target(sycl-chess ALL DEPENDS ${Depends})
