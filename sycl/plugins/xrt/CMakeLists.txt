
message(STATUS "Including the PI API Xilinx Runtime (XRT) backend.")


if (NOT DEFINED XILINX_RT_INCLUDE)
  # First check user environment
  if (DEFINED ENV{XILINX_XRT} AND 
      EXISTS "$ENV{XILINX_XRT}/include")
      message(STATUS "Got XRT installation path from $XILINX_XRT")
      set(XILINX_RT_INCLUDE "$ENV{XILINX_XRT}/include")
      
  # Check on standard XRT installation path
  elseif(EXISTS "/opt/xilinx/xrt/include")
    message(STATUS "Standard XRT installation path found")
    set(XILINX_RT_INCLUDE "/opt/xilinx/xrt/include")
  
  #TODO: Automatic download of XRT once their CMake is clean
  else()
    message(ERROR "No Xilinx runtime (XRT) installation found. Please set XILINX_XRT to point to a valid install")  
  endif()
endif()

if (DEFINED XILINX_RT_INCLUDE)
  file(GLOB_RECURSE XRT_SRC ${XILINX_RT_INCLUDE}/../lib/*.so)

  message(STATUS "Found xrt sources: ${XRT_SRC}")
  
  add_library(pi_xrt SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
  "pi_xrt.cpp"
  )

  add_dependencies(sycl-toolchain pi_xrt)

  set_target_properties(pi_xrt PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(pi_xrt PRIVATE ${sycl_inc_dir})
  target_include_directories(pi_xrt PRIVATE ${XILINX_RT_INCLUDE})
  target_link_libraries(pi_xrt
  PRIVATE
      OpenCL-Headers
      ${XILINX_RT_INCLUDE}/../lib/libxrt_coreutil.so
      -luuid
  )

  if (MSVC)
    # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
    # which are individually tagged for all pi* symbols in pi.h
    target_compile_definitions(pi_xrt PRIVATE __SYCL_BUILD_SYCL_DLL)
    # Install stripped PDB
    add_stripped_pdb(pi_xrt)
  else()
    # we set the visibility of all symbols 'hidden' by default.
    # In pi.h file, we set exported symbols with visibility==default individually
    target_compile_options(pi_xrt PUBLIC -fvisibility=hidden)
    # This script file is used to allow exporting pi* symbols only.
    # All other symbols are regarded as local (hidden)
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")
    # Filter symbols based on the scope defined in the script file,
    # and export pi* function symbols in the library.
    target_link_libraries(pi_xrt PRIVATE "-Wl,--version-script=${linker_script}")
  endif()
  add_common_options(pi_xrt)
  install(TARGETS pi_xrt
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_xrt
    RUNTIME DESTINATION "bin" COMPONENT pi_xrt
  )
endif()
