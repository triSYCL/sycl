cmake_minimum_required(VERSION 3.2)

project(sycl-solution)
# Requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
option(SYCL_ENABLE_WERROR "Treat all warnings as errors in SYCL project" OFF)

if(SYCL_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/W4 /WX ${CMAKE_CXX_FLAGS}")
    add_definitions(
      -wd4996 # Suppress 'function': was declared deprecated'
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-deprecated-declarations")
  endif()
endif()

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  # Skip asynchronous C++ exceptions catching and assume "extern C" functions
  # never throw C++ exceptions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# Get clang's version
include(VersionFromVCS)
set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

# If CLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED CLANG_VERSION_MAJOR)
  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED CLANG_VERSION_MINOR)
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED CLANG_VERSION_PATCHLEVEL)
  set(CLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, CLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(CLANG_VERSION "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}.${CLANG_VERSION_PATCHLEVEL}")

set(LLVM_INST_INC_DIRECTORY "lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include")
set(dst_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include)

# Find OpenCL headers and libraries installed in the system and use them to
# build SYCL runtime.
# WARNING: use with caution, building SYCL runtime with OpenCL implementation
# instead of Khronos ICD loader might cause build and/or portability issues.
option(OpenCL_BUILD_WITH_SYSTEM_SDK OFF)

if( (OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY) OR OpenCL_BUILD_WITH_SYSTEM_SDK)
  find_package(OpenCL)
endif()

include(ExternalProject)

if( NOT OpenCL_INCLUDE_DIRS )
  message("OpenCL_INCLUDE_DIRS is missed. Will try to download OpenCL headers from github.com")
  set(OpenCL_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc")
  ExternalProject_Add(ocl-headers
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${dst_dir}/CL
    INSTALL_COMMAND   ""
    STEP_TARGETS      build
    COMMENT           "Downloading OpenCL headers."
  )
else()
  add_custom_target( ocl-headers ALL
    DEPENDS ${OpenCL_INCLUDE_DIRS}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenCL_INCLUDE_DIRS}/CL ${dst_dir}/CL
    COMMENT "Copying OpenCL headers ..."
  )
endif()

if( NOT OpenCL_LIBRARIES )
  message("OpenCL_LIBRARIES is missed. Will try to download OpenCL ICD Loader from github.com")
  if(MSVC)
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenCL${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(OPENCL_ICD_LOADER_WDK "-DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF")
  else()
    set(OpenCL_LIBRARIES
        "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenCL${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(OPENCL_ICD_LOADER_WDK "")
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/icd_build)
  ExternalProject_Add(ocl-icd
    GIT_REPOSITORY    https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG           origin/master
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/OpenCL/icd"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/icd_build"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DOPENCL_ICD_LOADER_HEADERS_DIR=${CMAKE_CURRENT_BINARY_DIR}/OpenCL/inc
               -DCMAKE_INSTALL_PREFIX=${LLVM_BINARY_DIR}
               -DCMAKE_INSTALL_LIBDIR:PATH=lib
               ${OPENCL_ICD_LOADER_WDK}
    STEP_TARGETS      configure,build,install
    DEPENDS           ocl-headers
  )
else()
  file(GLOB ICD_LOADER_SRC "${OpenCL_LIBRARIES}*")
  file(COPY ${ICD_LOADER_SRC} DESTINATION ${LLVM_LIBRARY_OUTPUT_INTDIR})
  add_custom_target(ocl-icd DEPENDS ${OpenCL_LIBRARIES} COMMENT "Copying OpenCL ICD Loader ...")
endif()

set(OPENCL_INCLUDE "${OpenCL_INCLUDE_DIRS}")

add_library (OpenCL-Headers INTERFACE)
add_library (OpenCL::Headers ALIAS OpenCL-Headers)
target_include_directories(OpenCL-Headers
  INTERFACE ${OPENCL_INCLUDE}
)
install(DIRECTORY ${OPENCL_INCLUDE}/CL
  DESTINATION ${LLVM_INST_INC_DIRECTORY}
  COMPONENT opencl-headers
)

# Try to find and include XRT OpenCL in the build components to allow access to
# Xilinx OpenCL extensions in the source files when compiling the runtimes
# libSYCL.so, this unfortunately isn't relevant when a user is compiling there
# own files, they'll still have to add -I/opt/xilinx/xrt/include/.
# It's only relevant to the Clang build system and the libSYCL.so file that's
# compiled, the header files are recompiled on a per application basis as far as
# I'm aware.
# \todo We could enforce the download and installation of the XRT runtime
# similarly to what's done with the OpenCL headers at the moment that makes
# things a little easier. As it wouldn't matter if someone only wanted to build
# to run on another platform, they would always have XRT and it's extensions
# wouldn't break the runtime by being missing
if (DEFINED ENV{XILINX_XRT})
  if(EXISTS "$ENV{XILINX_XRT}/include" AND EXISTS "$ENV{XILINX_XRT}/lib")
    message(STATUS "XILINX_XRT enabled")
    message(STATUS "XILINX_XRT Include: $ENV{XILINX_XRT}/include")
    #message(STATUS "XILINX_XRT Lib: $ENV{XILINX_XRT}/lib")

    # Activate alternate code paths containing XRT extensions + normal SYCL code
    # path. Rather than only compiling the normal SYCL code path. When defined
    # by the CMake build system __SYCL_XILINX_ONLY__ is only really relevant to
    # source files that get compiled into the libSYCL.so library. The compiler
    # still has to internally define __SYCL_XILINX_ONLY__ when compiling user
    # code as it's still required by the headers that get compiled.
    # NOTE: I chose to use __SYCL_XILINX_ONLY__ for both the build define and
    # compiler generated define to cut down on the amount of macro usage, they
    # also at the time of creation do very similar things, this may change in
    # the future but I think it will be unlikely. As __SYCL_XILINX_ONLY__ is
    # mostly related to enforcing XILINX only code paths.
    add_definitions(-D__SYCL_XILINX_ONLY__)

    # Add SYSTEM infront to cheat the build system into thinking its a system
    # include and silencing warnings from XRT when building... don't want more
    # compiler noise that we can't control
    include_directories(SYSTEM $ENV{XILINX_XRT}/include)
    # link_directories(SYSTEM $ENV{XILINX_XRT}/lib)
  else()
    message(STATUS "Invalid XILINX_XRT no lib or include directories found")
  endif()
else()
  message(STATUS "XILINX_XRT not set, xilinx cl ext disabled")
endif()

# Configure SYCL version macro
set(sycl_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
set(version_header "${sycl_inc_dir}/CL/sycl/version.hpp")
configure_file("${version_header}.in" "${version_header}")

# Copy SYCL headers
add_custom_target(sycl-headers ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory ${sycl_inc_dir}/CL ${dst_dir}/CL
COMMENT "Copying SYCL headers ...")

# Configure SYCL headers
install(DIRECTORY "${sycl_inc_dir}/." DESTINATION "${LLVM_INST_INC_DIRECTORY}" COMPONENT sycl-headers)

# SYCL runtime library
add_subdirectory(source)

# SYCL toolchain builds all components: compiler, libraries, headers, etc.
add_custom_target( sycl-toolchain
  DEPENDS sycl
          clang
          clang-offload-wrapper
          clang-offload-bundler
          llc
          llvm-as
          llvm-ar
          llvm-dis
          llvm-no-spir-kernel
          llvm-spirv
          llvm-link
          sycl-xocc
          sycl-chess
          llvm-objcopy
          opt
  COMMENT "Building SYCL compiler toolchain..."
)

add_subdirectory( test )
add_subdirectory( unittests )
add_subdirectory( tools )

# Package deploy support
# Listed here are component names contributing the package
set( SYCL_TOOLCHAIN_DEPLOY_COMPONENTS
     clang
     clang-offload-wrapper
     clang-offload-bundler
     llc
     llvm-as
     llvm-ar
     llvm-dis
     llvm-no-spir-kernel
     llvm-spirv
     llvm-link
     llvm-objcopy
     opt
     clang-resource-headers
     opencl-headers
     sycl-headers
)

# Use it as fake dependency in order to force another command(s) to execute.
add_custom_command(OUTPUT __force_it
  COMMAND "${CMAKE_COMMAND}" -E echo
)
#Serialize installation to avoid missing components due to build race conditions
set(__chain_dep __force_it)

set(manifest_list)
foreach( comp ${SYCL_TOOLCHAIN_DEPLOY_COMPONENTS} )
  message( STATUS "Adding component ${comp} to deploy")

  set (manifest_file ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_${comp}.txt)
  add_custom_command(OUTPUT ${manifest_file}
    COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_INSTALL_COMPONENT=${comp}"
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    DEPENDS  ${__chain_dep}
    COMMENT "Deploying component ${comp}"
    USES_TERMINAL
  )
  list(APPEND manifest_list ${manifest_file})
  set(__chain_dep ${manifest_file})
endforeach( comp )

add_custom_target(deploy-sycl-toolchain
  DEPENDS sycl-toolchain ${manifest_list}
)
