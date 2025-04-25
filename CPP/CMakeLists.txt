cmake_minimum_required(VERSION 3.18)

# Option for code coverage
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# Try to find common CUDA paths
if(NOT DEFINED CMAKE_CUDA_COMPILER)
   # Common CUDA installation paths
   set(COMMON_CUDA_PATHS
      /usr/local/cuda
      /usr/local/cuda-12.8
      /usr/local/cuda-12.0
      /usr/local/cuda-11.8
      /usr/local/cuda-11.7
      /usr/local/cuda-11.6
      /usr/local/cuda-11.5
      /usr/local/cuda-11.4
      /usr/local/cuda-11.3
      /usr/local/cuda-11.2
      /usr/local/cuda-11.1
      /usr/local/cuda-11.0
      /usr/local/cuda-10.2
      /usr/local/cuda-10.1
      /usr/local/cuda-10.0
      /opt/cuda
      /usr/lib/cuda
      ${CUDA_PATH}
      ${CUDA_TOOLKIT_ROOT_DIR}
      $ENV{CUDA_PATH}
      $ENV{CUDA_TOOLKIT_ROOT_DIR}
      $ENV{CUDA_HOME}
   )
   
   # Search for nvcc in common paths
   foreach(cuda_path ${COMMON_CUDA_PATHS})
      if(EXISTS "${cuda_path}/bin/nvcc")
         message(STATUS "Found CUDA at: ${cuda_path}")
         set(CMAKE_CUDA_COMPILER "${cuda_path}/bin/nvcc")
         set(CUDA_TOOLKIT_ROOT_DIR "${cuda_path}")
         break()
      endif()
   endforeach()
   
   if(NOT DEFINED CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "Could not find CUDA compiler. Please set CMAKE_CUDA_COMPILER=/path/to/nvcc or install CUDA toolkit.")
   endif()
endif()

# Set project name and languages
project(MSVD LANGUAGES CXX CUDA)

# BLAS configuration options (passed to subprojects)
option(USE_OPENBLAS "Use OpenBLAS for CPU BLAS operations" OFF)
option(USE_MKL "Use Intel MKL for CPU BLAS operations" OFF)
set(OPENBLAS_PATH "" CACHE PATH "Path to OpenBLAS installation")
set(MKL_PATH "" CACHE PATH "Path to Intel MKL installation")

# Display configuration information
message(STATUS "BLAS configuration:")
message(STATUS "  USE_OPENBLAS: ${USE_OPENBLAS}")
message(STATUS "  USE_MKL: ${USE_MKL}")
if(USE_OPENBLAS AND OPENBLAS_PATH)
   message(STATUS "  OPENBLAS_PATH: ${OPENBLAS_PATH}")
endif()
if(USE_MKL AND MKL_PATH)
   message(STATUS "  MKL_PATH: ${MKL_PATH}")
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure code coverage if enabled
if(ENABLE_COVERAGE)
   message(STATUS "Enabling code coverage")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage -fprofile-arcs -ftest-coverage")
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --coverage -Xcompiler=-fprofile-arcs,-ftest-coverage")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Set CUDA architecture to auto-detection
# set(CUDA_ARCHITECTURES "60;70;75;80;86" CACHE STRING "CUDA architectures to compile for")
set(CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to compile for")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
message(STATUS "Setting CUDA architectures to: ${CUDA_ARCHITECTURES}")

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
)

# Add directories
add_subdirectory(core)
add_subdirectory(containers)
add_subdirectory(testproblems)
add_subdirectory(linalg)

# Build tests if requested
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    # Enable CTest
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples if requested
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()