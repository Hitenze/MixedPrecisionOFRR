cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# BLAS configuration options
option(USE_OPENBLAS "Use OpenBLAS for CPU BLAS operations" OFF)
option(USE_MKL "Use Intel MKL for CPU BLAS operations" OFF)
set(OPENBLAS_PATH "" CACHE PATH "Path to OpenBLAS installation")
set(MKL_PATH "" CACHE PATH "Path to Intel MKL installation")

# Use system BLAS if no other options specified
if(NOT USE_OPENBLAS AND NOT USE_MKL)
   find_package(BLAS REQUIRED)
   set(BLAS_LIB ${BLAS_LIBRARIES})
   message(STATUS "Using system BLAS: ${BLAS_LIBRARIES}")
endif()

# OpenBLAS configuration
if(USE_OPENBLAS)
   # Set OpenBLAS search paths
   set(OPENBLAS_SEARCH_PATHS
      ${OPENBLAS_PATH}
      ${CMAKE_SOURCE_DIR}/../OpenBLAS/build
      ${CMAKE_SOURCE_DIR}/../../OpenBLAS/build
      /usr/local
      /usr
      /opt/OpenBLAS
      $ENV{HOME}/OpenBLAS/build
   )
   
   # Find OpenBLAS library and headers
   find_library(OPENBLAS_LIB
      NAMES openblas
      PATHS ${OPENBLAS_SEARCH_PATHS}
      PATH_SUFFIXES lib lib64
   )
   
   find_path(OPENBLAS_INCLUDE_DIR
      NAMES cblas.h
      PATHS ${OPENBLAS_SEARCH_PATHS}
      PATH_SUFFIXES include
   )
   
   if(OPENBLAS_LIB AND OPENBLAS_INCLUDE_DIR)
      set(BLAS_LIB ${OPENBLAS_LIB})
      include_directories(${OPENBLAS_INCLUDE_DIR})
      message(STATUS "Using OpenBLAS: ${OPENBLAS_LIB}")
      message(STATUS "OpenBLAS include: ${OPENBLAS_INCLUDE_DIR}")
      # Define OpenBLAS macro for use in code
      add_definitions(-DUSE_OPENBLAS)
   else()
      message(FATAL_ERROR "OpenBLAS requested but not found. Please specify OPENBLAS_PATH.")
   endif()
endif()

# MKL configuration
if(USE_MKL)
   # Set MKL search paths
   set(MKL_SEARCH_PATHS
      ${MKL_PATH}
      /opt/intel/mkl
      /opt/intel/oneapi/mkl
      $ENV{MKLROOT}
   )
   
   # Find MKL libraries and headers
   find_library(MKL_CORE_LIB
      NAMES mkl_core
      PATHS ${MKL_SEARCH_PATHS}
      PATH_SUFFIXES lib lib/intel64
   )
   
   find_library(MKL_INTEL_LP64_LIB
      NAMES mkl_intel_lp64
      PATHS ${MKL_SEARCH_PATHS}
      PATH_SUFFIXES lib lib/intel64
   )
   
   find_library(MKL_SEQUENTIAL_LIB
      NAMES mkl_sequential
      PATHS ${MKL_SEARCH_PATHS}
      PATH_SUFFIXES lib lib/intel64
   )
   
   find_path(MKL_INCLUDE_DIR
      NAMES mkl.h
      PATHS ${MKL_SEARCH_PATHS}
      PATH_SUFFIXES include
   )
   
   if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB AND MKL_INCLUDE_DIR)
      set(BLAS_LIB ${MKL_INTEL_LP64_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
      include_directories(${MKL_INCLUDE_DIR})
      message(STATUS "Using Intel MKL: ${MKL_CORE_LIB}")
      message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")
      # Define MKL macro for use in code
      add_definitions(-DUSE_MKL)
   else()
      message(FATAL_ERROR "Intel MKL requested but not found. Please specify MKL_PATH.")
   endif()
endif()

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
)

# Create blas library
add_library(msvd_blas
   mvops.cu
)

# Set CUDA options
set_target_properties(msvd_blas PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(msvd_blas
   msvd_containers
   msvd_utils
   CUDA::cudart
   CUDA::cublas
   ${BLAS_LIB}
) 