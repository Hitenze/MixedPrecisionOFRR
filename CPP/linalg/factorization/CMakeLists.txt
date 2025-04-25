cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_SOURCE_DIR}/../
   ${CUDAToolkit_INCLUDE_DIRS}
)

# Set source files
set(FACTORIZATION_SOURCES
   qr.cu
   hessenberg.cu
)

# Set header files
set(FACTORIZATION_HEADERS
   qr.hpp
   hessenberg.hpp
)

# Create factorization library
add_library(msvd_factorization
   ${FACTORIZATION_SOURCES}
)

# Set CUDA options
set_target_properties(msvd_factorization PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Link with dependencies
target_link_libraries(msvd_factorization
   msvd_containers
   msvd_utils
   msvd_blas
) 