cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_SOURCE_DIR}/../
   ${CUDAToolkit_INCLUDE_DIRS}
)

# Set source files
set(SVD_SOLVER_SOURCES
   si_svd.cu
   arnoldi_svd.cu
   rr_svd.cu
)

# Set header files
set(SVD_SOLVER_HEADERS
   si_svd.hpp
   arnoldi_svd.hpp
   rr_svd.hpp
)

# Create svd solvers library
add_library(msvd_svd_solvers
   ${SVD_SOLVER_SOURCES}
)

# Set CUDA options
set_target_properties(msvd_svd_solvers PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Link with dependencies
target_link_libraries(msvd_svd_solvers
   msvd_containers
   msvd_utils
   msvd_blas
   msvd_factorization
   ${LAPACK_LIBRARIES}
) 