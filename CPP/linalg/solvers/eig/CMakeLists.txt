cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_SOURCE_DIR}/../
   ${CUDAToolkit_INCLUDE_DIRS}
)

# Set source files
set(EIG_SOLVER_SOURCES
   si_eig.cu
   arnoldi_eig.cu
   rr_eig.cu
)

# Set header files
set(EIG_SOLVER_HEADERS
   si_eig.hpp
   arnoldi_eig.hpp
   rr_eig.hpp
)

# Create eig solvers library
add_library(msvd_eig_solvers
   ${EIG_SOLVER_SOURCES}
)

# Set CUDA options
set_target_properties(msvd_eig_solvers PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Link with dependencies
target_link_libraries(msvd_eig_solvers
   msvd_containers
   msvd_utils
   msvd_blas
   msvd_factorization
   ${LAPACK_LIBRARIES}
) 