cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
)

# Create linalg library
add_library(msvd_linalg INTERFACE)

# Add subdirectories
add_subdirectory(blas)
add_subdirectory(factorization)
add_subdirectory(solvers)

# Link with all sub-libraries
target_link_libraries(msvd_linalg INTERFACE
   msvd_blas
   msvd_factorization   
   msvd_solvers
) 