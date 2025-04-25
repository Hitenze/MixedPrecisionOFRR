cmake_minimum_required(VERSION 3.18)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_SOURCE_DIR}/../
   ${CUDAToolkit_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(eig)
add_subdirectory(svd)

# Create a combined solvers library
add_library(msvd_solvers INTERFACE)

# Link with subdirectory libraries
target_link_libraries(msvd_solvers INTERFACE
   msvd_eig_solvers
   msvd_svd_solvers
) 