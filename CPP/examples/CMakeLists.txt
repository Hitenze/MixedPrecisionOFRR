cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
)

# Automatically find all example files
file(GLOB EXAMPLE_SOURCES "ex*.cu")

# Output found example files
message(STATUS "Found example files: ${EXAMPLE_SOURCES}")

# Create a list to hold targets
set(EXAMPLE_TARGETS "")

# Create executable for each example file
foreach(example_source ${EXAMPLE_SOURCES})
   # Get filename without extension
   get_filename_component(example_name ${example_source} NAME_WE)
   
   # Add executable
   add_executable(${example_name} ${example_source})
   
   # Add to target list
   list(APPEND EXAMPLE_TARGETS ${example_name})
   
   # Set compile options
   target_compile_definitions(${example_name} PRIVATE
      # Any required compile definitions
   )
   
   # Set CUDA options
   set_target_properties(${example_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
   )
   
   # Force output directory to be current binary dir (no subdirectories)
   set_target_properties(${example_name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
   )
   
   # Link libraries
   target_link_libraries(${example_name}
      msvd_containers
      msvd_linalg
      msvd_factorization
      msvd_core
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
   
   message(STATUS "Created example target: ${example_name}")
endforeach()

# Set installation rules (optional)
install(TARGETS ${EXAMPLE_TARGETS}
   RUNTIME DESTINATION examples
)