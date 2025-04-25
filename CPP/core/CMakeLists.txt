# Include subdirectories
add_subdirectory(memory)
add_subdirectory(utils)

# Create core library
add_library(msvd_core INTERFACE)

# Set dependencies
target_link_libraries(msvd_core INTERFACE
   msvd_memory
   msvd_utils
) 