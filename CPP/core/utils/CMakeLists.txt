# Add CUDA libraries
find_package(CUDAToolkit REQUIRED)

# Create utils library 
add_library(msvd_utils 
   cuda_handler.cu
)

# Set CUDA options
set_target_properties(msvd_utils PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Link with CUDA libraries
target_link_libraries(msvd_utils
   CUDA::cudart
   CUDA::cublas
   CUDA::cusparse
   CUDA::cusolver
) 