# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# Collect source files
set(CONTAINERS_SOURCES
   vector.cu
   matrix.cu
)

# Create library
add_library(msvd_containers STATIC ${CONTAINERS_SOURCES})

# Set CUDA compilation options
set_target_properties(msvd_containers PROPERTIES 
   CUDA_SEPARABLE_COMPILATION ON
)

# Link dependencies
target_link_libraries(msvd_containers
   msvd_core
   CUDA::cudart
   CUDA::cublas
   CUDA::cusparse
   CUDA::curand
) 