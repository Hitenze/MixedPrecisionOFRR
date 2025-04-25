cmake_minimum_required(VERSION 3.18)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add custom target for generating coverage report if coverage is enabled
if(ENABLE_COVERAGE)
   find_program(GCOV_PATH gcov)
   find_program(LCOV_PATH lcov)
   find_program(GENHTML_PATH genhtml)
   
   if(GCOV_PATH AND LCOV_PATH AND GENHTML_PATH)
      add_custom_target(coverage
         # Run tests
         COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
         COMMAND ${CMAKE_CTEST_COMMAND} --verbose
         
         # Generate coverage info
         COMMAND ${LCOV_PATH} --directory . --capture --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.info --ignore-errors mismatch,source
         
         # Filter out system headers and test files
         COMMAND ${LCOV_PATH} --remove ${CMAKE_BINARY_DIR}/coverage/coverage.info '/usr/include/*' '/usr/local/*' '${CMAKE_SOURCE_DIR}/tests/*' --output-file ${CMAKE_BINARY_DIR}/coverage/coverage.filtered.info --ignore-errors mismatch,source
         
         # Generate HTML report
         COMMAND ${GENHTML_PATH} ${CMAKE_BINARY_DIR}/coverage/coverage.filtered.info --output-directory ${CMAKE_BINARY_DIR}/coverage/html --synthesize-missing --ignore-errors source
         
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
         COMMENT "Generating code coverage report"
      )
   else()
      message(WARNING "gcov, lcov or genhtml not found. Coverage report generation not available. Please install lcov package.")
   endif()
endif()

# Option to download GTest instead of using system installation
option(DOWNLOAD_GTEST "Download and build GTest locally" ON)

if(DOWNLOAD_GTEST)
   # Download and build GTest from source
   include(FetchContent)
   FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.1
   )
   FetchContent_MakeAvailable(googletest)
else()
   # Use system-installed GTest
   find_package(GTest REQUIRED)
endif()

# Include directories
include_directories(
   ${CMAKE_SOURCE_DIR}
)

# Add GTest include directories if using system installation
if(NOT DOWNLOAD_GTEST)
   include_directories(${GTEST_INCLUDE_DIRS})
endif()

# Add the vector_test executable
add_executable(vector_test
   vector_test.cpp
)

# Set compile definitions
target_compile_definitions(vector_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options
set_target_properties(vector_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(vector_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries
if(DOWNLOAD_GTEST)
   target_link_libraries(vector_test
      msvd_containers
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(vector_test
      msvd_containers
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add the matrix_test executable
add_executable(matrix_test
   matrix_test.cpp
)

# Set compile definitions for matrix_test
target_compile_definitions(matrix_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options for matrix_test
set_target_properties(matrix_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(matrix_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries for matrix_test
if(DOWNLOAD_GTEST)
   target_link_libraries(matrix_test
      msvd_containers
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(matrix_test
      msvd_containers
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add the gaussian_kernel_test executable
add_executable(gaussian_kernel_test
   gaussian_kernel_test.cpp
)

# Set compile definitions for gaussian_kernel_test
target_compile_definitions(gaussian_kernel_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options for gaussian_kernel_test
set_target_properties(gaussian_kernel_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(gaussian_kernel_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries for gaussian_kernel_test
if(DOWNLOAD_GTEST)
   target_link_libraries(gaussian_kernel_test
      msvd_containers
      msvd_testproblems
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(gaussian_kernel_test
      msvd_containers
      msvd_testproblems
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add tests
add_test(NAME VectorTest 
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vector_test
)

add_test(NAME MatrixTest
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/matrix_test
)

add_test(NAME GaussianKernelTest
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/gaussian_kernel_test
)

# Add the mvops_test executable
add_executable(mvops_test
   mvops_test.cpp
)

# Set compile definitions for mvops_test
target_compile_definitions(mvops_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options for mvops_test
set_target_properties(mvops_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(mvops_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries for mvops_test
if(DOWNLOAD_GTEST)
   target_link_libraries(mvops_test
      msvd_containers
      msvd_linalg
      msvd_core
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(mvops_test
      msvd_containers
      msvd_linalg
      msvd_core
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add the qr_test executable
add_executable(qr_test
   qr_test.cpp
)

# Set compile definitions for qr_test
target_compile_definitions(qr_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options for qr_test
set_target_properties(qr_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(qr_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries for qr_test
if(DOWNLOAD_GTEST)
   target_link_libraries(qr_test
      msvd_containers
      msvd_linalg
      msvd_factorization
      msvd_core
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(qr_test
      msvd_containers
      msvd_linalg
      msvd_factorization
      msvd_core
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add mvops_test test
add_test(NAME MvopsTest
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/mvops_test
)

# Add qr_test test
add_test(NAME QRTest
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/qr_test
)

# Add the hessenberg_test executable
add_executable(hessenberg_test
   hessenberg_test.cpp
)

# Set compile definitions for hessenberg_test
target_compile_definitions(hessenberg_test PRIVATE
   -DGOOGLE_TEST
)

# Set CUDA options for hessenberg_test
set_target_properties(hessenberg_test PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)

# Force the output directory to be in the current binary dir without subdirectories
set_target_properties(hessenberg_test PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Link libraries for hessenberg_test
if(DOWNLOAD_GTEST)
   target_link_libraries(hessenberg_test
      msvd_containers
      msvd_linalg
      msvd_factorization
      msvd_core
      gtest
      gtest_main
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
else()
   target_link_libraries(hessenberg_test
      msvd_containers
      msvd_linalg
      msvd_factorization
      msvd_core
      ${GTEST_LIBRARIES}
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      CUDA::curand
      pthread
   )
endif()

# Add hessenberg test
add_test(NAME HessenbergTest
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/hessenberg_test
)

# Enable testing
enable_testing() 