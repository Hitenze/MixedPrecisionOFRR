cmake_minimum_required(VERSION 3.18)

# Add the testproblems headers to the project
add_library(msvd_testproblems INTERFACE)

target_include_directories(msvd_testproblems INTERFACE
   ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(msvd_testproblems INTERFACE
   msvd_containers
) 