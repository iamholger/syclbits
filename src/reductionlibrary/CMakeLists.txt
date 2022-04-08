cmake_minimum_required(VERSION 3.14)
project(reductionlibrary)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")


SET(SYCL_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda--sm_35,spir64,spir64_x86_64 -Wno-error -g3 -DDEBUG")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

#message(STATUS "Before: ${CMAKE_CXX_FLAGS}")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
execute_process (
    COMMAND sed -i "s| -Werror||g" ./_deps/googletest-src/googletest/cmake/internal_utils.cmake
)
FetchContent_MakeAvailable(googletest)
#execute_process (
    #COMMAND sed -i "s| -Werror||g" ./_deps/googletest-src/googletest/cmake/internal_utils.cmake
#)

# Need to get rid of Werror due to the target triplet stuff --- at least for now


enable_testing()


add_executable(
  test_reductionlibrary
  test_reductionlibrary.cxx
)

target_link_libraries(
  test_reductionlibrary
  gtest_main
)


#include(GoogleTest)
#gtest_discover_tests(test_reductionlibrary)
