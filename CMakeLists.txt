project(CppMetricsLib)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(GPU TRUE)
  add_definitions(-DGPU_CAPABLE)
  message(STATUS "CUDA library status:")
  message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "    version: ${CUDA_VERSION}")
  message(STATUS "    libraries: ${CUDA_LIBRARIES}")
  message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
else()
  message("CUDA not found, compiling without cuda")
endif()

add_definitions(-std=c++11)
add_subdirectory(src)

option(Examples "Examples" OFF)

if(Examples)
  message("Compiling examples")
  add_subdirectory(examples)
else()
  message("Not compiling examples")
endif()
