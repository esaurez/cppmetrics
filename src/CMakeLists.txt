set(SOURCES_STATISTICS
  statistics.cpp
)

add_library(libstatistics STATIC ${SOURCES_STATISTICS})
target_include_directories(libstatistics PUBLIC ${CMAKE_CURRENT_LIST_DIR})

set(SOURCES_LATENCY
  latency.cpp
)

if(${GPU})
  message("Latency with gpu clock")
  include_directories(${CUDA_INCLUDE_DIRS})
  CUDA_ADD_LIBRARY(liblatency STATIC ${SOURCES_LATENCY})
else()
  add_library(liblatency STATIC ${SOURCES_LATENCY})
endif()

target_include_directories(liblatency PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(liblatency PUBLIC libstatistics)

set(SOURCES_MEMORY
  memory.cpp
)

add_library(libmemory STATIC ${SOURCES_MEMORY})
target_include_directories(libmemory PUBLIC ${CMAKE_CURRENT_LIST_DIR})

set(SOURCES_CPU
  cpu.cpp
)

add_library(libcpu STATIC ${SOURCES_CPU})
target_include_directories(libcpu PUBLIC ${CMAKE_CURRENT_LIST_DIR})
