set(SOURCE_EXAMPLE
  examples.cpp
)

add_executable(metricexample ${SOURCE_EXAMPLE})
target_link_libraries(metricexample libcpu libmemory liblatency)
set_target_properties(metricexample PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examplebin/ )

