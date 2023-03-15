cmake_minimum_required(VERSION 3.12.0...${CMAKE_VERSION})

project(CALIBR8 VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
enable_testing()

function(set_code_standards target_name)
  set_property(TARGET ${target_name} PROPERTY CXX_STANDARD "14")
  set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ${target_name} PROPERTY CXX_EXTENSIONS OFF)
endfunction()

function(install_and_export target_name)
  install(
    TARGETS ${target_name}
    EXPORT ${target_name}-target
  )
  install(EXPORT ${target_name}-target
    DESTINATION lib/cmake/${PROJECT_NAME})
endfunction()

# deal with Kokkos' lack of CMake skills
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI)

include(cmake/dependencies.cmake)
add_subdirectory(doc)
add_subdirectory(src)
add_subdirectory(test)
