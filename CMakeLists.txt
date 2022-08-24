cmake_minimum_required(VERSION 3.12.0...${CMAKE_VERSION})

project(CALIBR8 VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
enable_testing()

function(set_code_standards target_name)
  set_property(TARGET ${target_name} PROPERTY CXX_STANDARD "17")
  set_property(TARGET ${target_name} PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ${target_name} PROPERTY CXX_EXTENSIONS OFF)
endfunction()

# deal with Kokkos' lack of CMake skills
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI)

include(cmake/dependencies.cmake)
#add_subdirectory(doc)
#add_subdirectory(src)
#add_subdirectory(test)

add_subdirectory(research)
