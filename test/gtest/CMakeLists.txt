add_library(my_gtest gtest-all.cc)
target_include_directories(my_gtest PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
