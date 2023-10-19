configure_file(unit.yaml.in unit.yaml @ONLY)
configure_file(quadrature.cpp.in quadrature.cpp @ONLY)
configure_file(disc.cpp.in disc.cpp @ONLY)

# the order of these tests matters
set(CALIBR8_UNIT_TEST_SOURCES
  control.cpp
  ${CMAKE_BINARY_DIR}/test/unit/quadrature.cpp
  ${CMAKE_BINARY_DIR}/test/unit/disc.cpp
  global_residual.cpp
  unit_tests.cpp
)

add_executable(calibr8_unit_tests ${CALIBR8_UNIT_TEST_SOURCES})
target_link_libraries(calibr8_unit_tests PRIVATE calibr8)
target_link_libraries(calibr8_unit_tests PRIVATE Gtest::gtest)
set_code_standards(calibr8_unit_tests)

add_test(NAME unit_tests COMMAND "./calibr8_unit_tests")
