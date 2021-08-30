configure_file(notch_hyperJ2_error.yaml.in notch_hyperJ2_error.yaml @ONLY)

add_test(
  NAME notch_hyperJ2_error
  COMMAND "${CMAKE_BINARY_DIR}/src/error" "${CMAKE_BINARY_DIR}/test/error/notch_hyperJ2_error.yaml")
