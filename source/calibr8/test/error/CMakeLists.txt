configure_file(notch_hyper_J2_error.yaml.in notch_hyper_J2_error.yaml @ONLY)

add_test(
  NAME notch_hyper_J2_error
  COMMAND "${CMAKE_BINARY_DIR}/src/error" "${CMAKE_BINARY_DIR}/test/error/notch_hyper_J2_error.yaml")
