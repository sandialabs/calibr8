configure_file(notch_hyperJ2_verify.yaml.in notch_hyperJ2_verify.yaml @ONLY)

add_test(
  NAME notch_hyperJ2_verify
  COMMAND "${CMAKE_BINARY_DIR}/src/verify" "${CMAKE_BINARY_DIR}/test/verify/notch_hyperJ2_verify.yaml")
