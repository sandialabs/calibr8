configure_file(notch_hyperJ2_adjoint_check.yaml.in notch_hyperJ2_adjoint_check.yaml @ONLY)

add_test(
  NAME notch_hyperJ2_adjoint_check
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/adjoint/notch_hyperJ2_adjoint_check.yaml")
