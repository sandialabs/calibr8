configure_file(notch2D_small_J2_adjoint_check.yaml.in notch2D_small_J2_adjoint_check.yaml @ONLY)

add_test(
  NAME notch2D_small_J2_adjoint_check
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/adjoint/notch2D_small_J2_adjoint_check.yaml")
