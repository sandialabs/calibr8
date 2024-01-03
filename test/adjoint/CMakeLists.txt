configure_file(notch2D_small_J2_adjoint_check.yaml.in notch2D_small_J2_adjoint_check.yaml @ONLY)
configure_file(2prob.yaml.in 2prob.yaml @ONLY)

add_test(
  NAME notch2D_small_J2_adjoint_check
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/adjoint/notch2D_small_J2_adjoint_check.yaml")

add_test(
  NAME 2prob
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/adjoint/2prob.yaml")
