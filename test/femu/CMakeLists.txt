configure_file(notch_hyperJ2_synthetic.yaml.in notch_hyperJ2_synthetic.yaml @ONLY)
configure_file(notch_hyperJ2_FEMU.yaml.in notch_hyperJ2_FEMU.yaml @ONLY)

add_test(
  NAME notch_hyperJ2_synthetic
  COMMAND "${CMAKE_BINARY_DIR}/src/primal" "${CMAKE_BINARY_DIR}/test/femu/notch_hyperJ2_synthetic.yaml")

add_test(
  NAME notch_hyperJ2_FEMU
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/femu/notch_hyperJ2_FEMU.yaml")

set_tests_properties(notch_hyperJ2_FEMU PROPERTIES DEPENDS notch_hyperJ2_synthetic)
