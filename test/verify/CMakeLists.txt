configure_file(notch_hyperJ2_verify.yaml.in notch_hyperJ2_verify.yaml @ONLY)
configure_file(notch_hyperJ2_model_form_error_verify.yaml.in notch_hyperJ2_model_form_error_verify.yaml @ONLY)
configure_file(notch_isotropic_elastic_model_form_error_verify.yaml.in notch_isotropic_elastic_model_form_error_verify.yaml @ONLY)

add_test(
  NAME notch_hyperJ2_verify
  COMMAND "${CMAKE_BINARY_DIR}/src/verify" "${CMAKE_BINARY_DIR}/test/verify/notch_hyperJ2_verify.yaml")
add_test(
  NAME notch_hyperJ2_model_form_error_verify
  COMMAND "${CMAKE_BINARY_DIR}/src/model_form_error_verify" "${CMAKE_BINARY_DIR}/test/verify/notch_hyperJ2_model_form_error_verify.yaml")
add_test(
  NAME notch_isotropic_elastic_model_form_error_verify
  COMMAND "${CMAKE_BINARY_DIR}/src/model_form_error_verify" "${CMAKE_BINARY_DIR}/test/verify/notch_isotropic_elastic_model_form_error_verify.yaml")
