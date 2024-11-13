configure_file(notch2D_small_J2_plane_stress.yaml.in notch2D_small_J2_plane_stress.yaml @ONLY)
configure_file(vfm_forward_sens_notch2D_small_J2_plane_stress.yaml.in vfm_forward_sens_notch2D_small_J2_plane_stress.yaml @ONLY)

add_test(
  NAME vfm_test_primal_run
  COMMAND "${CMAKE_BINARY_DIR}/src/primal" "${CMAKE_BINARY_DIR}/test/vfm/notch2D_small_J2_plane_stress.yaml")

add_test(
  NAME vfm_test_forward_sens
  COMMAND "${CMAKE_BINARY_DIR}/src/inverse" "${CMAKE_BINARY_DIR}/test/vfm/vfm_forward_sens_notch2D_small_J2_plane_stress.yaml")

set_tests_properties(vfm_test_forward_sens PROPERTIES DEPENDS vfm_test_primal_run)
