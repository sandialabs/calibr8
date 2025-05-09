set(CALIBR8_SOURCES
  adjoint.cpp
  adjoint_objective.cpp
  adjoint_sens_vfm_objective.cpp
  avg_disp.cpp
  avg_stress.cpp
  calibration.cpp
  control.cpp
  cspr.cpp
  dbcs.cpp
  disc.cpp
  disp_comp.cpp
  elastic.cpp
  equilibrium_gap.cpp
  error_weight.cpp
  evaluations.cpp
  femu_objective.cpp
  forward_sens_vfm_objective.cpp
  fd_vfm_objective.cpp
  fields.cpp
  global_residual.cpp
  isotropic_elastic.cpp
  hybrid_hyper_J2_plane_stress.cpp
  hyper_J2.cpp
  hyper_J2_plane_strain.cpp
  hyper_J2_plane_stress.cpp
  hypo_barlat.cpp
  hypo_hill.cpp
  hypo_hill_plane_strain.cpp
  hypo_hill_plane_stress.cpp
  hypo_hosford.cpp
  load_mismatch.cpp
  linear_alg.cpp
  linear_solve.cpp
  local_residual.cpp
  mechanics.cpp
  mechanics_plane_stress.cpp
  mesh_size.cpp
  nested.cpp
  NN_old.cpp
  NN.cpp
  ICNN.cpp
  normal_traction.cpp
  objective.cpp
  point_wise.cpp
  primal.cpp
  qoi.cpp
  reaction.cpp
  reaction_mismatch.cpp
  small_hill.cpp
  small_hill_plane_strain.cpp
  small_hill_plane_stress.cpp
  small_hosford.cpp
  small_J2.cpp
  state.cpp
  synthetic.cpp
  surface_mismatch.cpp
  tbcs.cpp
  virtual_power.cpp
  weight.cpp
)

add_library(calibr8 ${CALIBR8_SOURCES})

target_include_directories(calibr8 PUBLIC
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(calibr8 PUBLIC
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${Trilinos_EXTRA_LD_FLAGS}
  SCOREC::core
  MPI::MPI_CXX)

set_code_standards(calibr8)
install_and_export(calibr8)

add_executable(primal main_primal.cpp)
target_link_libraries(primal PRIVATE calibr8)
set_code_standards(primal)
install(TARGETS primal)

add_executable(inverse main_inverse.cpp)
target_link_libraries(inverse PRIVATE calibr8)
set_code_standards(inverse)
install(TARGETS inverse)

add_executable(error main_error.cpp)
target_link_libraries(error PRIVATE calibr8)
set_code_standards(error)
install(TARGETS error)

add_executable(spr_error main_spr_error.cpp)
target_link_libraries(spr_error PRIVATE calibr8)
set_code_standards(spr_error)
install(TARGETS spr_error)

add_executable(model_form_exact_error main_model_form_exact_error.cpp)
target_link_libraries(model_form_exact_error PRIVATE calibr8)
set_code_standards(model_form_exact_error)
install(TARGETS model_form_exact_error)

add_executable(model_form_error main_model_form_error.cpp)
target_link_libraries(model_form_error PRIVATE calibr8)
set_code_standards(model_form_error)
install(TARGETS model_form_error)

add_executable(model_form_error_verify main_model_form_error_verify.cpp)
target_link_libraries(model_form_error_verify PRIVATE calibr8)
set_code_standards(model_form_error_verify)
install(TARGETS model_form_error_verify)

add_executable(add_node_set_fields add_node_set_fields.cpp)
target_link_libraries(add_node_set_fields PRIVATE calibr8)
set_code_standards(add_node_set_fields)
install(TARGETS add_node_set_fields)

add_executable(moving_least_squares moving_least_squares.cpp)
target_link_libraries(moving_least_squares PRIVATE calibr8)
set_code_standards(moving_least_squares)
install(TARGETS moving_least_squares)

add_executable(moving_least_squares_two_meshes moving_least_squares_two_meshes.cpp)
target_link_libraries(moving_least_squares_two_meshes PRIVATE calibr8)
set_code_standards(moving_least_squares_two_meshes)
install(TARGETS moving_least_squares_two_meshes)

add_executable(objective main_objective.cpp)
target_link_libraries(objective PRIVATE calibr8)
set_code_standards(objective)
install(TARGETS objective)

add_executable(perturber perturber.cpp)
target_link_libraries(perturber PRIVATE calibr8)
set_code_standards(perturber)
install(TARGETS perturber)

add_executable(transfer_surface_data transfer_surface_data.cpp)
target_link_libraries(transfer_surface_data PRIVATE calibr8)
set_code_standards(transfer_surface_data)
install(TARGETS transfer_surface_data)

add_executable(verify main_verify.cpp)
target_link_libraries(verify PRIVATE calibr8)
set_code_standards(verify)
install(TARGETS verify)

add_executable(exact_error main_exact_error.cpp)
target_link_libraries(exact_error PRIVATE calibr8)
set_code_standards(exact_error)
install(TARGETS exact_error)

# TODO: maybe kill this if it doesn't work
add_executable(virtual_fields_method main_virtual_fields_method.cpp)
target_link_libraries(virtual_fields_method PRIVATE calibr8)
install(TARGETS virtual_fields_method)
set_code_standards(virtual_fields_method)

add_executable(evaluate_multiple_qois_and_gradients main_evaluate_multiple_qois_and_gradients.cpp)
target_link_libraries(evaluate_multiple_qois_and_gradients PRIVATE calibr8)
install(TARGETS evaluate_multiple_qois_and_gradients)
set_code_standards(evaluate_multiple_qois_and_gradients)

add_executable(primal_multiple_problems main_primal_multiple_problems.cpp)
target_link_libraries(primal_multiple_problems PRIVATE calibr8)
install(TARGETS primal_multiple_problems)
set_code_standards(primal_multiple_problems)
