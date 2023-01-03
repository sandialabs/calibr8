set(CALIBR8_SOURCES
  adjoint.cpp
  adjoint_objective.cpp
  avg_disp.cpp
  avg_stress.cpp
  calibration.cpp
  control.cpp
  dbcs.cpp
  disc.cpp
  disp_comp.cpp
  elastic.cpp
  error_weight.cpp
  evaluations.cpp
  femu_objective.cpp
  fd_vfm_objective.cpp
  fields.cpp
  global_residual.cpp
  isotropic_elastic.cpp
  hyper_J2.cpp
  hyper_J2_plane_strain.cpp
  hyper_J2_plane_stress.cpp
  hypo_hill.cpp
  hypo_hill_plane_strain.cpp
  hypo_hill_plane_stress.cpp
  load_mismatch.cpp
  linear_alg.cpp
  linear_solve.cpp
  local_residual.cpp
  mechanics.cpp
  mechanics_plane_stress.cpp
  mesh_size.cpp
  nested.cpp
  normal_traction.cpp
  objective.cpp
  point_wise.cpp
  primal.cpp
  qoi.cpp
  small_hill.cpp
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

add_executable(primal main_primal.cpp)
target_link_libraries(primal PRIVATE calibr8)
set_code_standards(primal)

add_executable(inverse main_inverse.cpp)
target_link_libraries(inverse PRIVATE calibr8)
set_code_standards(inverse)

add_executable(error main_error.cpp)
target_link_libraries(error PRIVATE calibr8)
set_code_standards(error)

add_executable(model_form_exact_error main_model_form_exact_error.cpp)
target_link_libraries(model_form_exact_error PRIVATE calibr8)
set_code_standards(model_form_exact_error)

add_executable(model_form_error main_model_form_error.cpp)
target_link_libraries(model_form_error PRIVATE calibr8)
set_code_standards(model_form_error)

add_executable(model_form_error_verify main_model_form_error_verify.cpp)
target_link_libraries(model_form_error_verify PRIVATE calibr8)
set_code_standards(model_form_error_verify)

add_executable(perturber perturber.cpp)
target_link_libraries(perturber PRIVATE calibr8)
set_code_standards(perturber)

add_executable(verify main_verify.cpp)
target_link_libraries(verify PRIVATE calibr8)
set_code_standards(verify)

add_executable(exact_error main_exact_error.cpp)
target_link_libraries(exact_error PRIVATE calibr8)
set_code_standards(exact_error)

add_executable(virtual_fields_method main_virtual_fields_method.cpp)
target_link_libraries(virtual_fields_method PRIVATE calibr8)
set_code_standards(virtual_fields_method)
