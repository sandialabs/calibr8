set(CALIBR8_SOURCES
  adjoint.cpp
  adjoint_objective.cpp
  avg_disp.cpp
  avg_disp_subdomain.cpp
  avg_stress.cpp
  calibration.cpp
  control.cpp
  dbcs.cpp
  disc.cpp
  error_weight.cpp
  evaluations.cpp
  femu_objective.cpp
  fields.cpp
  global_residual.cpp
  J2.cpp
  J2_plane_strain.cpp
  J2_small_strain.cpp
  load_mismatch.cpp
  linear_alg.cpp
  linear_solve.cpp
  local_residual.cpp
  mechanics.cpp
  mesh_size.cpp
  nested.cpp
  objective.cpp
  primal.cpp
  qoi.cpp
  state.cpp
  synthetic.cpp
  surface_mismatch.cpp
  tbcs.cpp
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

add_executable(perturber perturber.cpp)
target_link_libraries(perturber PRIVATE calibr8)
set_code_standards(perturber)

add_executable(verify main_verify.cpp)
target_link_libraries(verify PRIVATE calibr8)
set_code_standards(verify)
