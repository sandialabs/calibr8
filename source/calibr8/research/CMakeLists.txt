set(RESEARCH_SOURCES
  adapt.cpp
  bcs.cpp
  control.cpp
  cspr.cpp
  disc.cpp
  error.cpp
  error_adjoint.cpp
  linalg.cpp
  nlpoisson.cpp
  residual.cpp
  physics.cpp
  qoi.cpp
  weight.cpp
  research.cpp
)

add_executable(research ${RESEARCH_SOURCES})

target_include_directories(research PUBLIC
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(research PUBLIC
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${Trilinos_EXTRA_LD_FLAGS}
  SCOREC::core
  MPI::MPI_CXX)

set_code_standards(research)

install(TARGETS research)
