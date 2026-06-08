# Tests that exercise the Python calibration drivers in python/calibr8.
#
# These run through the uv virtual environment created per python/install.txt
# (uv pip install -e ".[test]"). The venv carries calibr8 plus numpy/scipy/yaml
# and pytest, so the tests invoke it directly. capp already places the FE
# binaries (primal, objective) and mpirun on PATH, so no PATH wiring is needed
# here. If the venv is missing, skip registration with a warning rather than
# failing configuration.

set(CALIBR8_PYTHON "${CMAKE_SOURCE_DIR}/python/.venv/bin/python"
    CACHE FILEPATH "Python interpreter (calibr8 venv) used to run the driver tests")

if(NOT EXISTS "${CALIBR8_PYTHON}")
  message(WARNING
    "calibr8 python venv not found at ${CALIBR8_PYTHON}; "
    "skipping python driver tests (see python/README.md to create it)")
  return()
endif()

# Fast, pure-Python unit tests for the optimizer support code (no FE solve).
add_test(
  NAME python_unit
  COMMAND "${CALIBR8_PYTHON}" -m pytest -q -p no:cacheprovider
          "${CMAKE_CURRENT_SOURCE_DIR}/unit"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# End-to-end calibration recovery on the 2D plane-stress notch:
#   1. primal writes the synthetic displacement mesh + per-step load (load.dat)
#   2. python_inverse recovers Y/S/D from perturbed starts via the calibration
#      objective (full-field displacement + axial-load mismatch), once per
#      SciPy optimizer
#   3. a checker confirms the recovered parameters match the truth values
# Each optimizer runs in its own working directory so scratch files do not
# collide; both read the shared synthetic data by absolute path.
configure_file(notch2D_small_J2_FEMU_synthetic.yaml.in notch2D_small_J2_FEMU_synthetic.yaml @ONLY)
configure_file(notch2D_small_J2_FEMU.yaml.in notch2D_small_J2_FEMU.yaml @ONLY)

file(MAKE_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}/lbfgsb"
  "${CMAKE_CURRENT_BINARY_DIR}/trust_region")

add_test(
  NAME py_femu_synthetic
  COMMAND "${CMAKE_BINARY_DIR}/src/primal" "${CMAKE_CURRENT_BINARY_DIR}/notch2D_small_J2_FEMU_synthetic.yaml"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_test(
  NAME py_femu_lbfgsb
  COMMAND "${CALIBR8_PYTHON}" -m calibr8.bin.inverse
          "${CMAKE_CURRENT_BINARY_DIR}/notch2D_small_J2_FEMU.yaml" -n 1
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lbfgsb")

add_test(
  NAME py_femu_trust_region
  COMMAND "${CALIBR8_PYTHON}" -m calibr8.bin.inverse
          "${CMAKE_CURRENT_BINARY_DIR}/notch2D_small_J2_FEMU.yaml" -n 1 --trust_region
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/trust_region")

add_test(
  NAME py_femu_lbfgsb_check
  COMMAND "${CALIBR8_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/check_recovered_params.py"
          calibrated_params.txt Y=2.0 S=10.0 D=2.0 --rtol 1e-3
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lbfgsb")

add_test(
  NAME py_femu_trust_region_check
  COMMAND "${CALIBR8_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/check_recovered_params.py"
          calibrated_params.txt Y=2.0 S=10.0 D=2.0 --rtol 1e-3
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/trust_region")

set_tests_properties(py_femu_lbfgsb py_femu_trust_region
  PROPERTIES DEPENDS py_femu_synthetic)
set_tests_properties(py_femu_lbfgsb_check PROPERTIES DEPENDS py_femu_lbfgsb)
set_tests_properties(py_femu_trust_region_check PROPERTIES DEPENDS py_femu_trust_region)
