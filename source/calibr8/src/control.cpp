#include <cstdarg>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <PCU.h>
#include <RTC_FunctionRTC.hh>
#include "control.hpp"

namespace calibr8 {

static bool is_calibr8_initd = false;
static bool is_mpi_initd = false;
static bool is_kokkos_initd = false;
static bool is_pcu_initd = false;

static PG_RuntimeCompiler::Function evaluator(5);

static void call_expr_init() {
  evaluator.addVar("double", "x");
  evaluator.addVar("double", "y");
  evaluator.addVar("double", "z");
  evaluator.addVar("double", "t");
  evaluator.addVar("double", "val");
}

static void call_mpi_init() {
  MPI_Init(0, 0);
  is_mpi_initd = true;
}

static void call_pcu_init() {
  PCU_Comm_Init();
  is_pcu_initd = true;
}

static void call_kokkos_init() {
  Kokkos::initialize();
  is_kokkos_initd = true;
}

void initialize() {
  if (is_calibr8_initd) return;
  call_mpi_init();
  call_kokkos_init();
  call_pcu_init();
  call_expr_init();
  is_calibr8_initd = true;
}

static void call_mpi_free() {
  MPI_Finalize();
  is_mpi_initd = false;
}

static void call_kokkos_free() {
  Kokkos::finalize();
  is_kokkos_initd = false;
}

static void call_pcu_free() {
  PCU_Comm_Free();
  is_pcu_initd = false;
}

static void assert_initd() {
  if (!is_calibr8_initd) {
    fail("calibr8::initialize() not called");
  }
}

void finalize() {
  assert_initd();
  if (is_pcu_initd) call_pcu_free();
  if (is_kokkos_initd) call_kokkos_free();
  if (is_mpi_initd) call_mpi_free();
  is_calibr8_initd = false;
}

void print(const char* message, ...) {
  assert_initd();
  if (PCU_Comm_Self()) return void();
  va_list ap;
  va_start(ap, message);
  vfprintf(stdout, message, ap);
  va_end(ap);
  printf("\n");
}

void fail(const char* why, ...) {
  assert_initd();
  va_list ap;
  va_start(ap, why);
  vfprintf(stderr, why, ap);
  va_end(ap);
  printf("\n");
  abort();
}

void assert_fail(const char* why, ...) {
  assert_initd();
  fprintf(stderr, "%s", why);
  abort();
}

double eval(
    std::string const& v,
    double x,
    double y,
    double z,
    double t) {
  assert_initd();
  std::string value = "val=" + v;
  evaluator.addBody(value);
  evaluator.varValueFill(0, x);
  evaluator.varValueFill(1, y);
  evaluator.varValueFill(2, z);
  evaluator.varValueFill(3, t);
  evaluator.varValueFill(4, 0.0);
  evaluator.execute();
  return evaluator.getValueOfVar("val");
}

double time() {
  assert_initd();
  return PCU_Time();
}

}
