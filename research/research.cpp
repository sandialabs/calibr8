#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "bcs.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "physics.hpp"
#include "qoi.hpp"
#include "weights.hpp"

using namespace calibr8;

struct Fields {
  apf::Field* u[NUM_SPACE] = {nullptr};
  apf::Field* z[NUM_SPACE] = {nullptr};
  apf::Field* uH_h = nullptr;
  apf::Field* uh_minus_uH_h = nullptr;
  apf::Field* E_L = nullptr;
  void destroy();
};

void Fields::destroy() {
  for (int space = 0; space < NUM_SPACE; ++space) {
    if (u[space]) apf::destroyField(u[space]);
    if (z[space]) apf::destroyField(z[space]);
    u[space] = nullptr;
    z[space] = nullptr;
  }
  if (uH_h) apf::destroyField(uH_h);
  if (uh_minus_uH_h) apf::destroyField(uh_minus_uH_h);
  if (E_L) apf::destroyField(E_L);
  uH_h = nullptr;
  uh_minus_uH_h = nullptr;
  E_L = nullptr;
}

class Driver {
  public:
    Driver(std::string const& input_file);
    ~Driver();
    void drive();
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
    Fields m_fields;
  private:
    void solve_primal(int space);
    void prolong_u_to_fine();
    void difference_u();
    void solve_adjoint(int space);
    void compute_linearization_error();
};

static void verify_adjoint(apf::Field* u) {
  if (!u) {
    throw std::runtime_error("solve adjoint - u doesn't exist");
  }
}

static void verify_linearization_error(
    apf::Field* uH_h,
    apf::Field* u_diff) {
  if (!uH_h) {
    throw std::runtime_error("linearization error - uH_h doesn't exist");
  }
  if (!u_diff) {
    throw std::runtime_error("linearization error - u_diff doesn't exist");
  }
}

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  ParameterList const resid_params = m_params->sublist("residual");
  ParameterList const disc_params = m_params->sublist("discretization");
  ParameterList const qoi_params = m_params->sublist("quantity of interest");
  m_disc = rcp(new Disc(disc_params));
  m_residual = create_residual<double>(resid_params, m_disc->num_dims());
  m_jacobian = create_residual<FADT>(resid_params, m_disc->num_dims());
  m_qoi = create_QoI<double>(qoi_params);
  m_qoi_deriv = create_QoI<FADT>(qoi_params);
}

Driver::~Driver() {
  m_disc->destroy_data();
}

static void print_banner(std::string const& banner) {
  print("****************");
  print("%s", banner.c_str());
  print("****************");
}

void Driver::solve_primal(int space) {
  std::string const banner = "primal " + m_disc->space_name(space);
  print_banner(banner);
  m_fields.u[space] = calibr8::solve_primal(
      space,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
  double const J = calibr8::compute_qoi(
      space,
      m_params,
      m_disc,
      m_residual,
      m_qoi,
      m_fields.u[space]);
  print("");
}

void Driver::prolong_u_to_fine() {
  print("* prolonging uH onto h");
  m_fields.uH_h = project(m_disc, m_fields.u[COARSE], "uH_h");
  print("");
}

void Driver::difference_u() {
  print("* computing uh-uH_h");
  m_fields.uh_minus_uH_h = subtract(
      m_disc, m_fields.u[FINE], m_fields.uH_h, "uh-uH_h");
  print("");
}

void Driver::solve_adjoint(int space) {
  std::string const banner = "adjoint " + m_disc->space_name(space);
  print_banner(banner);
  apf::Field* u = (space == COARSE) ? m_fields.u[space] : m_fields.uH_h;
  verify_adjoint(u);
  m_fields.z[space] = calibr8::solve_adjoint(
      space,
      m_params,
      m_disc,
      m_jacobian,
      m_qoi_deriv,
      u);
  print("");
}

void Driver::compute_linearization_error() {
  print_banner("linearization error");
  apf::Field* uH_h = m_fields.uH_h;
  apf::Field* u_diff = m_fields.uh_minus_uH_h;
  verify_linearization_error(uH_h, u_diff);
  LE const data = calibr8::compute_linearization_error(
      m_params,
      m_disc,
      m_residual,
      m_jacobian,
      uH_h,
      u_diff);
  m_fields.E_L = data.field;
  print("");
}

void Driver::drive() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
  this->solve_primal(COARSE);
  this->solve_primal(FINE);
  this->prolong_u_to_fine();
  this->difference_u();
  this->solve_adjoint(COARSE);
  this->solve_adjoint(FINE);
  this->compute_linearization_error();

  do_stuff(
      m_disc,
      m_residual,
      m_fields.z[FINE],
      m_fields.uH_h);





  // ---debug below---
  apf::writeVtkFiles("debug", m_disc->apf_mesh());

}

int main(int argc, char** argv) {
  initialize();
  ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    Driver driver(argv[1]);
    driver.drive();
  }
  finalize();
}
