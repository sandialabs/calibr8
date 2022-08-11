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

struct History {
  std::vector<double> JH;
  std::vector<double> Jh;
  std::vector<double> Rh_uH_h;
  std::vector<double> E_L;
};

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
    History m_history;
};

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

void Driver::drive() {

  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);

  print_banner("primal H");
  m_fields.u[COARSE] = solve_primal(
      COARSE,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
  double const JH = compute_qoi(
      COARSE,
      m_params,
      m_disc,
      m_residual,
      m_qoi,
      m_fields.u[COARSE]);
  m_history.JH.push_back(JH);
  print("");

  print_banner("primal h");
  m_fields.u[FINE] = solve_primal(
      FINE,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
  double const Jh = compute_qoi(
      FINE,
      m_params,
      m_disc,
      m_residual,
      m_qoi,
      m_fields.u[FINE]);
  m_history.Jh.push_back(Jh);
  print("");

  print("* projecting uH onto h");
  m_fields.uH_h = project(
      m_disc,
      m_fields.u[COARSE],
      "uH_h");

  print("* computing uh-uH_h");
  m_fields.uh_minus_uH_h = subtract(
      m_disc,
      m_fields.u[FINE],
      m_fields.uH_h,
      "uh-uH_h");
  print("");

  print_banner("linearization error");
  LE const data = compute_linearization_error(
      m_params,
      m_disc,
      m_residual,
      m_jacobian,
      m_fields.uH_h,
      m_fields.uh_minus_uH_h);
  m_fields.E_L = data.field;
  m_history.E_L.push_back(data.E_L);
  m_history.Rh_uH_h.push_back(data.Rh_uH_h);
  print("");

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
