#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <PCU.h>
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "state.hpp"

using namespace calibr8;

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("dirichlet bcs");
  p.sublist("traction bcs");
  p.sublist("linear algebra");
  p.sublist("quantity of interest");
  p.sublist("regression");
  return p;
}

class Solver {
  public:
    Solver(std::string const& input_file);
    void solve();
    void write_at_end();
    void write_at_step(int step, bool has_adjoint);
  private:
    std::string base_name();
    void write_pvd();
    void write_native();
  private:
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Primal> m_primal;
    bool m_eval_qoi = false;
    bool m_eval_regression = false;
};

Solver::Solver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_params->validateParameters(get_valid_params(), 0);
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  if (m_state->qoi != Teuchos::null) m_eval_qoi = true;
  if (m_params->isSublist("regression")) m_eval_regression = true;
}

std::string Solver::base_name() {
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const name = problem_params.get<std::string>("name");
  std::string const base = name + "_forward";
  return base;
}

static void mkdir(const char* path) {
  mode_t const mode = S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH;
  int err;
  errno = 0;
  err = mkdir(path, mode);
  if (err != 0 && errno != EEXIST) {
    fail("could not create directory \"%s\"\n", path);
  }
}

static std::vector<std::string> get_names(
    RCP<GlobalResidual<double>> global,
    RCP<LocalResidual<double>> local,
    bool has_adjoint) {
  std::vector<std::string> names;
  for (int i = 0; i < global->num_residuals(); ++i) {
    names.push_back(global->resid_name(i));
  }
  for (int i = 0; i < local->num_residuals(); ++i) {
    names.push_back(local->resid_name(i));
  }
  if (has_adjoint) {
    for (int i = 0; i < global->num_residuals(); ++i) {
      names.push_back("adjoint_" + global->resid_name(i));
    }
    for (int i = 0; i < local->num_residuals(); ++i) {
      names.push_back("adjoint_" + local->resid_name(i));
    }
  }
  return names;
}

void Solver::write_at_step(int step, bool has_adjoint) {
  std::string out_dir = base_name() + "_viz";
  mkdir(out_dir.c_str());
  std::string const out_name = base_name() + "_viz/out_" + std::to_string(step);
  RCP<GlobalResidual<double>> global = m_state->residuals->global;
  RCP<LocalResidual<double>> local = m_state->residuals->local;
  apf::Mesh* mesh = m_state->disc->apf_mesh();
  std::vector<std::string> names = get_names(global, local, has_adjoint);
  for (std::string const& name : names) {
    std::string const name_step = name + "_" + std::to_string(step);
    apf::Field* f = mesh->findField(name_step.c_str());
    apf::renameField(f, name.c_str());
  }
  // some mechanics specific stuff below
  apf::Field* sigma = eval_cauchy(m_state, step);
  names.push_back("sigma");
  apf::writeVtkFiles(out_name.c_str(), mesh, names);
  names.pop_back();
  apf::destroyField(sigma);
  for (std::string const& name : names) {
    std::string const name_step = name + "_" + std::to_string(step);
    apf::Field* f = mesh->findField(name.c_str());
    apf::renameField(f, name_step.c_str());
  }
}

void Solver::write_pvd() {
  if (PCU_Comm_Self()) return;
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const pvd_name = base_name() + "_viz/out.pvd";
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = dt;
  std::fstream pvdf;
  pvdf.open(pvd_name, std::ios::out);
  pvdf << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
  pvdf << "  <Collection>" << std::endl;
  for (int step = 1; step <= nsteps; ++step) {
    std::string const out_name = "out_" + std::to_string(step);
    std::string const vtu = out_name + "/" + out_name;
    pvdf << "    <DataSet timestep=\"" << t << "\" group=\"\" ";
    pvdf << "part=\"0\" file=\"" << vtu;
    pvdf << ".pvtu\"/>" << std::endl;
    t += dt;
  }
  pvdf << "  </Collection>" << std::endl;
  pvdf << "</VTKFile>" << std::endl;
  pvdf.close();
}

void Solver::write_native() {
  std::string const smb_name = base_name() + "/";
  m_state->disc->apf_mesh()->writeNative(smb_name.c_str());
}

void Solver::write_at_end() {
  write_pvd();
  write_native();
}

static void eval_regression(double J, RCP<ParameterList> params) {
  ParameterList regression_params = params->sublist("regression", true);
  double const J_expected = regression_params.get<double>("QoI");
  double const tol = regression_params.get<double>("relative error tol");
  double const err = std::abs((J-J_expected)/J_expected);
  std::cout << std::scientific << std::setprecision(17);
  std::cout << "------ regression summary -----\n";
  std::cout << "J computed: " << J << "\n";
  std::cout << "J expected: " << J_expected << "\n";
  std::cout << "relative error: " << err << "\n";
  if (err < tol) std::cout << " PASS\n";
  else {
    std::cout << " FAIL\n";
    abort();
  }
  std::cout << "-------------------------------\n";
}

void Solver::solve() {
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const name = problem_params.get<std::string>("name");
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0.;
  double J = 0.;
  //write_at_step(0, false);
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    if (m_eval_qoi) J += eval_qoi(m_state, m_state->disc, step);
    write_at_step(step, false);
  }
  write_at_end();
  if (m_eval_qoi) {
    J = PCU_Add_Double(J);
    print("J: %.16e\n", J);
  }
  if (m_eval_regression) {
    eval_regression(J, m_params);
  }
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    std::string const yaml_input = argv[1];
    Solver solver(yaml_input);
    solver.solve();
  }
  finalize();
}
