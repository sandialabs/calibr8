#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <lionPrint.h>
#include <PCU.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "state.hpp"
#include "virtual_power.hpp"

using namespace calibr8;

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("linear algebra");
  p.sublist("quantity of interest");
  p.sublist("virtual fields");
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
    RCP<VirtualPower> m_virtual_power;
    bool m_eval_qoi = false;
    bool m_eval_regression = false;
    bool m_write_measured = true;
};

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

static Array1D<std::string> get_names(
    RCP<GlobalResidual<double>> global,
    RCP<LocalResidual<double>> local,
    bool has_adjoint) {
  Array1D<std::string> names;
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
  int const model_form = m_state->model_form;
  RCP<LocalResidual<double>> local = m_state->residuals->local[model_form];
  apf::Mesh* mesh = m_state->disc->apf_mesh();
  Array1D<std::string> names = get_names(global, local, has_adjoint);
  for (std::string const& name : names) {
    std::string const name_step = name + "_" + std::to_string(step);
    apf::Field* f = mesh->findField(name_step.c_str());
    apf::renameField(f, name.c_str());
  }
  // some mechanics specific stuff below
  apf::Field* sigma = eval_cauchy(m_state, step);
  names.push_back("sigma");
  if (m_write_measured) {
    std::string const meas_name = "measured";
    std::string const name_step = meas_name + "_" + std::to_string(step);
    apf::Field* f = mesh->findField(name_step.c_str());
    apf::renameField(f, meas_name.c_str());
    names.push_back(meas_name);
  }
  apf::writeVtkFiles(out_name.c_str(), mesh, names);
  names.pop_back();
  apf::destroyField(sigma);
  if (m_write_measured) {
    std::string const meas_name = "measured";
    std::string const name_step = meas_name + "_" + std::to_string(step);
    apf::Field* f = mesh->findField(meas_name.c_str());
    apf::renameField(f, name_step.c_str());
    names.pop_back();
  }
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
  int const nsteps = m_state->disc->num_time_steps();
  double t = 0.;
  std::fstream pvdf;
  pvdf.open(pvd_name, std::ios::out);
  pvdf << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
  pvdf << "  <Collection>" << std::endl;
  for (int step = 1; step <= nsteps; ++step) {
    t = m_state->disc->time(step);
    std::string const out_name = "out_" + std::to_string(step);
    std::string const vtu = out_name + "/" + out_name;
    pvdf << "    <DataSet timestep=\"" << t << "\" group=\"\" ";
    pvdf << "part=\"0\" file=\"" << vtu;
    pvdf << ".pvtu\"/>" << std::endl;
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

Solver::Solver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_params->validateParameters(get_valid_params(), 0);
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  m_virtual_power = rcp(new VirtualPower(m_params, m_state, m_state->disc));
}

void Solver::solve() {
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const name = problem_params.get<std::string>("name");
  int const nsteps = m_state->disc->num_time_steps();
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    J += m_virtual_power->compute_at_step(step);
    write_at_step(step, false);
  }
  write_at_end();
  J = PCU_Add_Double(J);
  //print("Sum of Virtual Internal Power: %.16e\n", J);
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    std::string const yaml_input = argv[1];
    Solver solver(yaml_input);
    solver.solve();
  }
  finalize();
}
