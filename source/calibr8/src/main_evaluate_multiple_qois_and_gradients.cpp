#include <PCU.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "qoi.hpp"
#include "state.hpp"

using namespace calibr8;

static ParameterList get_valid_params()
{
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("dirichlet bcs");
  p.sublist("traction bcs");
  p.sublist("linear algebra");
  p.sublist("quantities of interest");
  return p;
}

class MultiQoI
{
  public:
    MultiQoI(RCP<ParameterList> params);
    ~MultiQoI() {};
    void evaluate();
    void write_output();
    void cleanup();
  private:
    void setup_opt_params(ParameterList const& inverse_params);
    void create_qois();
    void set_qois(int const q);
    Array1D<RCP<QoI<double>>> m_qois;
    Array1D<RCP<QoI<FADT>>> m_d_qois;
    Array1D<EVector> m_qoi_values;
    Array1D<EMatrix> m_qoi_gradients;
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Primal> m_primal;
    RCP<Adjoint> m_adjoint;
    int m_num_qois = 0;
    int const m_model_form = 0;
    int m_num_opt_params;
    bool m_compute_qoi_gradients = false;
};

MultiQoI::MultiQoI(RCP<ParameterList> params)
{
  m_params = params;
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  if (m_params->isSublist("inverse")) {
    m_compute_qoi_gradients = true;
    m_adjoint = rcp(new Adjoint(m_params, m_state, m_state->disc));
    setup_opt_params(m_params->sublist("inverse", true));
  }
  create_qois();
}

void MultiQoI::create_qois()
{
  auto qois_list = m_params->sublist("quantities of interest");
  for (auto qoi_entry : qois_list) {
    auto qoi_list = Teuchos::getValue<ParameterList>(qoi_entry.second);
    auto qoi = create_qoi<double>(qoi_list);
    auto d_qoi = create_qoi<FADT>(qoi_list);
    m_qois.push_back(qoi);
    m_d_qois.push_back(d_qoi);
  }
  m_num_qois = m_qois.size();
  int const nsteps = m_state->disc->num_time_steps();

  m_qoi_values.resize(m_num_qois);
  for (int q = 0; q < m_num_qois; ++q) {
    m_qoi_values[q].resize(nsteps);
  }

  if (m_compute_qoi_gradients) {
    m_qoi_gradients.resize(m_num_qois);
    for (int q = 0; q < m_num_qois; ++q) {
      m_qoi_gradients[q].resize(nsteps, m_num_opt_params);
    }
  }
}

void MultiQoI::set_qois(int const q)
{
  m_state->qoi = m_qois[q];
  m_state->d_qoi = m_d_qois[q];
}

void MultiQoI::write_output()
{
  if (PCU_Comm_Self() == 0) {
    for (int q = 0; q < m_num_qois; ++q) {
      auto const qoi_index_str = std::to_string(q);

      Eigen::IOFormat out_format(Eigen::FullPrecision, Eigen::DontAlignCols);

      std::ofstream obj_file;
      std::string const obj_filename = "objective_value_" + qoi_index_str + ".txt";
      obj_file.open(obj_filename);
      std::stringstream qoi_values_ss;
      qoi_values_ss << m_qoi_values[q].format(out_format);
      obj_file << qoi_values_ss.str();
      obj_file.close();

      if (m_compute_qoi_gradients) {
        std::ofstream grad_file;
        std::string const grad_filename = "objective_gradient_" + qoi_index_str + ".txt";
        grad_file.open(grad_filename);
        std::stringstream qoi_gradients_ss;
        qoi_gradients_ss << m_qoi_gradients[q].format(out_format);
        grad_file << qoi_gradients_ss.str();
        grad_file.close();
      }
    }
  }
}

void MultiQoI::cleanup()
{
  m_state->disc->destroy_primal();
}

void MultiQoI::setup_opt_params(ParameterList const& inverse_params)
{
  Array1D<std::string> const& elem_set_names =
      m_state->residuals->local[m_model_form]->elem_set_names();
  Array1D<std::string> const& param_names =
      m_state->residuals->local[m_model_form]->param_names();

  ParameterList const& all_material_params =
      inverse_params.sublist("materials");

  int const num_elem_sets = elem_set_names.size();
  int const num_params_total = param_names.size();

  Array2D<int> active_indices(num_elem_sets);
  Array2D<int> grad_indices(num_elem_sets);
  m_num_opt_params = 0;

  for (int es = 0; es < num_elem_sets; ++es) {
    ParameterList material_params =
        all_material_params.sublist(elem_set_names[es]);
    for (int i = 0; i < num_params_total; ++i) {
      std::string param_name = param_names[i];
      if (material_params.isParameter(param_name)) {
        active_indices[es].push_back(i);
        grad_indices[es].push_back(m_num_opt_params);
        m_num_opt_params++;
      }
    }
  }

  // Not able to have dfad params with multiple materials yet
  int const num_dfad_params = m_state->residuals->local[m_model_form]->num_dfad_params();
  if (num_dfad_params > 0) {
      ALWAYS_ASSERT(num_elem_sets == 1);
  }
  m_num_opt_params += num_dfad_params;

  m_state->residuals->local[m_model_form]->set_active_and_grad_indices(
      active_indices, grad_indices);
  m_state->d_residuals->local[m_model_form]->set_active_and_grad_indices(
      active_indices, grad_indices);
}

void MultiQoI::evaluate()
{
  double J_at_step = 0.;
  int const nsteps = m_state->disc->num_time_steps();
  for (int step = 1; step <= nsteps; ++step) {
    J_at_step = 0.;
    m_primal->solve_at_step(step);
    for (int q = 0; q < m_num_qois; ++q) {
      set_qois(q);
      J_at_step = eval_qoi(m_state, m_state->disc, step);
      J_at_step = PCU_Add_Double(J_at_step);
      m_qoi_values[q](step - 1) = J_at_step;
    }
  }

  if (m_compute_qoi_gradients) {
    Array1D<double> grad_at_step(m_num_opt_params);
    for (int q = 0; q < m_num_qois; ++q) {
      set_qois(q);
      m_state->disc->create_adjoint(m_state->residuals, nsteps);
      for (int step = nsteps; step > 0; --step) {
        m_adjoint->solve_at_step(step);
        grad_at_step = eval_qoi_gradient(m_state, step);
        PCU_Add_Doubles(grad_at_step.data(), m_num_opt_params);
        for (int p = 0; p < m_num_opt_params; ++p) {
          m_qoi_gradients[q](step - 1, p) = grad_at_step[p];
        }
      }
      m_state->disc->destroy_adjoint();
    }
  }
}

int main(int argc, char** argv)
{
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    std::string const yaml_input = argv[1];

    auto params = rcp(new ParameterList);
    Teuchos::updateParametersFromYamlFile(yaml_input, params.ptr());
    params->validateParameters(get_valid_params(), 0);

    RCP<MultiQoI> multi_qoi = rcp(new MultiQoI(params));
    multi_qoi->evaluate();
    multi_qoi->write_output();
    multi_qoi->cleanup();
  }
  finalize();
}
