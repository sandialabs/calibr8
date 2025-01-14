#include <Teuchos_YamlParameterListHelpers.hpp>
#include <PCU.h>
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
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
  p.sublist("inverse");
  p.sublist("virtual fields");
  return p;
}

class VFM_Objective {
  public:
    VFM_Objective(std::string const& input_file, bool evaluate_gradient);
    void evaluate();
  private:
    void setup_opt_params(ParameterList const& inverse_params);
    bool m_evaluate_gradient;
    RCP<ParameterList> m_params;
    Array1D<RCP<State>> m_state;
    Array1D<RCP<Primal>> m_primal;
    Array1D<RCP<ParameterList>> m_prob_params;
    int m_num_problems;
    int m_num_opt_params;
    int m_model_form = 0;
    RCP<VirtualPower> m_virtual_power;
    std::string m_load_in_file;
    Array1D<double> m_load_data;
};

VFM_Objective::VFM_Objective(std::string const& input_file, bool evaluate_gradient) {
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_params->validateParameters(get_valid_params(), 0);
  m_evaluate_gradient = evaluate_gradient;

  bool const is_multiple = m_params->isSublist("problems");
  if (is_multiple) {
    fail("VFM not set up for multiple problems");
    auto problems_list = m_params->sublist("problems");
    for (auto problem_entry : problems_list) {
      auto problem_list = Teuchos::getValue<ParameterList>(problem_entry.second);
      auto rcp_prob_list = RCP(new ParameterList);
      *rcp_prob_list = problem_list;
      m_prob_params.push_back(rcp_prob_list);
      auto state = rcp(new State(problem_list));
      auto primal = rcp(new Primal(rcp_prob_list, state, state->disc));
      m_state.push_back(state);
      m_primal.push_back(primal);
    }
    m_num_problems = m_state.size();
  } else {
    m_num_problems = 1;
    m_state.resize(1);
    m_primal.resize(1);
    m_prob_params.resize(1);
    m_state[0] = rcp(new State(*m_params));
    m_primal[0] = rcp(new Primal(m_params, m_state[0], m_state[0]->disc));
    m_prob_params[0] = m_params;
  }

  setup_opt_params(m_params->sublist("inverse", true));

  //TODO: generalize to multiple problems
  m_virtual_power = rcp(new VirtualPower(m_params, m_state[0], m_state[0]->disc,
      m_num_opt_params));
  ParameterList& inverse_params = m_params->sublist("inverse", true);
  std::string load_in_file = inverse_params.get<std::string>("load input file");
  std::ifstream in_file(load_in_file);
  std::string line;
  while (getline(in_file, line)) {
    m_load_data.push_back(std::stod(line));
  }


}

void VFM_Objective::setup_opt_params(ParameterList const& inverse_params) {
  Array1D<std::string> const& elem_set_names =
      m_state[0]->residuals->local[m_model_form]->elem_set_names();
  Array1D<std::string> const& param_names =
      m_state[0]->residuals->local[m_model_form]->param_names();

  ParameterList const& all_material_params =
      inverse_params.sublist("materials");

  int const num_elem_sets = elem_set_names.size();
  int const num_params_total = param_names.size();

  Array2D<int> active_indices(num_elem_sets);
  m_num_opt_params = 0;

  for (int es = 0; es < num_elem_sets; ++es) {
    ParameterList material_params =
        all_material_params.sublist(elem_set_names[es]);
    for (int i = 0; i < num_params_total; ++i) {
      std::string param_name = param_names[i];
      if (material_params.isParameter(param_name)) {
        active_indices[es].push_back(i);
        m_num_opt_params++;
      }
    }
  }

  int const num_dfad_params = m_state[0]->residuals->local[m_model_form]->num_dfad_params();
  m_num_opt_params += num_dfad_params;

  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->residuals->local[m_model_form]->set_active_indices(active_indices);
    m_state[prob]->d_residuals->local[m_model_form]->set_active_indices(active_indices);
  }
}

void VFM_Objective::evaluate() {

  ParameterList& inverse_params = m_params->sublist("inverse", true);
  double const obj_scale_factor = inverse_params.get<double>("objective scale factor");
  double dt;
  double internal_virtual_power;
  double load_at_step;
  double virtual_power_mismatch;

  double scaled_virtual_power_mismatch;
  Array1D<double> internal_virtual_power_vec;

  double J = 0.;
  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);

  for (int prob = 0; prob < m_num_problems; ++prob) {
    double J_prob = 0.;
    int const nsteps = m_state[prob]->disc->num_time_steps();
    internal_virtual_power_vec.resize(nsteps, 0.);
    double const total_time = m_state[prob]->disc->time(nsteps) - m_state[prob]->disc->time(0);

    m_state[prob]->disc->destroy_primal();

      for (int step = 1; step <= nsteps; ++step) {
        dt = m_state[prob]->disc->dt(step);
        internal_virtual_power = m_virtual_power->compute_at_step(step);
        internal_virtual_power_vec[step - 1] = internal_virtual_power;
      }

      for (int step = nsteps; step > 0; --step) {
        dt = m_state[prob]->disc->dt(step);
        load_at_step = m_load_data[step - 1];

        virtual_power_mismatch = internal_virtual_power_vec[step - 1] - load_at_step;
        scaled_virtual_power_mismatch = virtual_power_mismatch * obj_scale_factor * dt / total_time;
        J_prob += 0.5 * virtual_power_mismatch * scaled_virtual_power_mismatch;

        if (m_evaluate_gradient) {
          m_virtual_power->compute_at_step_adjoint(
              step, scaled_virtual_power_mismatch, grad_at_step
          );

          for (int i = 0; i < m_num_opt_params; ++i) {
            grad[i] += grad_at_step[i];
          }
        }
      }

    J += J_prob;
  }

  PCU_Add_Doubles(grad.data(), m_num_opt_params);

  if (PCU_Comm_Self() == 0) {
    std::ofstream obj_file;
    std::string const obj_filename = "objective_value.txt";
    obj_file.open(obj_filename);
    obj_file << std::scientific << std::setprecision(17);
    obj_file << J << "\n";
    obj_file.close();
    if (m_evaluate_gradient) {
      std::ofstream grad_file;
      std::string const grad_filename = "objective_gradient.txt";
      grad_file.open(grad_filename);
      grad_file << std::scientific << std::setprecision(17);
      for (int i = 0; i < m_num_opt_params; ++i) {
        grad_file << grad[i] << "\n";
      }
      grad_file.close();
    }
  }

  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->disc->destroy_primal();
  }
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 3);
  {
    std::string const yaml_input = argv[1];
    std::string const compute_gradient = argv[2];
    bool eval_grad;
    if (compute_gradient == "true") {
      eval_grad = true;
    } else {
      eval_grad = false;
    }
    VFM_Objective objective(yaml_input, eval_grad);
    objective.evaluate();
  }
  finalize();
}
