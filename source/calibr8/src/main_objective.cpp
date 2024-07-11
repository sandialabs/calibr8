#include <Teuchos_YamlParameterListHelpers.hpp>
#include <PCU.h>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
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
  p.sublist("inverse");
  return p;
}

class Objective {
  public:
    Objective(std::string const& input_file, bool evaluate_gradient);
    void evaluate();
  private:
    void setup_opt_params(ParameterList const& inverse_params);
    bool m_evaluate_gradient;
    RCP<ParameterList> m_params;
    Array1D<RCP<State>> m_state;
    Array1D<RCP<Primal>> m_primal;
    Array1D<RCP<Adjoint>> m_adjoint;
    Array1D<RCP<ParameterList>> m_prob_params;
    int m_num_problems;
    int m_num_opt_params;
    int m_model_form = 0;
};

Objective::Objective(std::string const& input_file, bool evaluate_gradient) {
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_params->validateParameters(get_valid_params(), 0);
  m_evaluate_gradient = evaluate_gradient;

  bool const is_multiple = m_params->isSublist("problems");
  if (is_multiple) {
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

  if (evaluate_gradient) {
    m_adjoint.resize(m_num_problems);
    for (int prob = 0; prob < m_num_problems; ++prob) {
      m_adjoint[prob] = rcp(new Adjoint(m_prob_params[prob], m_state[prob],
          m_state[prob]->disc));
    }
  }
}

void Objective::setup_opt_params(ParameterList const& inverse_params) {
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

  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->residuals->local[m_model_form]->set_active_indices(active_indices);
    m_state[prob]->d_residuals->local[m_model_form]->set_active_indices(active_indices);
  }
}


void Objective::evaluate() {
  double J = 0.;
  for (int prob = 0; prob < m_num_problems; ++prob) {
    double J_prob = 0.;
    int const nsteps = m_state[prob]->disc->num_time_steps();
    // solve the primal problem
    for (int step = 1; step <= nsteps; ++step) {
      m_primal[prob]->solve_at_step(step);
      J_prob += eval_qoi(m_state[prob], m_state[prob]->disc, step);
    }
    J_prob = PCU_Add_Double(J_prob);
    J += J_prob;
  }

  if (PCU_Comm_Self() == 0) {
    std::ofstream obj_file;
    std::string const obj_filename = "objective_value.txt";
    obj_file.open(obj_filename);
    obj_file << std::scientific << std::setprecision(17);
    obj_file << J << "\n";
    obj_file.close();
  }

  if (m_evaluate_gradient) {
    Array1D<double> grad(m_num_opt_params, 0.);
    for (int prob = 0; prob < m_num_problems; ++prob) {
      Array1D<double> grad_at_step(m_num_opt_params);
      int const nsteps = m_state[prob]->disc->num_time_steps();
      m_state[prob]->disc->create_adjoint(m_state[prob]->residuals, nsteps);
      // solve the adjoint problem
      for (int step = nsteps; step > 0; --step) {
        m_adjoint[prob]->solve_at_step(step);
        grad_at_step = eval_qoi_gradient(m_state[prob], step);
        for (int i = 0; i < m_num_opt_params; ++i) {
          grad[i] += grad_at_step[i];
        }
      }
      m_state[prob]->disc->destroy_adjoint();
    }
    PCU_Add_Doubles(grad.data(), m_num_opt_params);

    if (PCU_Comm_Self() == 0) {
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
    Objective objective(yaml_input, eval_grad);
    objective.evaluate();
  }
  finalize();
}
