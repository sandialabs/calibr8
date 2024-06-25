#include "control.hpp"
#include "objective.hpp"
#include "local_residual.hpp"

namespace calibr8 {

Objective::Objective(RCP<ParameterList> params) {
  m_params = params;
  bool const is_multiple = params->isSublist("problems");
  if (is_multiple) {
    auto problems_list = params->sublist("problems");
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
  setup_opt_params(params->sublist("inverse", true));
}

Objective::~Objective() {
  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->disc->destroy_primal();
  }
}

Array1D<double> Objective::transform_params(Array1D<double> const& params,
    bool scale_to_canonical) const {
  double upper, lower, mean, span;
  Array1D<double> transformed_params(m_num_opt_params);
  for (int i = 0; i < m_num_opt_params; ++i) {
    lower = m_lower_bounds[i];
    upper = m_upper_bounds[i];
    span = 0.5 * (upper - lower);
    mean = 0.5 * (upper + lower);
    if (scale_to_canonical) {
      transformed_params[i] = (params[i] - mean) / span;
      if (transformed_params[i] < -1.)
        transformed_params[i] = -1.;
      if (transformed_params[i] > 1.)
        transformed_params[i] = 1.;
    } else {
      transformed_params[i] = span * params[i] + mean;
    }
  }
  return transformed_params;
}

Array1D<double> Objective::active_params() const {
  return m_state[0]->residuals->local[m_model_form]->active_params();
}

Array2D<std::string> Objective::active_param_names() const {
  return m_state[0]->residuals->local[m_model_form]->active_param_names();
}

Array1D<std::string> Objective::elem_set_names() const {
  return m_state[0]->residuals->local[m_model_form]->elem_set_names();
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
  m_lower_bounds.resize(0);
  m_upper_bounds.resize(0);
  m_num_opt_params = 0;

  for (int es = 0; es < num_elem_sets; ++es) {
    ParameterList material_params =
        all_material_params.sublist(elem_set_names[es]);
    for (int i = 0; i < num_params_total; ++i) {
      std::string param_name = param_names[i];
      if (material_params.isParameter(param_name)) {
        active_indices[es].push_back(i);
        Teuchos::Array<double> const bounds
            = material_params.get<Teuchos::Array<double>>(param_name);
        m_lower_bounds.push_back(bounds[0]);
        m_upper_bounds.push_back(bounds[1]);
        m_num_opt_params++;
      }
    }
  }

  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->residuals->local[m_model_form]->set_active_indices(active_indices);
    m_state[prob]->d_residuals->local[m_model_form]->set_active_indices(active_indices);
  }

  // initialize p_old
  m_p_old.resize(m_num_opt_params);
  for (int i = 0; i < m_num_opt_params; ++i) {
    m_p_old[i] = 2.;
  }

}

Array1D<double> Objective::transform_gradient(
    Array1D<double> const& gradient) const {

  double upper, lower, span;
  Array1D<double> transformed_gradient(m_num_opt_params);
  for (int i = 0; i < m_num_opt_params; ++i) {
    lower = m_lower_bounds[i];
    upper = m_upper_bounds[i];
    span = 0.5 * (upper - lower);
    transformed_gradient[i] = span * gradient[i];
  }
  return transformed_gradient;
}

ROL::Ptr<Array1D<double> const> Objective::getVector(V const& vec) {
  return dynamic_cast<SV const&>(vec).getVector();
}

ROL::Ptr<Array1D<double>> Objective::getVector(V& vec) {
  return dynamic_cast<SV&>(vec).getVector();
}

bool Objective::param_diff(std::vector<double> const& p_new) {
  double diffnorm = 0.0;
  for (int i = 0; i < m_num_opt_params; ++i) {
    diffnorm += pow(p_new[i] - m_p_old[i] , 2);
    m_p_old[i] = p_new[i];
  }
  diffnorm = sqrt(diffnorm);
  if (diffnorm < m_difftol)
    return false;
  else
    return true;
}

}
