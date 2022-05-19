#include "control.hpp"
#include "evaluations.hpp"
#include "local_residual.hpp"
#include "objective.hpp"
#include "primal.hpp"
#include "state.hpp"

namespace calibr8 {

Objective::Objective(RCP<ParameterList> params) {
  m_params = params;
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  setup_opt_params(params->sublist("inverse", true));
}

Objective::~Objective() {
  m_state->disc->destroy_primal();
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
  return m_state->residuals->local->active_params();
}

Array2D<std::string> Objective::active_param_names() const {
  return m_state->residuals->local->active_param_names();
}

Array1D<std::string> Objective::elem_set_names() const {
  return m_state->residuals->local->elem_set_names();
}

void Objective::setup_opt_params(ParameterList const& inverse_params) {
  Array1D<std::string> const& elem_set_names =
      m_state->residuals->local->elem_set_names();
  Array1D<std::string> const& param_names =
      m_state->residuals->local->param_names();

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
  m_state->residuals->local->set_active_indices(active_indices);
  m_state->d_residuals->local->set_active_indices(active_indices);

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
