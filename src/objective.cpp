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

Objective::~Objective() {}

Array1D<double> Objective::model_params() const {
  return m_state->residuals->local->params();
}

ROL::Ptr<Array1D<double> const> Objective::getVector(V const& vec) {
  return dynamic_cast<SV const&>(vec).getVector();
}

ROL::Ptr<Array1D<double>> Objective::getVector(V& vec) {
  return dynamic_cast<SV&>(vec).getVector();
}

Array1D<double> Objective::extract_active_params(
    Array1D<double> const& model_params) const {
  Array1D<double> active_params(m_num_opt_params);
  for (size_t i = 0; i < m_num_opt_params; ++i) {
    active_params[i] = model_params[m_active_indices[i]];
  }
  return active_params;
}

void Objective::setup_opt_params(ParameterList const& inverse_params) {
  Array1D<std::string> const& param_names =
      m_state->residuals->local->param_names();
  size_t const num_params_total = param_names.size();
  m_active_indices.resize(0);
  m_lower_bounds.resize(0);
  m_upper_bounds.resize(0);
  m_active_param_names.resize(0);
  for (size_t i = 0; i < num_params_total; ++i) {
    std::string param_name = param_names[i];
    if (inverse_params.isParameter(param_name)) {
      m_active_param_names.push_back(param_name);
      m_active_indices.push_back(i);
      Teuchos::Array<double> const bounds
          = inverse_params.get<Teuchos::Array<double>>(param_name);
      m_lower_bounds.push_back(bounds[0]);
      m_upper_bounds.push_back(bounds[1]);
    }
  }
  m_num_opt_params = m_active_indices.size();
}

Array1D<double> Objective::transform_params(Array1D<double> const& params,
    bool scale_to_canonical) {
  double upper, lower, mean, span;
  Array1D<double> transformed_params(m_num_opt_params);
  for (size_t i = 0; i < m_num_opt_params; ++i) {
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

Array1D<double> Objective::transform_gradient(Array1D<double> const& gradient) {
    double upper, lower, span;
  Array1D<double> transformed_gradient(m_num_opt_params);
  for (size_t i = 0; i < m_num_opt_params; ++i) {
    lower = m_lower_bounds[i];
    upper = m_upper_bounds[i];
    span = 0.5 * (upper - lower);
    transformed_gradient[i] = span * gradient[i];
  }
  return transformed_gradient;
}

}
