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

Array2D<double> Objective::model_params() const {
  return m_state->residuals->local->params();
}

ROL::Ptr<Array1D<double> const> Objective::getVector(V const& vec) {
  return dynamic_cast<SV const&>(vec).getVector();
}

ROL::Ptr<Array1D<double>> Objective::getVector(V& vec) {
  return dynamic_cast<SV&>(vec).getVector();
}

Array1D<double> Objective::active_params() const {
  Array1D<double> active_params(m_num_opt_params);
  Array2D<double> const& all_model_params = model_params();
  int const num_elem_sets = m_elem_set_names.size();
  int p = 0;
  for (int es = 0; es < num_elem_sets; ++es) {
    int const num_active_params = m_active_param_names[es].size();
    for (int i = 0; i < num_active_params; ++i) {
      active_params[p] = all_model_params[es][i];
      ++p;
    }
  }
  return active_params;
}

void Objective::setup_opt_params(ParameterList const& inverse_params) {
  m_elem_set_names = m_state->residuals->local->elem_set_names();
  Array1D<std::string> const& param_names =
      m_state->residuals->local->param_names();

  ParameterList const& all_material_params =
      inverse_params.sublist("materials");

  int const num_elem_sets = m_elem_set_names.size();
  int const num_params_total = param_names.size();
  m_active_indices.resize(num_elem_sets);
  m_active_param_names.resize(num_elem_sets);
  m_lower_bounds.resize(0);
  m_upper_bounds.resize(0);
  m_num_opt_params = 0;
  for (int es = 0; es < num_elem_sets; ++es) {
    ParameterList material_params =
        all_material_params.sublist(m_elem_set_names[es]);
    for (int i = 0; i < num_params_total; ++i) {
      std::string param_name = param_names[i];
      if (material_params.isParameter(param_name)) {
        m_active_param_names[es].push_back(param_name);
        m_active_indices[es].push_back(i);
        Teuchos::Array<double> const bounds
            = material_params.get<Teuchos::Array<double>>(param_name);
        m_lower_bounds.push_back(bounds[0]);
        m_upper_bounds.push_back(bounds[1]);
        m_num_opt_params++;
      }
    }
  }
}

Array1D<double> Objective::transform_params(Array1D<double> const& params,
    bool scale_to_canonical) {
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

Array1D<double> Objective::transform_gradient(Array1D<double> const& gradient) {
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

}
