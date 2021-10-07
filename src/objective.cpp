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

Array1D<double> Objective::opt_params() const {
  return m_state->residuals->local->params();
}

ROL::Ptr<Array1D<double> const> Objective::getVector(const V& vec) {
  return dynamic_cast<const SV&>(vec).getVector();
}

ROL::Ptr<Array1D<double>> Objective::getVector(V& vec) {
  return dynamic_cast<SV&>(vec).getVector();
}

void Objective::setup_opt_params(ParameterList const& inverse_params) {
  Array1D<std::string> const& param_names =
      m_state->residuals->local->param_names();
  int const num_params_total = param_names.size();
  m_active_indices.resize(0);
  m_lower_bounds.resize(0);
  m_upper_bounds.resize(0);
  for (int i = 0; i < num_params_total; ++i) {
    std::string param_name = param_names[i];
    if (inverse_params.isParameter(param_name)) {
      m_active_indices.push_back(i);
      Teuchos::Array<double> const bounds
          = inverse_params.get<Teuchos::Array<double>>(param_name);
      m_lower_bounds.push_back(bounds[0]);
      m_upper_bounds.push_back(bounds[1]);
    }
  }
  m_num_opt_params = m_active_indices.size();
}

}
