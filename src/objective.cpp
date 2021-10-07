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
  m_num_opt_params = m_state->residuals->local->params().size();
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

}
