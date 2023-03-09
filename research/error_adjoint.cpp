#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  (void)params;
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {

  // compute and store discretization information
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));

  // solve the primal problem on the coarse space
  m_uH = physics->solve_primal(COARSE);
  double const JH = physics->compute_qoi(COARSE, m_uH);

  // solve the primal problem on the fine space
  m_uh = physics->solve_primal(FINE);
  double const Jh = physics->compute_qoi(FINE, m_uh);

  // prolong the coarse primal solution onto the fine space
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);

  // solve auxiliary problems on the fine space
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
  m_el = physics->solve_linearized_error(m_uH_h);
  m_yh = physics->solve_2nd_adjoint(m_uH_h, m_el);

  apf::Mesh2* m = physics->disc()->apf_mesh();
  apf::Field* e = apf::createStepField(m, "e", apf::SCALAR);
  apf::zeroField(e);
  return e;
}

void Adjoint::write_history(std::string const& file, double J_ex) {
  (void)file;
  (void)J_ex;
}

void Adjoint::destroy_intermediate_fields() {
  apf::destroyField(m_uH); m_uH = nullptr;
  apf::destroyField(m_uh); m_uh = nullptr;
  apf::destroyField(m_uH_h); m_uH_h = nullptr;
  apf::destroyField(m_zh); m_zh = nullptr;
  apf::destroyField(m_el); m_el = nullptr;
}

}
