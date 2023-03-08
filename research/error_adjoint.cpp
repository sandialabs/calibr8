#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  (void)params;
}

void Adjoint::solve_primal(int space, RCP<Physics> physics) {
  apf::Field* u = physics->solve_primal(space);
  double const J = physics->compute_qoi(space, u);
  if (space == COARSE) {
    m_uH = u;
    m_JH.push_back(J);
  }
  if (space == FINE) {
    m_uh = u;
    m_Jh.push_back(J);
  }
}

void Adjoint::post_process_primal(RCP<Physics> physics) {
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_uh_H = physics->restrict_u_fine_onto_fine(m_uh);
  m_uh_minus_m_uH_h = physics->subtract_u_coarse_from_u_fine(m_uh, m_uH_h);
}

void Adjoint::solve_adjoint(RCP<Physics> physics) {
  m_zH = physics->solve_adjoint(COARSE, m_uH);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
}

void Adjoint::post_process_adjoint(RCP<Physics> physics) {
  m_zH_h = physics->prolong_z_coarse_onto_fine(m_zH);
  m_zh_H = physics->restrict_z_fine_onto_fine(m_zh);
  m_zh_minus_m_zh_H = physics->subtract_z_coarse_from_z_fine(m_zh, m_zh_H);
}

void Adjoint::solve_linearized_error(RCP<Physics> physics) {
  m_uh_minus_m_uH_h_LE = physics->solve_linearized_error(m_uH_h);
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));
  solve_primal(COARSE, physics);
  solve_primal(FINE, physics);
  post_process_primal(physics);
  solve_adjoint(physics);
  post_process_adjoint(physics);
  solve_linearized_error(physics);

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
  apf::destroyField(m_uh_H); m_uh_H = nullptr;
  apf::destroyField(m_uH_h); m_uH_h = nullptr;
  apf::destroyField(m_uh_minus_m_uH_h); m_uh_minus_m_uH_h = nullptr;
  apf::destroyField(m_zH); m_zH = nullptr;
  apf::destroyField(m_zh); m_zh = nullptr;
  apf::destroyField(m_zh_H); m_zh_H = nullptr;
  apf::destroyField(m_zH_h); m_zH_h = nullptr;
  apf::destroyField(m_zh_minus_m_zh_H); m_zh_minus_m_zh_H = nullptr;
  apf::destroyField(m_uh_minus_m_uH_h_LE); m_uh_minus_m_uH_h_LE = nullptr;
}

}
