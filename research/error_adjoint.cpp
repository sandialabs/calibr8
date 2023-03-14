#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  (void)params;
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {

  // ---
  // DISCRETIZATION INFORMATION
  // ---

  // compute and store discretization information
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));

  // ---
  // PRIMAL PROBLEM DATA
  // ---

  // solve the primal problem on the coarse space
  print("solving coarse primal problem");
  m_u_coarse = physics->solve_primal(COARSE);

  // solve the primal problem on the fine space
  print("solving fine primal problem");
  m_u_fine = physics->solve_primal(FINE);

  // prolong the coarse primal solution onto the fine space
  print("prolonging the coarse primal solution onto the fine space");
  m_u_prolonged = physics->prolong(m_u_coarse, "u_prolonged");

  // restrict the fine primal solution onto the coarse space
  print("restricting the fine primal solution onto the coarse space");
  m_u_restricted = physics->restrict(m_u_fine, "u_restricted");

  // recover the fine primal solution using SPR
  print("applying SPR to the coarse primal solution");
  m_u_recovered = physics->recover(m_u_coarse, "u_recovered");

  // ---
  // DISCRETIZATION ERROR DATA
  // ---

  // compute the exact discretization error
  print("computing the exact discretization error");
  m_e_exact = physics->subtract(m_u_fine, m_u_prolonged, "e_exact");

  // compute the linearized discretization error
  print("computing the linearized discretization error");
  m_e_linearized = physics->solve_linearized_error(m_u_prolonged);

  // compute the recovered discretization error
  print("computing the recovered discretization error");
  m_e_recovered = physics->subtract(m_u_recovered, m_u_prolonged, "e_recovered");


#if 0
  // solve the primal problem on the coarse space
  print("solving primal H");
  m_uH = physics->solve_primal(COARSE);
  print("computing qoi H");
  double const JH = physics->compute_qoi(COARSE, m_uH);
  print(" > JH = %.15e", JH);

  // solve the primal problem on the fine space
  print("solving primal h");
  m_uh = physics->solve_primal(FINE);
  print("computing qoi h");
  double const Jh = physics->compute_qoi(FINE, m_uh);
  print(" > Jh = %.15e", Jh);

  // prolong the coarse primal solution onto the fine space
  print("prolonging uH onto h");
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);

  // solve auxiliary problems on the fine space
  print("solving linearized error h");
  m_elh = physics->solve_linearized_error(m_uH_h);
  print("solving adjoint h");
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
  print("solving 2nd adjoint h");
  m_yh = physics->solve_2nd_adjoint(m_uH_h, m_elh);

  // compute the residual error contributions
  print("evaluating eta1");
  m_Rh_uH = physics->evaluate_residual(m_uH_h);
  double const eta1 = -(physics->dot(m_zh, m_Rh_uH));
  print("> eta1 = %.15e", eta1);
  print("evaluating eta2");
  double const eta2 = -(physics->dot(m_yh, m_Rh_uH));
  print("> eta2 = %.15e", eta2);
#endif


  apf::Mesh2* m = physics->disc()->apf_mesh();
  apf::Field* e = apf::createStepField(m, "eta", apf::SCALAR);
  apf::zeroField(e);
  return e;
}

void Adjoint::write_history(std::string const& file, double J_ex) {
  (void)file;
  (void)J_ex;
}

void Adjoint::destroy_intermediate_fields() {
  apf::destroyField(m_u_coarse); m_u_coarse = nullptr;
  apf::destroyField(m_u_fine); m_u_fine = nullptr;
  apf::destroyField(m_u_prolonged); m_u_prolonged = nullptr;
  apf::destroyField(m_u_restricted); m_u_restricted = nullptr;
  apf::destroyField(m_u_recovered); m_u_recovered = nullptr;
  apf::destroyField(m_e_exact); m_e_exact = nullptr;
  apf::destroyField(m_e_linearized); m_e_linearized = nullptr;
  apf::destroyField(m_e_recovered); m_e_recovered = nullptr;
}

}
