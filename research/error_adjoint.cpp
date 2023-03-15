#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  (void)params;
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {


  // ---
  // PRIMAL PROBLEM
  // ---

  print("solving coarse primal problem");
  m_u_coarse = physics->solve_primal(COARSE);
  print("solving fine primal problem");
  m_u_fine = physics->solve_primal(FINE);
  print("prolonging the coarse primal solution onto the fine space");
  m_u_prolonged = physics->prolong(m_u_coarse, "u_prolonged");
  print("restricting the fine primal solution onto the coarse space");
  m_u_restricted = physics->restrict(m_u_fine, "u_restricted");
  print("applying SPR to the coarse primal solution");
  m_u_recovered = physics->recover(m_u_coarse, "u_recovered");

  // ---
  // QUANTITIES OF INTEREST
  // ---

  print("computing the QoI on the coarse space");
  double const J_coarse = physics->compute_qoi(COARSE, m_u_coarse);
  print("computing the QoI on the fine space");
  double const J_fine = physics->compute_qoi(FINE, m_u_fine);
  double const Jeh = J_fine - J_coarse;

  // ---
  // PRIMAL DISCRETIZATION ERROR
  // ---

  print("computing the exact primal discretization error");
  m_ue_exact = physics->subtract(m_u_fine, m_u_prolonged, "ue_exact");
  print("computing the linearized primal discretization error");
  m_ue_linearized = physics->solve_linearized_error(m_u_prolonged, "ue_linearized");
  print("computing the recovered primal discretization error");
  m_ue_recovered = physics->subtract(m_u_recovered, m_u_prolonged, "ue_recovered");

  // ---
  // LINEAR ADJOINT PROBLEM
  // ---

  print("solving coarse adjoint problem");
  m_z_coarse = physics->solve_adjoint(COARSE, m_u_coarse);
  print("solving fine adjoint problem");
  m_z_fine = physics->solve_adjoint(FINE, m_u_prolonged);
  print("prolonging the coarse adjoint solution onto the fine space");
  m_z_prolonged = physics->prolong(m_z_coarse, "z_prolonged");
  print("restricting the fine adjoint solution onto the coarse space");
  m_z_restricted = physics->restrict(m_z_fine, "z_restricted");
  print("prolonging the restricted adjoint solution onto the fine space");
  m_z_restricted_fine = physics->prolong(m_z_restricted, "z_restricted_fine");
  print("applying SPR to the coarse adjoint solution");
  m_z_recovered = physics->recover(m_z_coarse, "z_recovered");

  // ---
  // ADJOINT DISCRETIZATION ERROR
  // ---

  print("computing the exact adjoint discretization error");
  m_ze_exact = physics->subtract(m_z_fine, m_z_prolonged, "ze_exact");
  print("computing the restricted adjoint discretization error");
  m_ze_restricted = physics->subtract(m_z_fine, m_z_restricted_fine, "ze_restricted");
  print("computing the recovered adjoint discretization error");
  m_ze_recovered = physics->subtract(m_z_recovered, m_z_prolonged, "ze_recovered");

  // ---
  // 2nd ORDER ADJOINT PROBLEM
  // ---

  print("solving 2nd order adjoint with linearized discretization error");
  m_y_linearized = physics->solve_2nd_adjoint(m_u_prolonged, m_ue_linearized, "y_linearized");
  print("solving 2nd order adjoint with recovered discretization error");
  m_y_recovered = physics->solve_2nd_adjoint(m_u_prolonged, m_ue_recovered, "y_recovered");

  // ---
  // RESIDUAL LINEARIZATION ERROR
  // ---

  print("computing the exact residual linearization error");
  m_ERL_exact = physics->solve_ERL(m_u_prolonged, m_ue_exact, "ERL_exact");
  print("computing the recovered residual linearization error");
  m_ERL_recovered = physics->solve_ERL(m_u_prolonged, m_ue_recovered, "ERL_recovered");

  // ---
  // NONLINEAR ADJOINT PROBLEM
  // ---

  print("solving nonlinear adjoint problem");
  exact_in in;
  in.u_coarse = m_u_prolonged;
  in.u_fine = m_u_fine;
  in.ue_exact = m_ue_exact;
  in.J_coarse = J_coarse;
  in.J_fine = J_fine;
  exact_out out = physics->solve_exact_adjoint(in);
  m_u_star_J = out.u_star_J;
  m_u_star_R = out.u_star_R;
  m_z_star = out.z_star;

  // ---
  // COMPUTABLE ERROR CONTRIBUTIONS
  // ---

  print("computing error contributions");
  m_R_prolonged = physics->evaluate_residual(FINE, m_u_prolonged);
  int const num_elems = get_nelems(physics);
  int const num_coarse_dofs = get_ndofs(COARSE, physics);
  int const num_fine_dofs = get_ndofs(FINE, physics);
  double const eta1_exact = -(physics->dot(m_z_fine, m_R_prolonged));
  double const eta1_recovered = -(physics->dot(m_z_recovered, m_R_prolonged));
  double const eta2_linearized = -(physics->dot(m_y_linearized, m_R_prolonged));
  double const eta2_recovered = -(physics->dot(m_y_recovered, m_R_prolonged));

  // ---
  // ITERATION SUMMARY
  // ---

  print("summary for this adaptive iteration");
  print("> elems = %d", num_elems);
  print("> H_dofs = %d", num_coarse_dofs);
  print("> h_dofs = %d", num_fine_dofs);
  print("> JH = %.15e", J_coarse);
  print("> Jh = %.15e", J_fine);
  print("> Jeh = %.15e", Jeh);
  print("> eta1_exact = %.15e", eta1_exact);
  print("> eta1_recovered = %.15e", eta1_recovered);
  print("> eta2_linearized = %.15e", eta2_linearized);
  print("> eta2_recovered = %.15e", eta2_recovered);

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
  apf::destroyField(m_ue_exact); m_ue_exact = nullptr;
  apf::destroyField(m_ue_linearized); m_ue_linearized = nullptr;
  apf::destroyField(m_ue_recovered); m_ue_recovered = nullptr;
  apf::destroyField(m_z_coarse); m_z_coarse = nullptr;
  apf::destroyField(m_z_fine); m_z_fine = nullptr;
  apf::destroyField(m_z_prolonged); m_z_prolonged = nullptr;
  apf::destroyField(m_z_restricted); m_z_restricted = nullptr;
  apf::destroyField(m_z_restricted_fine); m_z_restricted_fine = nullptr;
  apf::destroyField(m_z_recovered); m_z_recovered = nullptr;
  apf::destroyField(m_ze_exact); m_ze_exact = nullptr;
  apf::destroyField(m_ze_restricted); m_ze_restricted = nullptr;
  apf::destroyField(m_ze_recovered); m_ze_recovered = nullptr;
  apf::destroyField(m_y_linearized); m_y_linearized = nullptr;
  apf::destroyField(m_y_recovered); m_y_recovered = nullptr;
  apf::destroyField(m_ERL_exact); m_ERL_exact = nullptr;
  apf::destroyField(m_ERL_recovered); m_ERL_recovered = nullptr;
  apf::destroyField(m_u_star_J); m_u_star_J = nullptr;
  apf::destroyField(m_u_star_R); m_u_star_R = nullptr;
  apf::destroyField(m_z_star); m_z_star = nullptr;
  apf::destroyField(m_R_prolonged); m_R_prolonged = nullptr;
}

}
