#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  auto const& resid_params = params.sublist("residual");
  auto const& qoi_params = params.sublist("quantity of interest");
  double const alpha = resid_params.get<double>("alpha");
  double const beta = qoi_params.get<double>("beta");
  m_linear_physics = (alpha == 0.);
  m_linear_qoi = (beta == 1.);
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
  print("applying SPR to the coarse primal solution");
  m_u_recovered = physics->recover(m_u_coarse, "u_recovered");

  // ---
  // QUANTITIES OF INTEREST
  // ---

  print("computing the QoI on the coarse space");
  double const J_coarse = physics->compute_qoi(COARSE, m_u_coarse);
  print("computing the QoI on the fine space");
  double const J_fine = physics->compute_qoi(FINE, m_u_fine);
  print("computing the QoI at the recovered solution");
  double const J_recovered = physics->compute_qoi(FINE, m_u_recovered);
  double const Jeh = J_fine - J_coarse;

  // ---
  // PRIMAL DISCRETIZATION ERROR
  // ---

  print("computing the exact primal discretization error");
  m_ue_exact = physics->subtract(m_u_fine, m_u_prolonged, "ue_exact");
  print("computing the recovered primal discretization error");
  m_ue_recovered = physics->subtract(m_u_recovered, m_u_prolonged, "ue_recovered");

  // ---
  // LINEAR ADJOINT PROBLEM
  // ---

  print("solving coarse adjoint problem");
  m_z_coarse = physics->solve_adjoint(COARSE, m_u_coarse);
  print("solving fine adjoint problem");
  m_z_fine = physics->solve_adjoint(FINE, m_u_prolonged);
  print("applying SPR to the coarse adjoint solution");
  m_z_recovered = physics->recover(m_z_coarse, "z_recovered");

  // ---
  // 2nd ORDER ADJOINT PROBLEM
  // ---

  print("solving 2nd order adjoint with exact discretization error");
  m_y_exact = physics->solve_2nd_adjoint(m_u_prolonged, m_ue_exact, "y_exact");
  print("solving 2nd order adjoint with recovered discretization error");
  m_y_recovered = physics->solve_2nd_adjoint(m_u_prolonged, m_ue_recovered, "y_recovered");
  if (m_linear_qoi) apf::zeroField(m_y_recovered);

  // ---
  // RESIDUAL LINEARIZATION ERROR
  // ---

  print("computing the exact residual linearization error");
  m_ERL_exact = physics->solve_ERL(m_u_prolonged, m_ue_exact, "ERL_exact");
  print("computing the recovered residual linearization error");
  m_ERL_recovered = physics->solve_ERL(m_u_prolonged, m_ue_recovered, "ERL_recovered");
  if (m_linear_physics) apf::zeroField(m_ERL_recovered);

  // ---
  // RESIDUAL EVALUATED AT THE PROLONGED COARSE SOLUTION
  // ---

  m_R_prolonged = physics->evaluate_residual(FINE, m_u_prolonged);

  // ---
  // EXACT MODIFIED ADJOINT PROBLEM
  // ---

  print("solving modified adjoint problem");
  {
    nonlinear_in in;
    in.u_coarse = m_u_prolonged;
    in.u_fine = m_u_fine;
    in.ue = m_ue_exact;
    in.J_coarse = J_coarse;
    in.J_fine = J_fine;
    nonlinear_out out = physics->solve_nonlinear_adjoint(in);
    m_u_star = out.u_star;
    m_z_star = out.z_star;
  }
  m_z_star_star = physics->modify_star(
      m_z_star, m_R_prolonged, m_ERL_exact, "z_star_star");

  // ---
  // RECOVERED MODIFIED ADJOINT PROBLEM
  // ---

  print("solving recovered modified adjoint problem");
  {
    nonlinear_in in;
    in.name_append = "_recovered";
    in.u_coarse = m_u_prolonged;
    in.u_fine = m_u_recovered;
    in.ue = m_ue_recovered;
    in.J_coarse = J_coarse;
    in.J_fine = J_recovered;
    nonlinear_out out = physics->solve_nonlinear_adjoint(in);
    m_u_star_recovered = out.u_star;
    m_z_star_recovered = out.z_star;
    in.J_fine = J_recovered;
  }
  m_z_star_star_recovered = physics->modify_star(
      m_z_star_recovered, m_R_prolonged, m_ERL_recovered,
      "z_star_star_recovered");

  // ---
  // COMPUTE ADJOINT DIFFS FOR WEIGHTING THE RESIDUAL
  // ---

  print("computing z_fine difference");
  m_z_fine_diff = physics->diff(m_z_fine, "z_fine_diff");
  print("computing z_star_star difference");
  m_z_star_star_diff = physics->diff(m_z_star_star, "z_star_star_diff");
  print("computing z_star_star_recovered difference");
  m_z_star_star_recovered_diff = physics->diff(m_z_star_star_recovered, "z_star_star_recovered_diff");

  // ---
  // LOCALIZE DIFFERENT ERROR CONTRIBUTIONS
  // ---
  
  print("localizing eta_1");
  m_eta1_local = physics->localize(m_R_prolonged, m_z_fine_diff, "eta1_local");
  print("localizing eta");
  m_eta_local = physics->localize(m_R_prolonged, m_z_star_star_diff, "eta_local");
  print("localizing eta_tilde");
  m_eta_tilde_local = physics->localize(m_R_prolonged, m_z_star_star_recovered_diff, "eta_tilde_local");

  // ---
  // COMPUTABLE ERROR CONTRIBUTIONS
  // ---

  print("computing error contributions");
  int const num_elems = get_nelems(physics);
  int const num_coarse_dofs = get_ndofs(COARSE, physics);
  int const num_fine_dofs = get_ndofs(FINE, physics);
  double const eta1 = -(physics->dot(m_z_fine, m_R_prolonged));
  double const eta2 = -(physics->dot(m_y_exact, m_R_prolonged));
  double const eta3 = -(physics->dot(m_z_fine, m_ERL_exact));
  double const eta4 = -(physics->dot(m_y_exact, m_ERL_exact));
  double const eta1_tilde = -(physics->dot(m_z_recovered, m_R_prolonged));
  double const eta2_tilde = -(physics->dot(m_y_recovered, m_R_prolonged));
  double const eta3_tilde = -(physics->dot(m_z_fine, m_ERL_recovered));
  double const eta4_tilde = -(physics->dot(m_y_recovered, m_ERL_recovered));
  double const eta = -(physics->dot(m_z_star_star, m_R_prolonged));
  double const eta_tilde = -(physics->dot(m_z_star_star_recovered, m_R_prolonged));

  // ---
  // ITERATION SUMMARY
  // ---

  print("summary for this adaptive iteration");
  print("> ---");
  print("> elems = %d", num_elems);
  print("> H_dofs = %d", num_coarse_dofs);
  print("> h_dofs = %d", num_fine_dofs);
  print("> ---");
  print("> JH = %.15e", J_coarse);
  print("> Jh = %.15e", J_fine);
  print("> Jr = %.15e", J_recovered);
  print("> Jeh = %.15e", Jeh);
  print("> ---");
  print("> eta = %.15e", eta);
  print("> eta_tilde = %.15e", eta_tilde);
  print("> ---");
  print("> eta1 = %.15e", eta1);
  print("> eta2 = %.15e", eta2);
  print("> eta3 = %.15e", eta3);
  print("> eta4 = %.15e", eta4);
  print("> eta1_tilde = %.15e", eta1_tilde);
  print("> eta2_tilde = %.15e", eta2_tilde);
  print("> eta3_tilde = %.15e", eta3_tilde);
  print("> eta4_tilde = %.15e", eta4_tilde);
  print("> ---");

  m_elems.push_back(num_elems);
  m_H_dofs.push_back(num_coarse_dofs);
  m_h_dofs.push_back(num_fine_dofs);
  m_JH.push_back(J_coarse);
  m_Jh.push_back(J_fine);
  m_eta.push_back(eta);
  m_eta_tilde.push_back(eta_tilde);
  m_eta1.push_back(eta1);
  m_eta2.push_back(eta2);
  m_eta3.push_back(eta3);
  m_eta4.push_back(eta4);
  m_eta1_tilde.push_back(eta1);
  m_eta2_tilde.push_back(eta2);
  m_eta3_tilde.push_back(eta3);
  m_eta4_tilde.push_back(eta4);

  // ---
  // INTERPOLATE THE CHOSEN ERROR FIELD TO CELL CENTERS
  // ---

  apf::Field* e = interp_error_to_cells(m_eta_local);

  return e;

}

void Adjoint::write_history(std::string const& file, double J_ex) {
  std::cout << std::scientific << std::setprecision(15);
  for (int i = 0; i < m_JH.size(); ++i) {
    std::cout << m_H_dofs[i] << ", " << J_ex << ", " << m_JH[i] << "\n";
  }
  (void)file;
  (void)J_ex;
}

void Adjoint::destroy_intermediate_fields() {
  apf::destroyField(m_u_coarse); m_u_coarse = nullptr;
  apf::destroyField(m_u_fine); m_u_fine = nullptr;
  apf::destroyField(m_u_prolonged); m_u_prolonged = nullptr;
  apf::destroyField(m_u_recovered); m_u_recovered = nullptr;
  apf::destroyField(m_ue_exact); m_ue_exact = nullptr;
  apf::destroyField(m_ue_recovered); m_ue_recovered = nullptr;
  apf::destroyField(m_z_coarse); m_z_coarse = nullptr;
  apf::destroyField(m_z_fine); m_z_fine = nullptr;
  apf::destroyField(m_z_recovered); m_z_recovered = nullptr;
  apf::destroyField(m_y_exact); m_y_exact = nullptr;
  apf::destroyField(m_y_recovered); m_y_recovered = nullptr;
  apf::destroyField(m_ERL_exact); m_ERL_exact = nullptr;
  apf::destroyField(m_ERL_recovered); m_ERL_recovered = nullptr;
  apf::destroyField(m_u_star); m_u_star = nullptr;
  apf::destroyField(m_z_star); m_z_star = nullptr;
  apf::destroyField(m_z_star_star); m_z_star_star = nullptr;
  apf::destroyField(m_u_star_recovered); m_u_star_recovered = nullptr;
  apf::destroyField(m_z_star_recovered); m_z_star_recovered = nullptr;
  apf::destroyField(m_z_star_star_recovered); m_z_star_star_recovered = nullptr;
  apf::destroyField(m_R_prolonged); m_R_prolonged = nullptr;
  apf::destroyField(m_z_fine_diff); m_z_fine_diff = nullptr;
  apf::destroyField(m_z_star_star_diff); m_z_star_star_diff = nullptr;
  apf::destroyField(m_z_star_star_recovered_diff); m_z_star_star_recovered_diff = nullptr;
  apf::destroyField(m_eta1_local); m_eta1_local = nullptr;
  apf::destroyField(m_eta_local); m_eta_local = nullptr;
  apf::destroyField(m_eta_tilde_local); m_eta_tilde_local = nullptr;
}

}
