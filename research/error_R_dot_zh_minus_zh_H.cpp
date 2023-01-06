#include "control.hpp"
#include "error_R_dot_zh_minus_zh_H.hpp"
#include "physics.hpp"

namespace calibr8 {

apf::Field* R_dot_zh_minus_zh_H::compute_error(RCP<Physics> physics) {

  // solve the adjoint problem
  m_uH = physics->solve_primal(COARSE);
  m_uh = physics->solve_primal(FINE);
  double const JH = physics->compute_qoi(COARSE, m_uH);
  double const Jh = physics->compute_qoi(FINE, m_uh);
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
  m_zh_H = physics->restrict_z_fine_onto_fine(m_zh);
  m_zh_minus_zh_H = physics->subtract_z_coarse_from_z_fine(m_zh, m_zh_H);

  // do the error localization
  if (localization == SIMPLE) {
    print("using localization: SIMPLE");
    m_Rh_uH_h = physics->evaluate_residual(FINE, m_uH_h);
    m_eta = physics->localize_error(m_Rh_uH_h, m_zh_minus_zh_H);
  } else if (localization == PU) {
    print("using localization: PU");
    m_eta = physics->compute_eta2(m_uH_h, m_zh_minus_zh_H);
  } else {
    throw std::runtime_error("invalid localization type");
  }
  double const eta = physics->estimate_error(m_eta);
  double const eta_bound = physics->estimate_error_bound(m_eta);

  // collect the data
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));
  m_JH.push_back(JH);
  m_Jh.push_back(Jh);
  m_estimate.push_back(eta);
  m_estimate_bound.push_back(eta_bound);

  // return the localized error interpolated to cells
  return interp_error_to_cells(m_eta);

}

void R_dot_zh_minus_zh_H::write_history(std::string const& file, double J_ex) {
  std::stringstream stream;
  stream << std::scientific << std::setprecision(16);
  stream << "elems H_dofs h_dofs JH Jh eta eta_bound Eh Ih Iboundh ";
  if (J_ex != 0.0) {
    stream << " E I Ibound ";
  }
  stream << "\n";
  for (size_t ctr = 0; ctr < m_nelems.size(); ++ctr) {
    double const Eh = m_Jh[ctr] - m_JH[ctr];
    double const Ih = m_estimate[ctr] / Eh;
    double const Iboundh = m_estimate_bound[ctr] / Eh;
    stream
      << m_nelems[ctr] << " "
      << m_H_dofs[ctr] << " "
      << m_h_dofs[ctr] << " "
      << m_JH[ctr] << " "
      << m_Jh[ctr] << " "
      << m_estimate[ctr] << " "
      << m_estimate_bound[ctr] << " "
      << Eh << " "
      << Ih << " "
      << Iboundh << " ";
    if (J_ex != 0.0) {
      double const E = J_ex - m_JH[ctr];
      double const I = m_estimate[ctr] / E;
      double const Ibound = m_estimate_bound[ctr] / E;
      stream
        << E << " "
        << I << " "
        << Ibound << " ";
    }
    stream << "\n";
  }
  write_stream(file + "/error.dat", stream);
}

void R_dot_zh_minus_zh_H::destroy_intermediate_fields() {
  if (m_uH) apf::destroyField(m_uH);
  if (m_uh) apf::destroyField(m_uh);
  if (m_uH_h) apf::destroyField(m_uH_h);
  if (m_zh) apf::destroyField(m_zh);
  if (m_zh_H) apf::destroyField(m_zh_H);
  if (m_zh_minus_zh_H) apf::destroyField(m_zh_minus_zh_H);
  if (m_Rh_uH_h) apf::destroyField(m_Rh_uH_h);
  if (m_eta) apf::destroyField(m_eta);
  m_uH = nullptr;
  m_uh = nullptr;
  m_uH_h = nullptr;
  m_zh = nullptr;
  m_zh_H = nullptr;
  m_zh_minus_zh_H = nullptr;
  m_Rh_uH_h = nullptr;
  m_eta = nullptr;
}

}
