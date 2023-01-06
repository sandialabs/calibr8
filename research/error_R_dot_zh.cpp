#include "control.hpp"
#include "error_R_dot_zh.hpp"
#include "physics.hpp"

namespace calibr8 {

enum {NONE, FINE_RESTRICTED, COARSE_EXACT};

R_dot_zh::R_dot_zh(ParameterList const& params) {
  std::string const ltype = params.get<std::string>("localization");
  std::string const stype = params.get<std::string>("subtraction");
  if (ltype == "simple") localization = SIMPLE;
  else if (ltype == "PU") localization = PU;
  else throw std::runtime_error("invalid localization");
  if (stype == "none") subtraction = NONE;
  else if (stype == "zh_H") subtraction = FINE_RESTRICTED;
  else if (stype == "zH_h") subtraction = COARSE_EXACT;
}

apf::Field* R_dot_zh::compute_error(RCP<Physics> physics) {

  // solve the primal problem on both spaces
  // so that we have access to exact error information in the QoI
  m_uH = physics->solve_primal(COARSE);
  m_uh = physics->solve_primal(FINE);
  double const JH = physics->compute_qoi(COARSE, m_uH);
  double const Jh = physics->compute_qoi(FINE, m_uh);

  // solve the adjoint problem on the fine space
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
  m_Rh_uH_h = physics->evaluate_residual(FINE, m_uH_h);

  // determine what we will dot the residual vector with
  // for the purposes of error estimation. this will subtract
  // something from the original fine space adjoint solution
  if (subtraction == NONE) {
    m_z_weight = m_zh;
  }
  if (subtraction == FINE_RESTRICTED) {
    m_zh_H = physics->restrict_z_fine_onto_fine(m_zh);
    m_z_weight = physics->subtract_z_coarse_from_z_fine(m_zh, m_zh_H);
  }
  if (subtraction == COARSE_EXACT) {
    m_zH = physics->solve_adjoint(COARSE, m_uH);
    m_zH_h = physics->prolong_z_coarse_onto_fine(m_zH);
    m_z_weight = physics->subtract_z_coarse_from_z_fine(m_zh, m_zH_h);
  }

  // perform the error localization using the chosen method
  if (localization == SIMPLE) {
    m_eta = physics->localize_error(m_Rh_uH_h, m_z_weight);
  }
  if (localization == PU) {
    m_eta = physics->compute_eta2(m_uH_h, m_z_weight);
  }

  // compute the scalar QoI error estimate value and
  // bound from the localized estimates
  double const eta = physics->estimate_error(m_eta);
  double const eta_bound = physics->estimate_error_bound(m_eta);

  // estimate the scalar QoI error estimate a 2nd way by dotting two
  // vectors just to ensure that it is identical to the 1st way
  double const check = physics->estimate_error2(m_Rh_uH_h, m_z_weight);
  double const eta_diff = std::abs(eta - check);
  print(" > eta_diff = %.15e", eta_diff);

  // collect the data
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));
  m_JH.push_back(JH);
  m_Jh.push_back(Jh);
  m_estimate.push_back(eta);
  m_estimate_bound.push_back(eta_bound);

  // some trickery to avoid memory errors during
  // the destroy intermediate fields stage
  if (subtraction == NONE) m_z_weight = nullptr;

  // return the localized error
  return interp_error_to_cells(m_eta);

}

void R_dot_zh::write_history(std::string const& file, double J_ex) {
  std::stringstream stream;
  stream << std::scientific << std::setprecision(16);
  stream << "elems H_dofs h_dofs JH Jh eta eta_bound Eh Ih Iboundh ";
  if (J_ex != 0.0) {
    stream << " E I Ibound";
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

void R_dot_zh::destroy_intermediate_fields() {
  std::cout << "0\n";
  if (m_uH) apf::destroyField(m_uH);
  std::cout << "1\n";
  if (m_uh) apf::destroyField(m_uh);
  std::cout << "2\n";
  if (m_uH_h) apf::destroyField(m_uH_h);
  std::cout << "3\n";
  if (m_zH) apf::destroyField(m_zH);
  std::cout << "4\n";
  if (m_zh) apf::destroyField(m_zh);
  std::cout << "5\n";
  if (m_zh_H) apf::destroyField(m_zh_H);
  std::cout << "6\n";
  if (m_zH_h) apf::destroyField(m_zH_h);
  std::cout << "7\n";
  if (m_z_weight) apf::destroyField(m_z_weight);
  std::cout << "8\n";
  if (m_Rh_uH_h) apf::destroyField(m_Rh_uH_h);
  std::cout << "9\n";
  if (m_eta) apf::destroyField(m_eta);
  std::cout << "10\n";
  m_uH = nullptr;
  m_uh = nullptr;
  m_uH_h = nullptr;
  m_zH = nullptr;
  m_zh = nullptr;
  m_zh_H = nullptr;
  m_zH_h = nullptr;
  m_z_weight = nullptr;
  m_Rh_uH_h = nullptr;
  m_eta = nullptr;
  std::cout << "\b";
}

}
