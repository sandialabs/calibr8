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
  m_uh_spr = physics->recover_u_fine_from_u_coarse(m_uH);
  m_uh_minus_m_uH_h = physics->subtract_u_coarse_from_u_fine(m_uh, m_uH_h);
  m_uh_spr_minus_m_uH_h = physics->subtract_u_coarse_from_u_spr(m_uh_spr, m_uH_h);
}

void Adjoint::solve_adjoint(RCP<Physics> physics) {
  m_zH = physics->solve_adjoint(COARSE, m_uH);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
}

void Adjoint::post_process_adjoint(RCP<Physics> physics) {
  m_zH_h = physics->prolong_z_coarse_onto_fine(m_zH);
  m_zh_H = physics->restrict_z_fine_onto_fine(m_zh);
  m_zh_spr = physics->recover_z_fine_from_z_coarse(m_zH);
  m_zh_minus_m_zh_H = physics->subtract_z_coarse_from_z_fine(m_zh, m_zh_H);
  m_zh_spr_minus_m_zH_h = physics->subtract_z_coarse_from_z_spr(m_zh_spr, m_zH_h);
}

void Adjoint::compute_first_order_errors(RCP<Physics> physics) {
  m_Rh_uH_h = physics->evaluate_fine_residual_at_coarse_u(m_uH_h);
  double const eta1 = physics->dot(
      m_zh, m_Rh_uH_h, "eta1", "zh.Rh(uH_h)");
  double const eta2 = physics->dot(
      m_zh_minus_m_zh_H, m_Rh_uH_h, "eta2", "(zh-zH).Rh(uH_h)");
  double const eta1_spr = physics->dot(
      m_zh_spr, m_Rh_uH_h, "eta1_spr", "zh_spr.Rh(uH_h)");
  double const eta2_spr = physics->dot(
      m_zh_spr_minus_m_zH_h, m_Rh_uH_h, "eta2_spr", "(zh_spr-zH).Rh(uH_h)");
  m_eta1.push_back(eta1);
  m_eta2.push_back(eta2);
  m_eta1_spr.push_back(eta1_spr);
  m_eta2_spr.push_back(eta2_spr);
}

void Adjoint::compute_residual_linearization_errors(RCP<Physics> physics) {
  m_ERL_h = physics->compute_residual_linearization_error(
      m_uH_h, m_uh_minus_m_uH_h, "");
  double const etaL = physics->dot(m_zh, m_ERL_h, "etaL", "zh.E^R_L");
  m_ERL_h_spr = physics->compute_residual_linearization_error(
      m_uH_h, m_uh_spr_minus_m_uH_h, "spr");
  double const etaL_spr = physics->dot(m_zh, m_ERL_h_spr, "etaL_spr", "zh.E^R_L");
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
  compute_first_order_errors(physics);
  compute_residual_linearization_errors(physics);




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
  apf::destroyField(m_uh_spr); m_uh_spr = nullptr;
  apf::destroyField(m_uh_minus_m_uH_h); m_uh_minus_m_uH_h = nullptr;
  apf::destroyField(m_uh_spr_minus_m_uH_h); m_uh_spr_minus_m_uH_h = nullptr;
  apf::destroyField(m_zH); m_zH = nullptr;
  apf::destroyField(m_zh); m_zh = nullptr;
  apf::destroyField(m_zh_H); m_zh_H = nullptr;
  apf::destroyField(m_zH_h); m_zH_h = nullptr;
  apf::destroyField(m_zh_spr); m_zh_spr = nullptr;
  apf::destroyField(m_zh_minus_m_zh_H); m_zh_minus_m_zh_H = nullptr;
  apf::destroyField(m_zh_spr_minus_m_zH_h); m_zh_spr_minus_m_zH_h = nullptr;
  apf::destroyField(m_Rh_uH_h); m_Rh_uH_h = nullptr;
  apf::destroyField(m_ERL_h); m_ERL_h = nullptr;
  apf::destroyField(m_ERL_h_spr); m_ERL_h_spr = nullptr;
}

#if 0
void Adjoint::compute_linearization_error(RCP<Physics> physics, double& eta_L) {
  if (!linearization) return;
  m_uh_minus_uH_h = physics->subtract_u_coarse_from_u_fine(m_uh, m_uH_h);
  m_ELh = physics->compute_linearization_error(m_uH_h, m_uh_minus_uH_h);
  eta_L = physics->compute_eta_L(m_zh, m_ELh);
  m_Rh_uH_h_plus_ELh = physics->add_R_fine_to_EL_fine(m_Rh_uH_h, m_ELh);
}

void Adjoint::localize_error(RCP<Physics> physics) {
  if (localization == SIMPLE) {
    if (linearization) {
      apf::Field* m_eta1 = physics->localize_error(m_Rh_uH_h, m_z_weight, 1);
      apf::Field* m_eta2 = physics->localize_error(m_ELh, m_zh, 2);
      m_eta = physics->localize_linearization_error(m_eta1, m_eta2);
      apf::destroyField(m_eta1);
      apf::destroyField(m_eta2);
    } else {
      m_eta = physics->localize_error(m_Rh_uH_h, m_z_weight);
    }
  }
  if (localization == PU) {
    m_eta = physics->compute_eta2(m_uH_h, m_z_weight);
  }
}

void Adjoint::compute_error(RCP<Physics> physics, double& eta, double& eta_bound) {
  eta = physics->estimate_error(m_eta);
  eta_bound = physics->estimate_error_bound(m_eta);
  double const check = physics->estimate_error2(m_Rh_uH_h, m_z_weight);
  double const eta_diff = std::abs(eta - check);
  print(" > eta_diff = %.15e", eta_diff);
}

void Adjoint::collect_data(
    RCP<Physics> physics,
    double JH,
    double Jh,
    double eta,
    double eta_bound,
    double eta_L) {
  m_nelems.push_back(get_nelems(physics));
  m_H_dofs.push_back(get_ndofs(COARSE, physics));
  m_h_dofs.push_back(get_ndofs(FINE, physics));
  m_JH.push_back(JH);
  m_Jh.push_back(Jh);
  m_estimate.push_back(eta);
  m_estimate_bound.push_back(eta_bound);
  if (linearization) m_estimate_L.push_back(eta_L);
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {
  double JH(0), Jh(0), eta(0), eta_bound(0), eta_L(0);
  solve_primal(physics, JH, Jh);
  solve_adjoint(physics);
  compute_adjoint_weight(physics);
  compute_linearization_error(physics, eta_L);
  localize_error(physics);
  compute_error(physics, eta, eta_bound);
  collect_data(physics, JH, Jh, eta, eta_bound, eta_L);
  if (!subtraction) m_z_weight = nullptr;
  return interp_error_to_cells(m_eta);
}

void Adjoint::write_history(std::string const& file, double J_ex) {
  std::stringstream stream;
  stream << std::scientific << std::setprecision(16);
  stream << "elems H_dofs h_dofs JH Jh eta eta_bound Eh Ih Iboundh ";
  if (linearization) {
    stream << "eta_L ";
  }
  if (J_ex != 0.0) {
    stream << "E I Ibound";
  }
  stream << "\n";
  for (size_t ctr = 0; ctr < m_nelems.size(); ++ctr) {
    double const Eh = m_Jh[ctr] - m_JH[ctr];
    double const Ih = m_estimate[ctr] / Eh;
    double const Iboundh = m_estimate_bound[ctr] / std::abs(Eh);
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
    if (linearization) {
      stream << m_estimate_L[ctr] << " ";
    }
    if (J_ex != 0.0) {
      double const E = J_ex - m_JH[ctr];
      double const I = m_estimate[ctr] / E;
      double const Ibound = m_estimate_bound[ctr] / std::abs(E);
      stream
        << E << " "
        << I << " "
        << Ibound << " ";
    }
    stream << "\n";
  }
  write_stream(file + "/error.dat", stream);
}
#endif

}
