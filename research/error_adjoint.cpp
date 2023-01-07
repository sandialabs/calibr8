#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

enum {SIMPLE, PU};
enum {SPR, FULL, BOTH};

static void check_inputs(bool linearization, int adjoint, int localization) {
  if (linearization) {
    if (adjoint != FULL) {
      throw std::runtime_error("linearization must be done with adjoint=full");
    }
    if (localization != SIMPLE) {
      throw std::runtime_error("linearization must be done with localization=simple");
    }
  }
}

Adjoint::Adjoint(ParameterList const& params) {
  std::string const atype = params.get<std::string>("adjoint");
  std::string const ltype = params.get<std::string>("localization");
  subtraction = params.get<bool>("subtraction");
  linearization = params.get<bool>("linearization");
  if (ltype == "simple") localization = SIMPLE;
  else if (ltype == "PU") localization = PU;
  else throw std::runtime_error("invalid localization");
  if (atype == "full") adjoint = FULL;
  else if (atype == "spr") adjoint = SPR;
  else throw std::runtime_error("invalid adjoint");
  check_inputs(linearization, adjoint, localization);
}

void Adjoint::solve_primal(RCP<Physics> physics, double& JH, double& Jh) {
  m_uH = physics->solve_primal(COARSE);
  m_uh = physics->solve_primal(FINE);
  JH = physics->compute_qoi(COARSE, m_uH);
  Jh = physics->compute_qoi(FINE, m_uh);
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_Rh_uH_h = physics->evaluate_residual(FINE, m_uH_h);
}

void Adjoint::solve_adjoint(RCP<Physics> physics) {
  if (adjoint == SPR) {
    m_zH = physics->solve_adjoint(COARSE, m_uH);
    m_zh_spr = physics->recover_z_fine_from_z_coarse(m_zH);
  }
  if (adjoint == FULL) {
    m_zh = physics->solve_adjoint(FINE, m_uH_h);
  }
  if (adjoint == BOTH) {
    m_zH = physics->solve_adjoint(COARSE, m_uH);
    m_zh = physics->solve_adjoint(FINE, m_uH_h);
  }
}

void Adjoint::compute_adjoint_weight(RCP<Physics> physics) {
  if (!subtraction) {
    if (adjoint == SPR) m_z_weight = m_zh_spr;
    if (adjoint == FULL) m_z_weight = m_zh;
    if (adjoint == BOTH) m_z_weight = m_zh;
  } else {
    if (adjoint == SPR) {
      m_zh_H_spr = physics->restrict_z_fine_onto_fine(m_zh_spr);
      m_z_weight = physics->subtract_z_coarse_from_z_fine(m_zh_spr, m_zh_H_spr);
    }
    if (adjoint == FULL) {
      m_zh_H = physics->restrict_z_fine_onto_fine(m_zh);
      m_z_weight = physics->subtract_z_coarse_from_z_fine(m_zh, m_zh_H);
    }
    if (adjoint == BOTH) {
      m_zH_h = physics->prolong_z_coarse_onto_fine(m_zH);
      m_z_weight = physics->subtract_z_coarse_from_z_fine(m_zh, m_zH_h);
    }
  }
}

void Adjoint::localize_error(RCP<Physics> physics) {
  if (localization == SIMPLE) {
    if (linearization) {
      m_eta = physics->localize_error(m_Rh_uH_h_plus_ELh, m_z_weight);
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

void Adjoint::compute_linearization_error(RCP<Physics> physics, double& eta_L) {
  if (!linearization) return;
  m_uh_minus_uH_h = physics->subtract_u_coarse_from_u_fine(m_uh, m_uH_h);
  m_ELh = physics->compute_linearization_error(m_uH_h, m_uh_minus_uH_h);
  eta_L = physics->compute_eta_L(m_zh, m_ELh);
  m_Rh_uH_h_plus_ELh = physics->add_R_fine_to_EL_fine(m_Rh_uH_h, m_ELh);
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
  if (localization) m_estimate_L.push_back(eta_L);
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

void Adjoint::destroy_intermediate_fields() {
  if (m_uH) apf::destroyField(m_uH);
  if (m_uh) apf::destroyField(m_uh);
  if (m_uH_h) apf::destroyField(m_uH_h);
  if (m_uh_minus_uH_h) apf::destroyField(m_uh_minus_uH_h);
  if (m_zH) apf::destroyField(m_zH);
  if (m_zh) apf::destroyField(m_zh);
  if (m_zh_H) apf::destroyField(m_zh_H);
  if (m_zH_h) apf::destroyField(m_zH_h);
  if (m_zh_spr) apf::destroyField(m_zh_spr);
  if (m_zh_H_spr) apf::destroyField(m_zh_H_spr);
  if (m_z_weight) apf::destroyField(m_z_weight);
  if (m_Rh_uH_h) apf::destroyField(m_Rh_uH_h);
  if (m_Rh_uH_h_plus_ELh) apf::destroyField(m_Rh_uH_h_plus_ELh);
  if (m_eta) apf::destroyField(m_eta);
  m_uH = nullptr;
  m_uh = nullptr;
  m_uH_h = nullptr;
  m_uh_minus_uH_h = nullptr;
  m_zH = nullptr;
  m_zh = nullptr;
  m_zh_H = nullptr;
  m_zH_h = nullptr;
  m_zh_spr = nullptr;
  m_zh_H_spr = nullptr;
  m_z_weight = nullptr;
  m_Rh_uH_h = nullptr;
  m_Rh_uH_h_plus_ELh = nullptr;
  m_eta = nullptr;
}

}
