#include "control.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"

namespace calibr8 {

Adjoint::Adjoint(ParameterList const& params) {
  auto const& resid_params = params.sublist("residual");
  auto const& qoi_params = params.sublist("quantity of interest");
  auto const& error_params = params.sublist("error");
  double const alpha = resid_params.get<double>("alpha");
  double const beta = qoi_params.get<double>("beta");
  m_linear_physics = (alpha == 0.);
  m_linear_qoi = (beta == 1.);
  m_error_field = error_params.get<std::string>("field");
}

apf::Field* Adjoint::compute_error(RCP<Physics> physics) {

  print("solving coarse primal problem");
  m_u_coarse = physics->solve_primal(COARSE);
  print("solving fine primal problem");
  m_u_fine = physics->solve_primal(FINE);
  print("prolonging the coarse primal solution onto the fine space");
  m_u_prolonged = physics->prolong(m_u_coarse, "u_prolonged");

  print("computing the QoI on the coarse space");
  double const J_coarse = physics->compute_qoi(COARSE, m_u_coarse);
  print("computing the QoI on the fine space");
  double const J_fine = physics->compute_qoi(FINE, m_u_fine);
  double const Jeh = J_fine - J_coarse;

  print("computing the exact primal discretization error");
  m_ue = physics->subtract(m_u_fine, m_u_prolonged, "ue");

  print("solving fine adjoint problem");
  m_z_fine = physics->solve_adjoint(FINE, m_u_prolonged);

  print("computing the residual linearization error");
  m_ERL = physics->solve_ERL(m_u_prolonged, m_ue, "ERL");

  print("evaluating the residual at the prolonged solution");
  m_R_prolonged = physics->evaluate_residual(FINE, m_u_prolonged);

  print("solving modified adjoint problem");
  nonlinear_in in;
  in.u_coarse = m_u_prolonged;
  in.u_fine = m_u_fine;
  in.ue = m_ue;
  in.J_coarse = J_coarse;
  in.J_fine = J_fine;
  nonlinear_out out = physics->solve_nonlinear_adjoint(in);
  m_u_star = out.u_star;
  m_z_star = out.z_star;
  m_z_star_star = physics->modify_star(m_z_star, m_R_prolonged, m_ERL, "z_star_star");

  print("computing z_fine difference");
  m_z_fine_diff = physics->diff(m_z_fine, "z_fine_diff");
  print("computing z_star_star difference");
  m_z_star_star_diff = physics->diff(m_z_star_star, "z_star_star_diff");

  print("localizing eta1");
  m_eta1_local = physics->localize(m_u_prolonged, m_z_fine_diff, "eta1_local");
  print("localizing eta2");
  m_eta2_local = physics->localize(m_u_prolonged, m_z_star_star_diff, "eta2_local");

  print("computing error contributions");
  int const num_elems = get_nelems(physics);
  int const num_coarse_dofs = get_ndofs(COARSE, physics);
  int const num_fine_dofs = get_ndofs(FINE, physics);
  double const eta1 = -(physics->dot(m_z_fine, m_R_prolonged));
  double const eta2 = -(physics->dot(m_z_star_star, m_R_prolonged));
  double const etaR_L = -(physics->dot(m_z_fine, m_ERL));
  double const eta1_sum = physics->compute_sum(m_eta1_local);
  double const eta2_sum = physics->compute_sum(m_eta2_local);
  double const eta1_bound = physics->compute_bound(m_eta1_local);
  double const eta2_bound = physics->compute_bound(m_eta2_local);
  double const norm_ERL = std::sqrt(physics->dot(m_ERL, m_ERL));

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
  print("> Jeh = %.15e", Jeh);
  print("> ---");
  print("> eta1 = %.15e", eta1);
  print("> eta2 = %.15e", eta2);
  print("> etaR_L = %.15e", etaR_L);
  print("> eta1 sum = %.15e", eta1_sum);
  print("> eta2 sum = %.15e", eta2_sum);
  print("> eta1 bound = %.15e", eta1_bound);
  print("> eta2 bound = %.15e", eta2_bound);
  print("> ---");
  print("> ||E^R_L|| = %.15e", norm_ERL);

  m_elems.push_back(num_elems);
  m_H_dofs.push_back(num_coarse_dofs);
  m_h_dofs.push_back(num_fine_dofs);
  m_JH.push_back(J_coarse);
  m_Jh.push_back(J_fine);
  m_eta1.push_back(eta1);
  m_eta2.push_back(eta2);
  m_etaR_L.push_back(etaR_L);
  m_eta1_bound.push_back(eta1_bound);
  m_eta2_bound.push_back(eta2_bound);
  m_norm_ERL.push_back(norm_ERL);

  // ---
  // INTERPOLATE THE CHOSEN ERROR FIELD TO CELL CENTERS
  // ---

  apf::Field* e = nullptr;
  if      (m_error_field == "eta1") e = interp_error_to_cells(m_eta1_local);
  else if (m_error_field == "eta2") e = interp_error_to_cells(m_eta2_local);
  else {
    throw std::runtime_error("invalid error field: " + m_error_field);
  }
  return e;

}

static void write_stream(
    std::filesystem::path const& path,
    std::stringstream const& stream) {
  std::ofstream file_stream(path.c_str());
  if (!file_stream.is_open()) {
    throw std::runtime_error(
        "write_stream - could not open: " + path.string());
  }
  file_stream << stream.rdbuf();
  file_stream.close();
}

void Adjoint::write_history(std::string const& file, double J_ex) {
  std::stringstream stream;
  stream << std::scientific << std::setprecision(15);
  stream << "elems H_dofs h_dofs ";
  if (J_ex != 0.) stream << "J ";
  stream << "JH ";
  stream << "Jh ";
  stream << "eta1 ";
  stream << "eta2 ";
  stream << "etaR_L ";
  stream << "eta1_bound ";
  stream << "eta2_bound ";
  stream << "norm_ERL\n";
  for (size_t i = 0; i < m_elems.size(); ++i) {
    stream << m_elems[i] << " ";
    stream << m_H_dofs[i] << " ";
    stream << m_h_dofs[i] << " ";
    if (J_ex != 0.) stream << J_ex << " ";
    stream << m_JH[i] << " ";
    stream << m_Jh[i] << " ";
    stream << m_eta1[i] << " ";
    stream << m_eta2[i] << " ";
    stream << m_etaR_L[i] << " ";
    stream << m_eta1_bound[i] << " ";
    stream << m_eta2_bound[i] << " ";
    stream << m_norm_ERL[i] << "\n";
  }
  std::string const out = file + ".dat";
  write_stream(out, stream);
}

void Adjoint::destroy_intermediate_fields() {
  apf::destroyField(m_u_coarse); m_u_coarse = nullptr;
  apf::destroyField(m_u_fine); m_u_fine = nullptr;
  apf::destroyField(m_u_prolonged); m_u_prolonged = nullptr;
  apf::destroyField(m_ue); m_ue = nullptr;
  apf::destroyField(m_z_fine); m_z_fine = nullptr;
  apf::destroyField(m_ERL); m_ERL = nullptr;
  apf::destroyField(m_u_star); m_u_star = nullptr;
  apf::destroyField(m_z_star); m_z_star = nullptr;
  apf::destroyField(m_z_star_star); m_z_star_star = nullptr;
  apf::destroyField(m_R_prolonged); m_R_prolonged = nullptr;
  apf::destroyField(m_z_fine_diff); m_z_fine_diff = nullptr;
  apf::destroyField(m_z_star_star_diff); m_z_star_star_diff = nullptr;
  apf::destroyField(m_eta1_local); m_eta1_local = nullptr;
  apf::destroyField(m_eta2_local); m_eta2_local = nullptr;
}

}
