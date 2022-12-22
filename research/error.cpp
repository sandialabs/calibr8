#include <fstream>
#include <PCU.h>
#include <spr.h>
#include "control.hpp"
#include "error.hpp"
#include "physics.hpp"

namespace calibr8 {

static int const get_ndofs(int space, RCP<Physics> physics) {
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  int const neqs = disc->num_eqs();
  int dofs = neqs * disc->owned_nodes(space).getSize();
  dofs = PCU_Add_Double(dofs);
  return dofs;
}

static int const get_nelems(RCP<Physics> physics) {
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  int nelems = mesh->count(disc->num_dims());
  nelems = PCU_Add_Double(nelems);
  return nelems;
}

static void write_stream(
    std::string const& path,
    std::stringstream const& stream) {
  std::ofstream file_stream(path.c_str());
  if (!file_stream.is_open()) {
    throw std::runtime_error("write_stream - could not open: " + path);
  }
  file_stream << stream.rdbuf();
  file_stream.close();
}

void Error::write_mesh(RCP<Physics> physics, std::string const& file, int ctr) {
  std::string names[NUM_SPACE];
  names[COARSE] = file + "/" + "coarse_" + std::to_string(ctr);
  names[FINE] = file + "/" + "fine_" + std::to_string(ctr);
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  for (int space = 0; space < NUM_SPACE; ++space) {
    disc->change_shape(space);
    apf::writeVtkFiles(names[space].c_str(), mesh);
  }
}

static void write_pvd(
    std::stringstream& stream,
    std::string const& name,
    int nctr) {
  stream << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
  stream << "  <Collection>" << std::endl;
  for (int ctr = 1; ctr <= nctr; ++ctr) {
    std::string const out = name + "_" + std::to_string(ctr);
    std::string const pvtu = out + "/" + out;
    stream << "    <DataSet timestep=\"" << ctr << "\" group=\"\" ";
    stream << "part=\"0\" file=\"" << pvtu;
    stream << ".pvtu\"/>" << std::endl;
  }
  stream << "  </Collection>" << std::endl;
  stream << "</VTKFile>" << std::endl;
}

void Error::write_pvd(std::string const& file, int nctr) {
  if (PCU_Comm_Self()) return;
  std::stringstream coarse_stream;
  std::stringstream fine_stream;
  calibr8::write_pvd(coarse_stream, "coarse", nctr);
  calibr8::write_pvd(fine_stream, "fine", nctr);
  write_stream(file + "/" + "coarse.pvd", coarse_stream);
  write_stream(file + "/" + "fine.pvd", fine_stream);
}

apf::Field* interp_error_to_cells(apf::Field* eta) {
  print("interpolating error field to cell centers");
  apf::Mesh* mesh = apf::getMesh(eta);
  apf::Field* error = apf::createStepField(mesh, "error", apf::SCALAR);
  int const neqs = apf::countComponents(eta);
  Array1D<double> values(neqs, 0.);
  apf::MeshEntity* ent;
  apf::MeshIterator* elems = mesh->begin(mesh->getDimension());
  while ((ent = mesh->iterate(elems))) {
    apf::Vector3 xi;
    apf::MeshElement* me = apf::createMeshElement(mesh, ent);
    apf::Element* e = apf::createElement(eta, me);
    apf::getIntPoint(me, 1, 0, xi);
    apf::getComponents(e, xi, &(values[0]));
    double error_val = 0.;
    for (int eq = 0; eq < neqs; ++eq) {
      error_val += std::abs(values[eq]);
    }
    apf::setScalar(error, ent, 0, error_val);
    apf::destroyElement(e);
    apf::destroyMeshElement(me);
  }
  mesh->end(elems);
  return error;
}

class SPR : public Error {
  public:
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uH_value_type = nullptr;
    apf::Field* m_grad_uH = nullptr;
    apf::Field* m_grad_uH_star = nullptr;
  private:
    std::vector<int> m_ndofs;
    std::vector<int> m_nelems;
    std::vector<double> m_J;
};

static apf::Field* create_value_type_field(
    RCP<Physics> physics,
    apf::Field* u) {
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(COARSE);
  int const neqs = disc->num_eqs();
  int const ndims = disc->num_dims();
  int type = 0;
  if (neqs == 1) type = apf::SCALAR;
  else if ((neqs == 2) & (ndims == 2)) type = apf::VECTOR;
  else if ((neqs == 3) & (ndims == 3)) type = apf::VECTOR;
  else throw std::runtime_error("spr error - can't make field");
  apf::Field* u_vt = createField(mesh, "uH_vt", type, shape);
  apf::copyData(u_vt, u);
  return u_vt;
}

apf::Field* SPR::compute_error(RCP<Physics> physics) {
  RCP<Disc> disc = physics->disc();
  int const order = disc->order(COARSE);
  m_uH = physics->solve_primal(COARSE);
  double const J = physics->compute_qoi(COARSE, m_uH);
  m_uH_value_type = create_value_type_field(physics, m_uH);
  m_grad_uH = spr::getGradIPField(m_uH_value_type, "grad_uH", order);
  m_grad_uH_star = spr::recoverField(m_grad_uH);
  m_J.push_back(J);
  m_ndofs.push_back(get_ndofs(COARSE, physics));
  m_nelems.push_back(get_nelems(physics));
  //TODO: compute the element error given these two terms
  return m_uH;
}

void SPR::destroy_intermediate_fields() {
  apf::destroyField(m_uH);
  apf::destroyField(m_uH_value_type);
  apf::destroyField(m_grad_uH);
  apf::destroyField(m_grad_uH_star);
  m_uH = nullptr;
  m_uH_value_type = nullptr;
  m_grad_uH = nullptr;
  m_grad_uH_star = nullptr;
}

void SPR::write_history(std::string const& file, double J_ex) {
  //TODO: write history here
  (void)file;
  (void)J_ex;
}

enum {SIMPLE, PU};

class R_zh : public Error {
  public:
    R_zh(int ltype) : localization(ltype) {}
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uh = nullptr;
    apf::Field* m_uH_h = nullptr;
    apf::Field* m_zh = nullptr;
    apf::Field* m_Rh_uH_h = nullptr;
    apf::Field* m_eta = nullptr;
  private:
    int localization = -1;
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_estimate;
    std::vector<double> m_estimate_bound;
};

apf::Field* R_zh::compute_error(RCP<Physics> physics) {

  // solve the adjoint problem
  m_uH = physics->solve_primal(COARSE);
  m_uh = physics->solve_primal(FINE);
  double const JH = physics->compute_qoi(COARSE, m_uH);
  double const Jh = physics->compute_qoi(FINE, m_uh);
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);
  m_Rh_uH_h = physics->evaluate_residual(FINE, m_uH_h);

  // do the error localization
  if (localization == SIMPLE) {
    print("using localization: SIMPLE");
    m_eta = physics->localize_error(m_Rh_uH_h, m_zh);
  } else if (localization == PU) {
    print("using localization: PU");
    m_eta = physics->compute_eta2(m_uH_h, m_zh);
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

  // return the localized error
  return interp_error_to_cells(m_eta);

}

void R_zh::write_history(std::string const& file, double J_ex) {
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

void R_zh::destroy_intermediate_fields() {
  if (m_uH) apf::destroyField(m_uH);
  if (m_uh) apf::destroyField(m_uh);
  if (m_uH_h) apf::destroyField(m_uH_h);
  if (m_zh) apf::destroyField(m_zh);
  if (m_Rh_uH_h) apf::destroyField(m_Rh_uH_h);
  if (m_eta) apf::destroyField(m_eta);
  m_uH = nullptr;
  m_uh = nullptr;
  m_uH_h = nullptr;
  m_zh = nullptr;
  m_Rh_uH_h = nullptr;
  m_eta = nullptr;
}

class R_zh_minus_zh_H : public Error {
  public:
    R_zh_minus_zh_H(int ltype) : localization(ltype) {}
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uh = nullptr;
    apf::Field* m_uH_h = nullptr;
    apf::Field* m_zh = nullptr;
    apf::Field* m_zh_H = nullptr;
    apf::Field* m_zh_minus_zh_H = nullptr;
    apf::Field* m_Rh_uH_h = nullptr;
    apf::Field* m_eta = nullptr;
  private:
    int localization = -1;
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_estimate;
    std::vector<double> m_estimate_bound;
};

apf::Field* R_zh_minus_zh_H::compute_error(RCP<Physics> physics) {

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

void R_zh_minus_zh_H::write_history(std::string const& file, double J_ex) {
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

void R_zh_minus_zh_H::destroy_intermediate_fields() {
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

class R_plus_E_zh : public Error {
  public:
    R_plus_E_zh(int ltype) : localization(ltype) {}
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uh = nullptr;
    apf::Field* m_uH_h = nullptr;
    apf::Field* m_uh_minus_uH_h = nullptr;
    apf::Field* m_zh = nullptr;
    apf::Field* m_Rh_uH_h = nullptr;
    apf::Field* m_Rh_plus_ELh = nullptr;
    apf::Field* m_ELh = nullptr;
    apf::Field* m_eta = nullptr;
  private:
    int localization = -1;
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_eta_L;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_estimate;
    std::vector<double> m_estimate_bound;
};

apf::Field* R_plus_E_zh::compute_error(RCP<Physics> physics) {

  // solve the adjoint problem
  m_uH = physics->solve_primal(COARSE);
  m_uh = physics->solve_primal(FINE);
  double const JH = physics->compute_qoi(COARSE, m_uH);
  double const Jh = physics->compute_qoi(FINE, m_uh);
  m_uH_h = physics->prolong_u_coarse_onto_fine(m_uH);
  m_uh_minus_uH_h = physics->subtract_u_coarse_from_u_fine(m_uh, m_uH_h);
  m_zh = physics->solve_adjoint(FINE, m_uH_h);

  // compute the linearization error
  double norm_R, norm_E;
  m_ELh = physics->compute_linearization_error(
      m_uH_h,
      m_uh_minus_uH_h,
      norm_R,
      norm_E);
  double const eta_L = physics->compute_eta_L(m_zh, m_ELh);

  // do the error localization
  if (localization == SIMPLE) {
    print("using localization: SIMPLE");
    m_Rh_uH_h = physics->evaluate_residual(FINE, m_uH_h);
    m_Rh_plus_ELh = physics->add_R_fine_to_EL_fine(m_Rh_uH_h, m_ELh);
    m_eta = physics->localize_error(m_Rh_plus_ELh, m_zh);
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
  m_eta_L.push_back(eta_L);
  m_estimate_bound.push_back(eta_bound);

  // return the error interpolated to cells
  return interp_error_to_cells(m_eta);
}

void R_plus_E_zh::write_history(std::string const& file, double J_ex) {

  std::stringstream stream;
  stream << std::scientific << std::setprecision(16);
  stream << "elems H_dofs h_dofs JH Jh eta_L eta_1 eta eta_bound Eh Ih Iboundh ";
  if (J_ex != 0.0) {
    stream << " E I Ibound";
  }
  stream << "\n";
  for (size_t ctr = 0; ctr < m_nelems.size(); ++ctr) {
    double const Eh = m_Jh[ctr] - m_JH[ctr];
    double const Ih = m_estimate[ctr] / Eh;
    double const eta_1 = m_estimate[ctr] - m_eta_L[ctr];
    double const Iboundh = m_estimate_bound[ctr] / Eh;
    stream
      << m_nelems[ctr] << " "
      << m_H_dofs[ctr] << " "
      << m_h_dofs[ctr] << " "
      << m_JH[ctr] << " "
      << m_Jh[ctr] << " "
      << m_eta_L[ctr] << " "
      << eta_1 << " "
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

void R_plus_E_zh::destroy_intermediate_fields() {
  if (m_uH) apf::destroyField(m_uH);
  if (m_uh) apf::destroyField(m_uh);
  if (m_uH_h) apf::destroyField(m_uH_h);
  if (m_uh_minus_uH_h) apf::destroyField(m_uh_minus_uH_h);
  if (m_zh) apf::destroyField(m_zh);
  if (m_Rh_uH_h) apf::destroyField(m_Rh_uH_h);
  if (m_Rh_plus_ELh) apf::destroyField(m_Rh_plus_ELh);
  if (m_ELh) apf::destroyField(m_ELh); 
  if (m_eta) apf::destroyField(m_eta);
  m_uH = nullptr;
  m_uh = nullptr;
  m_uH_h = nullptr;
  m_uh_minus_uH_h = nullptr;
  m_zh = nullptr;
  m_Rh_uH_h = nullptr;
  m_Rh_plus_ELh = nullptr;
  m_ELh = nullptr;
  m_eta = nullptr;
}

RCP<Error> create_error(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  int ltype = -1;
  if (params.isType<std::string>("localization")) {
    std::string const localization = params.get<std::string>("localization");
    if (localization == "simple") ltype = SIMPLE;
    if (localization == "PU") ltype = PU;
  }
  if (type == "SPR") {
    return rcp(new SPR);
  } else if (type == "R dot zh") {
    return rcp(new R_zh(ltype));
  } else if (type == "R dot zh minus zh_H") {
    return rcp(new R_zh_minus_zh_H(ltype));
  } else if (type == "R plus E dot zh") {
    return rcp(new R_plus_E_zh(ltype));
  } else {
    throw std::runtime_error("invalid error");
  }
}

}
