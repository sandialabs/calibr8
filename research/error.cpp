#include <spr.h>
#include "control.hpp"
#include "error.hpp"
#include "physics.hpp"

namespace calibr8 {

void Error::write_mesh(RCP<Physics> physics, std::string const& file, int ctr) {
  std::string names[NUM_SPACE];
  names[COARSE] = file + "/" + "coarse_" + std::to_string(ctr);
  names[FINE] = file + "/" + "fine_" + std::to_string(ctr);
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  for (int space = 0; space < NUM_SPACE; ++space) {
    apf::FieldShape* shape = disc->shape(space);
    mesh->changeShape(shape, true);
    apf::writeVtkFiles(names[space].c_str(), mesh);
  }
}

void Error::write_pvd(std::string const& file, int nctr) {
  (void)file;
  (void)nctr;
}

class SPR_Error : public Error {
  public:
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uH_value_type = nullptr;
    apf::Field* m_grad_uH = nullptr;
    apf::Field* m_grad_uH_star = nullptr;
  private:
    std::vector<int> m_dofs;
    std::vector<int> m_elems;
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

apf::Field* SPR_Error::compute_error(RCP<Physics> physics) {
  int const order = physics->disc()->order(COARSE);
  m_uH = physics->solve_primal(COARSE);
  m_uH_value_type = create_value_type_field(physics, m_uH);
  m_grad_uH = spr::getGradIPField(m_uH_value_type, "grad_uH", order);
  m_grad_uH_star = spr::recoverField(m_grad_uH);
  //TODO: compute the element error given these two terms
  return m_uH;
}

void SPR_Error::destroy_intermediate_fields() {
  apf::destroyField(m_uH);
  apf::destroyField(m_uH_value_type);
  apf::destroyField(m_grad_uH);
  apf::destroyField(m_grad_uH_star);
}

void SPR_Error::write_history(std::string const& file) {
  (void)file;
}

RCP<Error> create_error(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "SPR") {
    return rcp(new SPR_Error);
  } else {
    throw std::runtime_error("invalid residual");
  }
}



#if 0
  double norm_R, norm_E;
  apf::Field* uH = m_physics->solve_primal(COARSE);
  apf::Field* uh = m_physics->solve_primal(FINE);
  double const JH = m_physics->compute_qoi(COARSE, uH);
  double const Jh = m_physics->compute_qoi(FINE, uh);
  apf::Field* uH_h = m_physics->prolong_u_coarse_onto_fine(uH);
  apf::Field* zh = m_physics->solve_adjoint(FINE, uH_h);
  apf::Field* uh_minus_uH_h =
    m_physics->op(subtract, uh, uH_h, "uh-uH_h");
  apf::Field* zh_H = m_physics->restrict_z_fine_onto_fine(zh);
  apf::Field* zh_minus_zh_H =
    m_physics->op(subtract, zh, zh_H, "zh-zh_H");
  apf::Field* E_L = m_physics->compute_linearization_error(
      uH_h, uh_minus_uH_h, norm_R, norm_E);

  apf::Field* eta = 
  double eta_zh = m_physics->compute_eta(uH_h, zh);
  double eta_zh_minus_zh_H = m_physics->compute_eta(uH_h, zh_minus_zh_H);
  double eta_L = m_physics->compute_eta_L(zh, E_L);

  apf::writeVtkFiles("debug", m_physics->disc()->apf_mesh());

  apf::destroyField(E_L);
  apf::destroyField(uh_minus_uH_h);
  apf::destroyField(zh);
  apf::destroyField(uH_h);
  apf::destroyField(uh);
  apf::destroyField(uH);
#endif

}
