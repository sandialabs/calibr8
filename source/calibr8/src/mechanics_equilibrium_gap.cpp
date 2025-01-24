#include <apf.h>
#include <apfMesh.h>
#include <ree.h>
#include "control.hpp"
#include "defines.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "mappings.hpp"
#include "material_params.hpp"
#include "mechanics_equilibrium_gap.hpp"

namespace calibr8 {

using minitensor::det;
using minitensor::inverse;
using minitensor::transpose;

/* from ree/reeResidualFunctionals.cc */
static apf::MeshEntity* getFaceOppVert(
    apf::Mesh* m, apf::MeshEntity* f, apf::MeshEntity* e)
{
  apf::Downward evs;
  int env = m->getDownward(e, 0, evs);
  apf::Downward fvs;
  int fnv = m->getDownward(f, 0, fvs);
  PCU_ALWAYS_ASSERT(env == 2 && fnv == 3);
  for (int i = 0; i < fnv; i++) {
    if (apf::findIn(evs, env, fvs[i]) == -1)
      return fvs[i];
  }
  return 0;
}

static apf::Vector3 computeOutwardEdgeNormal(
    apf::Mesh* mesh, apf::MeshEntity* face, apf::MeshEntity* edge)
{
  apf::Downward downward_nodes;
  int num_down_nodes = mesh->getDownward(edge, 0, downward_nodes);
  ALWAYS_ASSERT(num_down_nodes == 2);

  apf::Vector3 p0;
  mesh->getPoint(downward_nodes[0], 0, p0);
  apf::Vector3 p1;
  mesh->getPoint(downward_nodes[1], 0, p1);
  apf::Vector3 const p1_to_p0 = p1 - p0;
  apf::Vector3 const mid_point = p1_to_p0 * 0.5 + p0;

  auto opp_vert = getFaceOppVert(mesh, face, edge);
  apf::Vector3 opp_point;
  mesh->getPoint(opp_vert, 0, opp_point);

  apf::Vector3 const opp_to_mid_point = opp_point - mid_point;
  apf::Vector3 const z(0., 0., 1.);
  apf::Vector3 unit_normal = (apf::cross(p1_to_p0, z)).normalize();

  double dot_product = opp_to_mid_point * unit_normal;
  if (dot_product > 0.) {
    unit_normal = unit_normal * -1.;
  }
  return unit_normal;
}

template <typename T>
MechanicsEquilibriumGap<T>::MechanicsEquilibriumGap(
    ParameterList const& params,
    int ndims) {

  auto p = params;
  m_thickness = p.get<double>("thickness", 1.);

  int const num_residuals = 1;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "u";
  this->m_var_types[0] = VECTOR;
  this->m_num_eqs[0] = get_num_eqs(VECTOR, ndims);

  int const num_ip_sets = 1;
  this->m_ip_sets.resize(num_ip_sets);
  // quadrature order for each integration point set
  this->m_ip_sets[0] = 1;

  ALWAYS_ASSERT(p.isParameter("traction boundaries"));
  auto side_set_names = p.get<Teuchos::Array<std::string>>("traction boundaries");
  m_side_set_names = side_set_names.toVector();
  ALWAYS_ASSERT(m_side_set_names.size() >= 1);
}

template <typename T>
MechanicsEquilibriumGap<T>::~MechanicsEquilibriumGap() {}

template <typename T>
void MechanicsEquilibriumGap<T>::before_elems(
    RCP<Disc> disc,
    int mode,
    Array1D<apf::Field*> const& adjoint_fields)
{
  GlobalResidual<T>::before_elems(disc, mode, adjoint_fields);
  if (!m_mapping_is_initd) {
    m_mapping_is_initd = setup_side_sets_mapping(m_side_set_names, disc, m_mapping);
  }
}

template <typename T>
void MechanicsEquilibriumGap<T>::evaluate(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv,
    int ip_set) {

  // gather information from this class
  int const ndims = this->m_num_dims;
  int const nnodes = this->m_num_nodes;
  int const momentum_idx = 0;

  // coupled ip set (lowest quadrature order)
  ALWAYS_ASSERT(ip_set == 0);

  // compute stress measures
  RCP<GlobalResidual<T>> global = rcp(this, false);
  // Cauchy for these models is dev_cauchy
  Tensor<T> stress = local->cauchy(global);

  if (local->is_finite_deformation()) {
    // gather variables from this residual quantities
    Tensor<T> const grad_u = this->grad_vector_x(momentum_idx);

    // compute kinematic quantities
    Tensor<T> const I = minitensor::eye<T>(ndims);
    Tensor<T> const F = grad_u + I;
    Tensor<T> const F_inv = inverse(F);
    Tensor<T> const F_invT = transpose(F_inv);
    T const J = det(F);

    int const z_stretch_idx = local->z_stretch_idx();
    T const z_stretch = local->scalar_xi(z_stretch_idx);

    stress = z_stretch * J * stress * F_invT;
  }

  // compute the balance of linear momentum residual
  for (int n = 0; n < nnodes; ++n) {
    for (int i = 0; i < ndims; ++i) {
      for (int j = 0; j < ndims; ++j) {
        double const dbasis_dx = this->grad_weight(momentum_idx, n, i, j);
        this->R_nodal(momentum_idx, n, i) +=
          stress(i, j) * dbasis_dx * w * m_thickness * dv;
      }
    }
  }

  // get some info for the mapping array
  int const es = this->m_elem_set_idx;
  int const elem = this->m_elem_idx;

  // get the id of the side wrt element if this side is on a traction boundary
  // do not evaluate if the side is not on a traction boundary
  int const side_id = m_mapping[es][elem];
  if (side_id < 0) return;

  // store some information contained in this class as local variables
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the side to integrate over
  apf::Downward elem_sides;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_sides);
  apf::MeshEntity* side = elem_sides[side_id];

  // get quadrature information over the side
  int const q_order = 1;
  apf::MeshElement* me_side = apf::createMeshElement(mesh, side);

  // should be 1; we assume constant stress over the element
  int const num_qps = apf::countIntPoints(me_side, q_order);

  // numerically integrate the QoI over the side
  for (int pt = 0; pt < num_qps; ++pt) {

    // get the integration point info on the side
    apf::Vector3 iota_side;
    apf::getIntPoint(me_side, q_order, pt, iota_side);
    double const w_side = apf::getIntWeight(me_side, q_order, pt);
    double const dv_side = apf::getDV(me_side, iota_side);

    // compute normal
    apf::Vector3 const unit_normal = computeOutwardEdgeNormal(mesh, elem_entity, side);
    apf::NewArray<double> BF;
    apf::getBF(this->m_shape, me_side, iota_side, BF);

    /* compute the traction part of the residual */
    for (int n = 0; n < nnodes; ++n) {
      for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ndims; ++j) {
          this->R_nodal(momentum_idx, n, i) -= BF[n]
              * stress(i, j) * unit_normal[j]
              * m_thickness * w_side * dv_side;
        }
      }
    }
  }
  apf::destroyMeshElement(me_side);
}

template <typename T>
void MechanicsEquilibriumGap<T>::after_elems()
{
  GlobalResidual<T>::after_elems();
  this->m_elem_set_idx = -1;
  this->m_elem_idx = -1;
}

template class MechanicsEquilibriumGap<double>;
template class MechanicsEquilibriumGap<FADT>;
template class MechanicsEquilibriumGap<DFADT>;

}
