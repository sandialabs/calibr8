#include <fstream>
#include <iomanip>
#include "disc.hpp"
#include "global_residual.hpp"
#include "load_mismatch.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "ree.h"

namespace calibr8 {

template <typename T>
LoadMismatch<T>::LoadMismatch(ParameterList const& params) {
  m_side_set = params.get<std::string>("side set");
  m_write_load = params.isParameter("load out file");
  m_read_load = params.isParameter("load input file");
  if (m_write_load) m_load_out_file = params.get<std::string>("load out file");
  if (m_read_load) m_load_in_file = params.get<std::string>("load input file");
  ALWAYS_ASSERT((m_write_load + m_read_load) == 1);
  if (m_read_load) {
    std::ifstream in_file(m_load_in_file);
    std::string line;
    while(getline(in_file, line)) {
      m_load_data.push_back(std::stod(line));
    }
  }
  m_has_normal_2D = params.isParameter("2D surface normal");
  if (m_has_normal_2D) {
    m_normal_2D =
        params.get<Teuchos::Array<double>>("2D surface normal").toVector();
    ALWAYS_ASSERT(m_normal_2D.size() == 2);
  }
}

template <typename T>
LoadMismatch<T>::~LoadMismatch() {
}

template <typename T>
void LoadMismatch<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;

  // initialize the surface mesh information
  if (!is_initd) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_faces;
    m_mapping.resize(disc->num_elem_sets());
    SideSet const& sides = disc->sides(m_side_set);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        m_mapping[es][elem] = -1;
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_faces);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_faces[down];
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              m_mapping[es][elem] = down;
            }
          }
        }
      }
    }
    is_initd = true;
  }

}

template <typename T>
T LoadMismatch<T>::compute_load(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input) {

  T load_pt = T(0.);

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the face to integrate over
  int const facet_id = m_mapping[elem_set][elem];
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id];

  // get quadrature information over the face
  int const q_order = 1;
  apf::MeshElement* me = apf::createMeshElement(mesh, face);
  int const num_qps = apf::countIntPoints(me, q_order);

  // numerically integrate the QoI over the face
  for (int pt = 0; pt < num_qps; ++pt) {

    // get the integration point info on the face
    apf::Vector3 iota_face;
    apf::getIntPoint(me, q_order, pt, iota_face);
    double const w = apf::getIntWeight(me, q_order, pt);
    double const dv = apf::getDV(me, iota_face);

    // map the integration point on the face parametric space to
    // the element parametric space
    apf::Vector3 iota_elem = boundaryToElementXi(
        mesh, face, elem_entity, iota_face);

    // interpolate the global variable solution to face integration point
    // we assume displacements are index 0 and pressure is index 1
    int const disp_idx = 0;
    int const pressure_idx = 1;
    global->interpolate(iota_elem);
    Tensor<T> const grad_u = global->grad_vector_x(disp_idx);

    // compute the stress
    Tensor<T> stress = local->cauchy(global);
    if (local->is_finite_deformation()) {
      Tensor<T> const I = minitensor::eye<T>(ndims);
      Tensor<T> const F = grad_u + I;
      Tensor<T> const F_inv = inverse(F);
      Tensor<T> const F_invT = transpose(F_inv);
      T const J = det(F);
      stress = J * stress * F_invT;
      int const z_stretch_idx = local->z_stretch_idx();
      if (z_stretch_idx > -1) {
        T const z_stretch = local->scalar_xi(z_stretch_idx);
        stress *= z_stretch;
      }
    }

    apf::Vector3 N(0., 0., 0.);

    if (ndims == 3) {
      N = ree::computeFaceOutwardNormal(mesh, elem_entity, face, iota_face);
    } else if (ndims == 2) {
      N[0] = m_normal_2D[0];
      N[1] = m_normal_2D[1];
    }

    // compute the normal load at the integration point
    for (int i = 0; i < ndims; ++i) {
      for (int j = 0; j < ndims; ++j) {
        load_pt += N[i] * stress(i, j) * N[j];
      }
    }

    // integrate the normal loadintegrate the normal load
    load_pt *= w * dv;

  }

  // clean up allocated memory
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);

  return load_pt;
}

template <typename T>
void LoadMismatch<T>::preprocess_finalize(int step) {
  double load_meas = 0.;
  if (m_read_load) {
    load_meas = m_load_data[step - 1];
  }
  m_total_load = PCU_Add_Double(m_total_load);

  if (m_write_load && PCU_Comm_Self() == 0) {
    std::ofstream out_file;
    if (step == 1) {
      out_file.open(m_load_out_file);
    } else {
      out_file.open(m_load_out_file, std::ios::app | std::ios::out);
    }
    out_file << std::scientific << std::setprecision(17);
    out_file << m_total_load << "\n";
    out_file.close();
  }

  m_load_mismatch = m_total_load - load_meas;
  // reset the total load
  m_total_load = 0.;
}

template <typename T>
void LoadMismatch<T>::postprocess(double& J) {
  J += 0.5 * std::pow(m_load_mismatch, 2) / PCU_Comm_Peers();
}

template <typename T>
void LoadMismatch<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id = m_mapping[elem_set][elem];
  if (facet_id < 0) return;

  T load = compute_load(elem_set, elem, global, local, iota_input);
  m_total_load += val(load);
}

template <>
void LoadMismatch<double>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<double>> global,
    RCP<LocalResidual<double>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  this->initialize_value_pt();

}

template <>
void LoadMismatch<FADT>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<FADT>> global,
    RCP<LocalResidual<FADT>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  this->initialize_value_pt();

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id = m_mapping[elem_set][elem];
  if (facet_id < 0) return;

  // weight the load by the mismatch
  FADT load = compute_load(elem_set, elem, global, local, iota_input);
  this->value_pt = m_load_mismatch * load;
}

template class LoadMismatch<double>;
template class LoadMismatch<FADT>;

}
