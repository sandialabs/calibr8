#include <fstream>
#include <iomanip>
#include "calibration.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "ree.h"

namespace calibr8 {

template <typename T>
Calibration<T>::Calibration(ParameterList const& params) {
  m_balance_factor = params.get<double>("balance factor");
  m_side_set_disp = params.get<std::string>("displacement side set");
  m_side_set_load = params.get<std::string>("load side set");
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
}

template <typename T>
Calibration<T>::~Calibration() {
}

template <typename T>
void Calibration<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;

  //TODO: make this a general function in QoI?

  // initialize the surface mesh information
  if (!is_initd_disp) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_faces;
    m_mapping_disp.resize(disc->num_elem_sets());
    SideSet const& sides = disc->sides(m_side_set_disp);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping_disp[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        m_mapping_disp[es][elem] = -1;
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_faces);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_faces[down];
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              m_mapping_disp[es][elem] = down;
            }
          }
        }
      }
    }
    is_initd_disp = true;
  }

  // initialize the surface mesh information
  if (!is_initd_load) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_faces;
    m_mapping_load.resize(disc->num_elem_sets());
    SideSet const& sides = disc->sides(m_side_set_load);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping_load[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        m_mapping_load[es][elem] = -1;
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_faces);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_faces[down];
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              m_mapping_load[es][elem] = down;
            }
          }
        }
      }
    }
    is_initd_load = true;
  }

}

template <typename T>
T Calibration<T>::compute_load(
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
  int const facet_id = m_mapping_load[elem_set][elem];
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
    T const p = global->scalar_x(pressure_idx);

    // compute the stress
    Tensor<T> stress = local->cauchy(global, p);
    if (local->is_finite_deformation()) {
      Tensor<T> const I = minitensor::eye<T>(ndims);
      Tensor<T> const F = grad_u + I;
      Tensor<T> const F_inv = inverse(F);
      Tensor<T> const F_invT = transpose(F_inv);
      T const J = det(F);
      stress = J * stress * F_invT;
    }

    apf::Vector3 N = ree::computeFaceOutwardNormal(mesh, elem_entity, face,
        iota_face);

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
void Calibration<T>::preprocess_finalize(int step) {
  if (m_write_load) {
    std::ofstream out_file;
    out_file.open(m_load_out_file, std::ios::app | std::ios::out);
    out_file << std::scientific << std::setprecision(17);
    out_file << m_total_load << "\n";
    out_file.close();
  }
  double load_meas = 0.;
  if (m_read_load) {
    load_meas = m_load_data[step - 1];
  }
  //print("meas load at step %d = %.16e", step, load_meas);
  m_load_mismatch = m_total_load - load_meas;
  // reset the total load
  m_total_load = 0.;
}

template <typename T>
void Calibration<T>::postprocess(double& J) {
  //print("J_disp = %.16e", J);
  double J_forc = 0.5 * m_balance_factor * std::pow(m_load_mismatch, 2);
  //print("J_forc = %.16e", J_forc);
  J += J_forc;
}

template <typename T>
void Calibration<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id_load = m_mapping_load[elem_set][elem];
  if (facet_id_load < 0) return;

  T load = compute_load(elem_set, elem, global, local, iota_input);
  m_total_load += val(load);
}

template <>
void Calibration<double>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<double>> global,
    RCP<LocalResidual<double>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  this->value_pt = 0.;

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id_disp = m_mapping_disp[elem_set][elem];
  if (facet_id_disp < 0) return;

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the field for the measured displacement data at the step
  std::string name = "measured_" + std::to_string(step);
  apf::Field* f_meas = mesh->findField(name.c_str());
  ALWAYS_ASSERT(f_meas);
  apf::Element* e_meas = createElement(f_meas, mesh_elem);

  // grab the face to integrate over
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id_disp];

  // get quadrature information over the face
  int const q_order = 2;
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
    // we assume displacements are index 0
    int const disp_idx = 0;
    global->interpolate(iota_elem);
    Vector<double> const u_fem = global->vector_x(disp_idx);

    // interpolate the measured displacement data to the point
    apf::Vector3 u_meas;
    apf::getVector(e_meas, iota_elem, u_meas);

    // compute the QoI contribution at the point
    double const qoi =
      (u_fem[0] - u_meas[0]) * (u_fem[0] - u_meas[0]) +
      (u_fem[1] - u_meas[1]) * (u_fem[1] - u_meas[1]) +
      (u_fem[2] - u_meas[2]) * (u_fem[2] - u_meas[2]);

    // compute the difference between the FEM displacement and
    // the measured input displacement data
    this->value_pt += qoi * w * dv;

  }

  // clean up allocated memory
  apf::destroyElement(e_meas);
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);

}

template <>
void Calibration<FADT>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<FADT>> global,
    RCP<LocalResidual<FADT>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  // initialize the QoI contribution to 0
  FADT const dummy1 = global->vector_x(0)[0];
  FADT const dummy2 = local->first_value();
  Array2D<int> const& active_indices = local->active_indices();
  FADT const dummy3 = local->params(active_indices[0][0]);
  this->value_pt = 0. * (dummy1 + dummy2 + dummy3);

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id_disp = m_mapping_disp[elem_set][elem];
  int const facet_id_load = m_mapping_load[elem_set][elem];
  if (facet_id_disp + facet_id_load == -2) return;

  if (facet_id_disp > -1) {

    // store some information contained in this class as local variables
    int const ndims = this->m_num_dims;
    int const step = this->m_step;
    apf::Mesh* mesh = this->m_mesh;
    apf::MeshElement* mesh_elem = this->m_mesh_elem;

    // grab the field for the measured displacement data at the step
    std::string name = "measured_" + std::to_string(step);
    apf::Field* f_meas = mesh->findField(name.c_str());
    ALWAYS_ASSERT(f_meas);
    apf::Element* e_meas = createElement(f_meas, mesh_elem);

    // grab the face to integrate over
    apf::Downward elem_faces;
    apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
    mesh->getDownward(elem_entity, ndims - 1, elem_faces);
    apf::MeshEntity* face = elem_faces[facet_id_disp];

    // get quadrature information over the face
    int const q_order = 2;
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
      // we assume displacements are index 0
      int const disp_idx = 0;
      global->interpolate(iota_elem);
      Vector<FADT> const u_fem = global->vector_x(disp_idx);

      // interpolate the measured displacement data to the point
      apf::Vector3 u_meas;
      apf::getVector(e_meas, iota_elem, u_meas);

      // compute the QoI contribution at the point
      FADT const qoi =
        (u_fem[0] - u_meas[0]) * (u_fem[0] - u_meas[0]) +
        (u_fem[1] - u_meas[1]) * (u_fem[1] - u_meas[1]) +
        (u_fem[2] - u_meas[2]) * (u_fem[2] - u_meas[2]);
      
      // compute the difference between the FEM displacement and
      // the measured input displacement data
      this->value_pt += qoi * w * dv;

    }

    // clean up allocated memory
    apf::destroyElement(e_meas);
    apf::destroyMeshElement(me);

    // reset the state in global residual to what it was on input
    global->interpolate(iota_input);

  }

  if (facet_id_load > -1) {
    // weight the load by the mismatch
    FADT load = compute_load(elem_set, elem, global, local, iota_input);
    this->value_pt += m_balance_factor * m_load_mismatch * load;
  }

}

template class Calibration<double>;
template class Calibration<FADT>;

}
