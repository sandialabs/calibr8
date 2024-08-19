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
  m_coord_idx = params.get<int>("coordinate index");
  m_coord_value = params.get<double>("coordinate value");
  m_reaction_force_comp = params.get<int>("reaction force component");
  m_write_obj_at_step = params.isParameter("objective out file");
  m_write_load = params.isParameter("load out file");
  m_read_load = params.isParameter("load input file");
  m_has_weights = params.isParameter("displacement weights");
  if (m_has_weights) {
    m_weights =
        params.get<Teuchos::Array<double>>("displacement weights").toVector();
    int const num_weights = m_weights.size();
    ALWAYS_ASSERT((num_weights == 2) || (num_weights == 3));
  }
  m_has_distance_threshold = params.isParameter("distance threshold");
  if (m_has_distance_threshold) {
    m_distance_threshold = params.get<double>("distance threshold");
  }
  if (params.isParameter("displacement side set")) {
    m_side_set_disp = params.get<std::string>("displacement side set");
   }
  if (m_write_load) m_load_out_file = params.get<std::string>("load out file");
  if (m_read_load) m_load_in_file = params.get<std::string>("load input file");
  if (m_write_obj_at_step) m_obj_out_file = params.get<std::string>("objective out file");
  if (m_read_load) {
    std::ifstream in_file(m_load_in_file);
    std::string line;
    while (getline(in_file, line)) {
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

  int ndims = this->m_num_dims;

  if (!is_initd_disp) {
    m_area = 0.;
    apf::Field* f_dist = this->m_mesh->findField(m_distance_field_name.c_str());
    if (m_has_distance_threshold) ALWAYS_ASSERT(f_dist);
    Array1D<double> distance(1);
    apf::Vector3 iota;
    m_mapping_disp.resize(disc->num_elem_sets());
    double w = 0.;
    double dv = 0.;

    if (ndims == 2) {
      for (int es = 0; es < disc->num_elem_sets(); ++es) {
        std::string const& es_name = disc->elem_set_name(es);
        ElemSet const& elems = disc->elems(es_name);
        m_mapping_disp[es].resize(elems.size());
        for (size_t elem = 0; elem < elems.size(); ++elem) {
          m_mapping_disp[es][elem] = -1;
          apf::MeshEntity* elem_entity = elems[elem];
          apf::MeshElement* me = apf::createMeshElement(this->m_mesh, elem_entity);
          apf::getIntPoint(me, 1, 0, iota);
          if (f_dist) {
            apf::Element* e_dist = createElement(f_dist, me);
            apf::getComponents(e_dist, iota, &(distance[0]));
            if (distance[0] > m_distance_threshold) {
              m_mapping_disp[es][elem] = 1;
              w = apf::getIntWeight(me, 0, 0);
              dv = apf::getDV(me, iota);
              m_area += w * dv;
            }
            apf::destroyElement(e_dist);
          } else {
            m_mapping_disp[es][elem] = 1;
            w = apf::getIntWeight(me, 0, 0);
            dv = apf::getDV(me, iota);
            m_area += w * dv;
          }
          apf::destroyMeshElement(me);
        }
      }
    } else if (ndims == 3) {
      SideSet const& sides = disc->sides(m_side_set_disp);
      apf::Downward downward_faces;
      for (int es = 0; es < disc->num_elem_sets(); ++es) {
        std::string const& es_name = disc->elem_set_name(es);
        ElemSet const& elems = disc->elems(es_name);
        m_mapping_disp[es].resize(elems.size());
        for (size_t elem = 0; elem < elems.size(); ++elem) {
          m_mapping_disp[es][elem] = -1;
          apf::MeshEntity* elem_entity = elems[elem];
          int ndown = this->m_mesh->getDownward(elem_entity, ndims - 1, downward_faces);
          for (int down = 0; down < ndown; ++down) {
            apf::MeshEntity* downward_entity = downward_faces[down];
            for (apf::MeshEntity* side : sides) {
              if (side == downward_entity) {
                apf::MeshElement* me = apf::createMeshElement(this->m_mesh, side);
                apf::getIntPoint(me, 1, 0, iota);
                if (f_dist) {
                  apf::Element* e_dist = createElement(f_dist, me);
                  apf::getComponents(e_dist, iota, &(distance[0]));
                  if (distance[0] > m_distance_threshold) {
                    m_mapping_disp[es][elem] = down;
                    w = apf::getIntWeight(me, 0, 0);
                    dv = apf::getDV(me, iota);
                    m_area += w * dv;
                  }
                  apf::destroyElement(e_dist);
                } else {
                  m_mapping_disp[es][elem] = down;
                  w = apf::getIntWeight(me, 0, 0);
                  dv = apf::getDV(me, iota);
                  m_area += w * dv;
                }
                apf::destroyMeshElement(me);
              }
            }
          }
        }
      }
    }
    m_area = PCU_Add_Double(m_area);
    is_initd_disp = true;
  }

  if (!is_initd_load) {
    is_initd_load = this->setup_coord_based_node_mapping(m_coord_idx, m_coord_value,
        disc, m_mapping_load);
  }

  this->m_dt = disc->dt(step);
  if (!is_initd_total_time) {
    int const num_time_steps = disc->num_time_steps();
    m_total_time = disc->time(num_time_steps) - disc->time(0);
    is_initd_total_time = true;
  }
}

template <typename T>
T Calibration<T>::compute_disp_mismatch(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input) {

  T mismatch = T(0.);

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

  // get quadrature information over the face
  int const q_order = 2;
  int const num_qps = apf::countIntPoints(mesh_elem, q_order);

  // numerically integrate the QoI over the face
  for (int pt = 0; pt < num_qps; ++pt) {

    // get the integration point info on the face
    apf::Vector3 iota;
    apf::getIntPoint(mesh_elem, q_order, pt, iota);
    double const w = apf::getIntWeight(mesh_elem, q_order, pt);
    double const dv = apf::getDV(mesh_elem, iota);

    // interpolate the global variable solution to face integration point
    // we assume displacements are index 0
    int const disp_idx = 0;
    global->interpolate(iota);
    Vector<T> const u_fem = global->vector_x(disp_idx);

    // interpolate the measured displacement data to the point
    apf::Vector3 u_meas;
    apf::getVector(e_meas, iota, u_meas);

    // compute the QoI contribution at the point
    T qoi = 0.;
    for (int d = 0; d < ndims; ++d) {
      qoi += m_weights[d] * (u_fem[d] - u_meas[d]) * (u_fem[d] - u_meas[d]);
    }

    // compute the difference between the FEM displacement and
    // the measured input displacement data
    mismatch += 0.5 * qoi * w * dv / m_area * m_dt / m_total_time;
  }

  // clean up allocated memory
  apf::destroyElement(e_meas);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);

  return mismatch;
}

template <typename T>
T Calibration<T>::compute_surface_mismatch(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input) {

  T mismatch = T(0.);

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the face to integrate over
  int const facet_id = m_mapping_disp[elem_set][elem];
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id];

  // grab the field for the measured displacement data at the step
  std::string name = "measured_" + std::to_string(step);
  apf::Field* f_meas = mesh->findField(name.c_str());
  ALWAYS_ASSERT(f_meas);
  apf::Element* e_meas = createElement(f_meas, mesh_elem);

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
    Vector<T> const u_fem = global->vector_x(disp_idx);

    // interpolate the measured displacement data to the point
    apf::Vector3 u_meas;
    apf::getVector(e_meas, iota_elem, u_meas);

    // compute the QoI contribution at the point
    T qoi = 0.;
    for (int d = 0; d < ndims; ++d) {
      qoi += m_weights[d] * (u_fem[d] - u_meas[d]) * (u_fem[d] - u_meas[d]);
    }

    // compute the difference between the FEM displacement and
    // the measured input displacement data
    mismatch += 0.5 * qoi * w * dv / m_area * m_dt / m_total_time;

  }

  // clean up allocated memory
  apf::destroyElement(e_meas);
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);

  return mismatch;
}

template <typename T>
T Calibration<T>::compute_load(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  T load_pt = T(0.);

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);

  // get the nodes for the element
  apf::Downward elem_verts;
  int const num_elem_nodes = mesh->getDownward(elem_entity, 0, elem_verts);

  // grab the nodes for the reactions
  Array1D<int> const& node_ids = m_mapping_load[elem_set][elem];
  int const num_nodes = node_ids.size();

  // compute the internal force vector
  global->zero_residual();
  global->evaluate(local, iota, w, dv, 0);

  // sum relevant entries
  int const disp_idx = 0;
  for (size_t i = 0; i < num_nodes; ++i) {
    int const node_id = node_ids[i];
    load_pt += global->R_nodal(disp_idx, node_id, m_reaction_force_comp);
  }

  return load_pt;
}

template <typename T>
void Calibration<T>::preprocess_finalize(int step) {
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
void Calibration<T>::postprocess(double& J) {
  J = PCU_Add_Double(J);
  double const J_disp = J;
  double J_forc = 0.5 * m_balance_factor * m_dt / m_total_time
      * std::pow(m_load_mismatch, 2);
  J += J_forc;
  J /= PCU_Comm_Peers(); // to account for reduction in the objective

  if (m_write_obj_at_step && PCU_Comm_Self() == 0) {
    std::ofstream out_file;
    if (this->m_step == 1) {
      out_file.open(m_obj_out_file);
    } else {
      out_file.open(m_obj_out_file, std::ios::app | std::ios::out);
    }
    out_file << std::scientific << std::setprecision(17);
    out_file << J_disp << " " << J_forc << "\n";
    out_file.close();
  }
}

template <typename T>
void Calibration<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  // get the id of the node wrt element if this node is on the load surface
  // do not evaluate if the node is not on the load surface
  int const node_id_load = m_mapping_load[elem_set][elem][0];
  if (node_id_load < 0) return;

  T load = compute_load(elem_set, elem, global, local, iota, w, dv);
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

  this->initialize_value_pt();

  // see if the element (2D) / facet (3D) is included in the objective function
  int const facet_id_disp = m_mapping_disp[elem_set][elem];
  if (facet_id_disp < 0) return;

  int const ndims = this->m_num_dims;
  if (ndims == 2) {
    this->value_pt =
        compute_disp_mismatch(elem_set, elem, global, local, iota_input);
  } else if (ndims == 3) {
    this->value_pt =
        compute_surface_mismatch(elem_set, elem, global, local, iota_input);
  }
}

template <>
void Calibration<FADT>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<FADT>> global,
    RCP<LocalResidual<FADT>> local,
    apf::Vector3 const& iota_input,
    double w,
    double dv) {

  this->initialize_value_pt();

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  // similar for node ...
  int const facet_id_disp = m_mapping_disp[elem_set][elem];
  int const node_id_load = m_mapping_load[elem_set][elem][0];
  if (facet_id_disp + node_id_load == -2) return;

  int const ndims = this->m_num_dims;
  if (facet_id_disp > -1) {
    FADT mismatch;
    if (ndims == 2) {
      mismatch =
          compute_disp_mismatch(elem_set, elem, global, local, iota_input);
    } else if (ndims == 3) {
      mismatch =
          compute_surface_mismatch(elem_set, elem, global, local, iota_input);
    }
    this->value_pt += mismatch;
  }

  if (node_id_load > -1) {
    FADT load = compute_load(elem_set, elem, global, local, iota_input, w, dv);
    this->value_pt += m_balance_factor * m_dt / m_total_time
        * m_load_mismatch * load;
  }

}

template class Calibration<double>;
template class Calibration<FADT>;

}
