#include <PCU.h>
#include <fstream>
#include <iomanip>
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "reaction_mismatch.hpp"

namespace calibr8 {

template <typename T>
ReactionMismatch<T>::ReactionMismatch(ParameterList const& params) {
  m_node_set = params.get<std::string>("node set");
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
  m_reaction_force_comp = params.get<int>("reaction force component");
}

template <typename T>
ReactionMismatch<T>::~ReactionMismatch() {
}

template <typename T>
void ReactionMismatch<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;

  Array1D<int> node_ids;

  // initialize the list of elements that touch the node set
  if (!is_initd) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_nodes;
    m_mapping.resize(disc->num_elem_sets());
    NodeSet const& nodes = disc->nodes(m_node_set);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, 0, downward_nodes);
        node_ids.resize(0);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_nodes[down];
          for (apf::Node node : nodes) {
            if (node.entity == downward_entity) {
              node_ids.push_back(down);
            }
          }
        }
        int const num_node_ids = node_ids.size();
        if (num_node_ids > 0) {
          m_mapping[es][elem].resize(num_node_ids);
          m_mapping[es][elem] = node_ids;
        } else {
          m_mapping[es][elem].resize(1);
          m_mapping[es][elem][0] = -1;
        }
      }
    }
    is_initd = true;
  }

}

template <typename T>
T ReactionMismatch<T>::compute_load(
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
  Array1D<int> const& node_ids = m_mapping[elem_set][elem];
  int const num_nodes = node_ids.size();

  // compute the internal force vector
  global->zero_residual();
  global->evaluate(local, iota, w, dv, 0);

  // sum relevant entries
  int const disp_idx = 0;
  for (size_t i = 0; i < num_nodes; ++i) {
    int const node_id = node_ids[i];
    T const load_vec_at_node = global->R_nodal(disp_idx, node_id,
        m_reaction_force_comp);
    load_pt += load_vec_at_node;
  }

  return load_pt;
}

template <typename T>
void ReactionMismatch<T>::preprocess_finalize(int step) {
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
void ReactionMismatch<T>::postprocess(double& J) {
  J += 0.5 * std::pow(m_load_mismatch, 2) / PCU_Comm_Peers();
}

template <typename T>
void ReactionMismatch<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  // get the id of the node wrt element if this node is on the QoI node set
  // do not evaluate if the element contains no nodes in the QoI node set
  int const node_id = m_mapping[elem_set][elem][0];
  if (node_id < 0) return;

  T load = compute_load(elem_set, elem, global, local, iota, w, dv);
  m_total_load += val(load);
}

template <>
void ReactionMismatch<double>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<double>> global,
    RCP<LocalResidual<double>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  this->initialize_value_pt();

}

template <>
void ReactionMismatch<FADT>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<FADT>> global,
    RCP<LocalResidual<FADT>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  this->initialize_value_pt();

  // get the id of the node wrt element if this node is on the QoI node set
  // do not evaluate if the element contains no nodes in the QoI node set
  int const node_id = m_mapping[elem_set][elem][0];
  if (node_id < 0) return;

  // weight the load by the mismatch
  FADT load = compute_load(elem_set, elem, global, local, iota, w , dv);
  this->value_pt = m_load_mismatch * load;
}

template class ReactionMismatch<double>;
template class ReactionMismatch<FADT>;

}
