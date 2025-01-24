#include "mappings.hpp"

namespace calibr8 {

bool setup_side_set_mapping(std::string const& side_set,
    RCP<Disc> disc,
    Array2D<int>& mapping) {

  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  apf::Downward downward_entities;
  mapping.resize(disc->num_elem_sets());
  SideSet const& sides = disc->sides(side_set);
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      mapping[es][elem] = -1;
      apf::MeshEntity* elem_entity = elems[elem];
      int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_entities);
      for (int down = 0; down < ndown; ++down) {
        apf::MeshEntity* downward_entity = downward_entities[down];
        for (apf::MeshEntity* side : sides) {
          if (side == downward_entity) {
            mapping[es][elem] = down;
          }
        }
      }
    }
  }
  return true;
}

bool setup_side_sets_mapping(
    Array1D<std::string> const& side_sets,
    RCP<Disc> disc,
    Array2D<int>& mapping) {

  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  apf::Downward downward_entities;
  mapping.resize(disc->num_elem_sets());
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      mapping[es][elem] = -1;
      apf::MeshEntity* elem_entity = elems[elem];
      int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_entities);
      for (int down = 0; down < ndown; ++down) {
        apf::MeshEntity* downward_entity = downward_entities[down];
        for (std::string const& side_set : side_sets) {
          SideSet const& sides = disc->sides(side_set);
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              mapping[es][elem] = down;
            }
          }
        }
      }
    }
  }
  return true;
}

bool setup_side_set_to_node_mapping(
    std::string const& side_set,
    RCP<Disc> disc,
    Array3D<int>& mapping) {

  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  Array1D<int> node_ids;
  apf::Downward downward_entities;
  apf::Downward downward_nodes;
  mapping.resize(disc->num_elem_sets());
  SideSet const& sides = disc->sides(side_set);
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      mapping[es][elem].resize(1);
      mapping[es][elem][0] = -1;
      apf::MeshEntity* elem_entity = elems[elem];
      int ndown_entities = mesh->getDownward(elem_entity, ndims - 1, downward_entities);
      for (int down_entity = 0; down_entity < ndown_entities; ++down_entity) {
        apf::MeshEntity* downward_entity = downward_entities[down_entity];
        for (apf::MeshEntity* side : sides) {
          if (side == downward_entity) {
            int ndown_nodes = mesh->getDownward(downward_entity, 0, downward_nodes);
            node_ids.resize(0);
            for (int down_node = 0; down_node < ndown_nodes; ++down_node) {
              node_ids.push_back(down_node);
            }
            int const num_node_ids = node_ids.size();
            if (num_node_ids > 0) {
              mapping[es][elem].resize(num_node_ids);
              mapping[es][elem] = node_ids;
            }
          }
        }
      }
    }
  }
  return true;
}

bool setup_node_set_mapping(std::string const& node_set,
    RCP<Disc> disc,
    Array3D<int>& mapping) {

  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  Array1D<int> node_ids;
  apf::Downward downward_nodes;
  mapping.resize(disc->num_elem_sets());
  NodeSet const& nodes = disc->nodes(node_set);
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
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
        mapping[es][elem].resize(num_node_ids);
        mapping[es][elem] = node_ids;
      } else {
        mapping[es][elem].resize(1);
        mapping[es][elem][0] = -1;
      }
    }
  }
  return true;
}

bool setup_coord_based_node_mapping(
    int coord_idx,
    double coord_value,
    RCP<Disc> disc,
    Array3D<int>& mapping) {

  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  double const tol = 1e-12;
  Array1D<int> node_ids;
  apf::Vector3 x;
  apf::Downward downward_nodes;
  mapping.resize(disc->num_elem_sets());
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    mapping[es].resize(elems.size());
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshEntity* elem_entity = elems[elem];
      int ndown = mesh->getDownward(elem_entity, 0, downward_nodes);
      node_ids.resize(0);
      for (int down = 0; down < ndown; ++down) {
        apf::MeshEntity* downward_entity = downward_nodes[down];
        mesh->getPoint(downward_entity, 0, x);
        if (std::abs(x[coord_idx] - coord_value) < tol) {
          node_ids.push_back(down);
        }
      }
      int const num_node_ids = node_ids.size();
      if (num_node_ids > 0) {
        mapping[es][elem].resize(num_node_ids);
        mapping[es][elem] = node_ids;
      } else {
        mapping[es][elem].resize(1);
        mapping[es][elem][0] = -1;
      }
    }
  }
  return true;
}

}
