#include <apfGeometry.h>
#include <gmi_mesh.h>
#include <PCU.h>

#include "arrays.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "macros.hpp"

#include <cassert>
#include <iostream>

using namespace calibr8;

static void print_usage(int argc, char** argv) {
  if (argc == 5) {
    return;
  } else {
    std::cout << "usage: " << argv[0]
              << " <geom.dmg> <mesh.smb> <assoc.txt>"
              << " <outmesh>\n";
    abort();
  }
}

NodeSets compute_node_sets(apf::Mesh2* mesh, apf::StkModels* sets) {
  int const num_node_sets = sets->models[0].size();
  NodeSets node_sets;
  for (int i = 0; i < num_node_sets; ++i) {
    resize(node_sets[sets->models[0][i]->stkName], 0);
  }
  apf::Numbering* owned_nmbr = apf::numberOwnedNodes(mesh, "owned");
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(owned_nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    apf::MeshEntity* ent = node.entity;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(
        mesh, sets->invMaps[0], mesh->toModel(ent), mset);
    if (mset.empty()) continue;
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* stkm = *mit;
      std::string const name = stkm->stkName;
      node_sets[name].push_back(node);
    }
  }
  apf::destroyNumbering(owned_nmbr);
  owned_nmbr = nullptr;
  return node_sets;
}

void create_node_set_fields(apf::Mesh2* mesh, NodeSets const& node_sets) {
  for (auto const& pair : node_sets) {
    std::string const& node_set_name = pair.first;
    NodeSet const& node_set = pair.second;
    apf::Field* f = apf::createFieldOn(mesh, node_set_name.c_str(),
        apf::SCALAR);
    apf::MeshEntity* vert;
    apf::MeshIterator* nodes = mesh->begin(0);
    while ((vert = mesh->iterate(nodes))) {
      apf::setScalar(f, vert, 0, 0.);
    }
    mesh->end(nodes);
    for (auto const& node : node_set) {
      apf::setScalar(f, node.entity, node.node, 1.);
    }
    apf::synchronize(f);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  PCU_Comm_Init();
  print_usage(argc, argv);
  gmi_register_mesh();

  auto const geo_file = argv[1];
  auto const input_mesh_file = argv[2];
  auto const assoc_file = argv[3];
  auto const output_mesh_file = argv[4];

  auto mesh = apf::loadMdsMesh(geo_file, input_mesh_file);
  mesh->verify();

  auto sets = read_sets(mesh, assoc_file);
  auto node_sets = compute_node_sets(mesh, sets);
  create_node_set_fields(mesh, node_sets);

  mesh->writeNative(output_mesh_file);

  mesh->destroyNative();
  apf::destroyMesh(mesh);

  PCU_Comm_Free();
  MPI_Finalize();
}
