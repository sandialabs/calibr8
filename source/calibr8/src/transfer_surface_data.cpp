#include <apf.h>
#include <apfGeometry.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <gmi_mesh.h>
#include <Kokkos_Core.hpp>
#include <PCU.h>

#include "arrays.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "macros.hpp"

#include <cassert>
#include <iostream>

using namespace calibr8;

static void print_usage(int argc, char** argv) {
  if (argc == 9) {
    return;
  } else {
    std::cout << "usage: " << argv[0]
              << " <geom_3d.dmg> <mesh_3d.smb> <assoc_3d.txt> <num steps>"
              << " <surface_set_name>"
              << " <geom_2d.dmg> <mesh_2d.smb>"
              << " <outmesh.smb>\n";
    abort();
  }
  if (!PCU_Comm_Self()) {
    return;
  } else {
    std::cout << "meshes must be serial --- use collapse\n";
    abort();
  }
}

static apf::StkModels* read_sets(apf::Mesh* m, std::string const& assoc_file) {
  apf::StkModels* sets = new apf::StkModels;
  char const* filename = assoc_file.c_str();
  static std::string const setNames[3] = {
    "node set", "side set", "elem set"};
  int const d = m->getDimension();
  int const dims[3] = {0, d - 1, d};
  std::ifstream f(filename);
  if (!f.good()) fail("cannot open file: %s", filename);
  std::string sline;
  int lc = 0;
  while (std::getline(f, sline)) {
    if (!sline.length()) break;
    ++lc;
    int sdi = -1;
    for (int di = 0; di < 3; ++di)
      if (sline.compare(0, setNames[di].length(), setNames[di]) == 0) sdi = di;
    if (sdi == -1)
      fail("invalid association line # %d:\n\t%s", lc, sline.c_str());
    int sd = dims[sdi];
    std::stringstream strs(sline.substr(setNames[sdi].length()));
    auto set = new apf::StkModel();
    strs >> set->stkName;
    int nents;
    strs >> nents;
    if (!strs) fail("invalid association line # %d:\n\t%s", lc, sline.c_str());
    for (int ei = 0; ei < nents; ++ei) {
      std::string eline;
      std::getline(f, eline);
      if (!f || !eline.length())
        fail("invalid association after line # %d", lc);
      ++lc;
      std::stringstream strs2(eline);
      int mdim, mtag;
      strs2 >> mdim >> mtag;
      if (!strs2) fail("bad associations line # %d:\n\t%s", lc, eline.c_str());
      set->ents.push_back(m->findModelEntity(mdim, mtag));
      if (!set->ents.back())
        fail("no model entity with dim: %d and tag: %d", mdim, mtag);
    }
    sets->models[sd].push_back(set);
  }
  sets->computeInverse();
  return sets;
}

apf::Field* get_measured_step_data(apf::Mesh2* m, int step) {
  auto name = "measured_" + std::to_string(step);
  auto measured_data = m->findField(name.c_str());
  assert(measured_data);
  return measured_data;
}

NodeSet get_surface_nodes(
    apf::Mesh2* m,
    apf::StkModels* sets,
    std::string const& surface_set_name) {
  apf::Numbering* owned_nmbr = apf::numberOwnedNodes(m, "owned");
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(owned_nmbr, owned);
  size_t const num_owned = owned.size();
  NodeSet surface_nodes;
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    apf::MeshEntity* ent = node.entity;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(
        m, sets->invMaps[0], m->toModel(ent), mset);
    if (mset.empty()) continue;
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* stkm = *mit;
      std::string const name = stkm->stkName;
      if (name == surface_set_name) {
        surface_nodes.push_back(node);
      }
    }
  }
  owned_nmbr = nullptr;
  return surface_nodes;
}

Array1D<apf::MeshEntity*> get_surface_mapping(
    apf::Mesh2* mesh_3d,
    apf::Mesh2* mesh_2d,
    NodeSet surface_nodes) {
  size_t const num_surface_nodes = surface_nodes.size();
  size_t const num_nodes_2d = mesh_2d->count(0);
  ALWAYS_ASSERT(num_surface_nodes == num_nodes_2d);
  Array1D<apf::MeshEntity*> mapping(num_surface_nodes);
  apf::Vector3 x_3d(0., 0., 0.);
  apf::Vector3 x_2d(0., 0., 0.);
  apf::MeshEntity* vert;
  double const tol = 1e-10;
  int m = 0;
  for (auto& surface_node : surface_nodes) {
    apf::MeshEntity* ent = surface_node.entity;
    mesh_3d->getPoint(ent, 0, x_3d);
    int n = 0;
    apf::MeshIterator* nodes_2d = mesh_2d->begin(0);
    while ((vert = mesh_2d->iterate(nodes_2d))) {
      mesh_2d->getPoint(vert, 0, x_2d);
      if (apf::areClose(x_3d, x_2d, tol)) {
        mapping[m] = vert;
        break;
      }
    ++n;
    }
    mesh_2d->end(nodes_2d);
    ++m;
  }
  return mapping;
}

void transfer_field(
    apf::Field* f_3d,
    apf::Field* f_2d,
    NodeSet surface_nodes,
    Array1D<apf::MeshEntity*> mapping) {
  int const num_comps = apf::countComponents(f_3d);
  std::vector<double> vals(num_comps, 0.);
  std::vector<double> vals_2d(num_comps, 0.);
  int n = 0;
  for (auto& surface_node : surface_nodes) {
    apf::MeshEntity* ent = surface_node.entity;
    apf::getComponents(f_3d, ent, 0, &(vals[0]));
    apf::setComponents(f_2d, mapping[n], 0, &(vals[0]));
    ++n;
  }
}

void transfer_measured_fields(
    apf::Mesh2* mesh_3d,
    apf::Mesh2* mesh_2d,
    int const num_steps,
    NodeSet surface_nodes,
    Array1D<apf::MeshEntity*> mapping) {
  for (int step = 0; step <= num_steps; ++step) {
    auto f_3d = get_measured_step_data(mesh_3d, step);
    auto name = "measured_" + std::to_string(step);
    apf::Field* f_2d = apf::createFieldOn(mesh_2d, name.c_str(),
        apf::getValueType(f_3d));
    transfer_field(f_3d, f_2d, surface_nodes, mapping);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize();
  PCU_Comm_Init();
  print_usage(argc, argv);
  gmi_register_mesh();

  auto const geo_file_3d = argv[1];
  auto const input_mesh_file_3d = argv[2];
  auto const assoc_file_3d = argv[3];
  int const num_steps = std::stoi(argv[4]);
  auto const surface_set_name = argv[5];

  auto const geo_file_2d = argv[6];
  auto const input_mesh_file_2d = argv[7];

  auto const output_mesh_file = argv[8];

  auto mesh_3d = apf::loadMdsMesh(geo_file_3d, input_mesh_file_3d);
  mesh_3d->verify();
  auto sets_3d = read_sets(mesh_3d, assoc_file_3d);

  auto mesh_2d = apf::loadMdsMesh(geo_file_2d, input_mesh_file_2d);
  mesh_2d->verify();

  auto surface_nodes = get_surface_nodes(mesh_3d, sets_3d, surface_set_name);
  auto mapping = get_surface_mapping(mesh_3d, mesh_2d, surface_nodes);
  transfer_measured_fields(mesh_3d, mesh_2d, num_steps,
      surface_nodes, mapping);

  mesh_2d->writeNative(output_mesh_file);

  mesh_2d->destroyNative();
  apf::destroyMesh(mesh_2d);

  mesh_3d->destroyNative();
  apf::destroyMesh(mesh_3d);

  PCU_Comm_Free();
  Kokkos::finalize();
  MPI_Finalize();
}
