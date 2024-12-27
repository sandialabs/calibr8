#include <apf.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <gmi_mesh.h>
#include <PCU.h>

#include <cassert>
#include <iostream>

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>

using Compadre::CreatePointCloudSearch;
using Compadre::Evaluator;
using Compadre::GMLS;
using Compadre::VectorPointEvaluation;
using Compadre::TargetOperation;
using Compadre::WeightingFunctionType;
using DoubleView = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>;
using DoubleDoubleView = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>;
using IntView = Kokkos::View<int*, Kokkos::DefaultExecutionSpace>;

static void print_usage(int argc, char** argv) {
  if (argc == 8) {
    return;
  } else {
    std::cout << "usage: " << argv[0]
              << " <geom.dmg> <mesh.smb> <num steps>"
              << " <poly_order> <power_kernel_exponent> <epsilon_multiplier>"
              << " <outmesh.smb>\n";
    abort();
  }
  if (!PCU_Comm_Self()) {
    return;
  } else {
    std::cout << "mesh must be serial --- use collapse\n";
    abort();
  }
}

apf::Field* get_measured_step_data(apf::Mesh2* m, int step) {
  auto name = "measured_" + std::to_string(step);
  auto measured_data = m->findField(name.c_str());
  assert(measured_data);
  return measured_data;
}

DoubleDoubleView get_coords(apf::Mesh2* m) {
  int const ndim = m->getDimension();
  apf::Vector3 x(0., 0., 0.);
  size_t const num_nodes = m->count(0);
  DoubleDoubleView coords("coordinates", num_nodes, 3);
  apf::MeshEntity* vert;
  apf::MeshIterator* nodes = m->begin(0);
  int n = 0;
  while ((vert = m->iterate(nodes))) {
    m->getPoint(vert, 0, x);
    for (int dim = 0; dim < ndim; ++dim) {
      coords(n, dim) = x[dim];
    }
    ++n;
  }
  m->end(nodes);
  return coords;
}

DoubleDoubleView get_field_values(apf::Mesh2* m, apf::Field* f) {
  int const ndim = m->getDimension();
  size_t const num_nodes = m->count(0);
  int const num_comps = apf::countComponents(f);
  DoubleDoubleView field_values("values", num_nodes, ndim);
  std::vector<double> vals(num_comps, 0.);
  apf::MeshEntity* vert;
  apf::MeshIterator* nodes = m->begin(0);
  int n = 0;
  while ((vert = m->iterate(nodes))) {
    apf::getComponents(f, vert, 0, &(vals[0]));
    for (int dim = 0; dim < ndim; ++dim) {
      field_values(n, dim) = vals[dim];
    }
    ++n;
  }
  m->end(nodes);
  return field_values;
}

void replace_measured_field(
    apf::Mesh2* m,
    apf::Field* measured_field,
    DoubleDoubleView filtered_field) {
  int const ndim = m->getDimension();
  size_t const num_nodes = m->count(0);
  int const num_comps = apf::countComponents(measured_field);
  std::vector<double> vals(num_comps, 0.);
  apf::MeshEntity* vert;
  apf::MeshIterator* nodes = m->begin(0);
  int n = 0;
  while ((vert = m->iterate(nodes))) {
    for (int dim = 0; dim < ndim; ++dim) {
      vals[dim] = filtered_field(n, dim);
    }
    apf::setComponents(measured_field, vert, 0, &(vals[0]));
    ++n;
  }
  m->end(nodes);
}

void filter_measured_fields(
    apf::Mesh2* m,
    int num_steps,
    int poly_order,
    int power_kernel_exponent,
    double epsilon_multiplier) {

  int const number_of_batches = 1;
  bool const keep_coefficients = true;

  int const ndim = m->getDimension();
  DoubleDoubleView coords = get_coords(m);
  size_t const num_nodes = coords.extent(0);

  GMLS gmls(poly_order, ndim);
  auto point_cloud_search(CreatePointCloudSearch(coords, ndim));

  int const min_neighbors = GMLS::getNP(poly_order, ndim);
  IntView neighbor_lists("neighbor lists", 0);
  IntView number_of_neighbors_list("number of neighbors list", num_nodes);
  DoubleView epsilon("h supports", num_nodes);
  size_t storage_size = point_cloud_search.generateCRNeighborListsFromKNNSearch(
      true, coords, neighbor_lists, number_of_neighbors_list,
      epsilon, min_neighbors, epsilon_multiplier
  );
  Kokkos::resize(neighbor_lists, storage_size);
  point_cloud_search.generateCRNeighborListsFromKNNSearch(
      false, coords, neighbor_lists, number_of_neighbors_list,
      epsilon, min_neighbors, epsilon_multiplier
  );

  gmls.setProblemData(neighbor_lists, number_of_neighbors_list,
      coords, coords, epsilon);

  std::vector<TargetOperation> target_operations = {VectorPointEvaluation};
  gmls.addTargets(target_operations);

  gmls.setWeightingType(WeightingFunctionType::Power);
  gmls.setWeightingParameter(power_kernel_exponent);

  gmls.generateAlphas(number_of_batches, keep_coefficients);

  Evaluator gmls_evaluator(&gmls);

  for (int step = 0; step <= num_steps; ++step) {
    auto f = get_measured_step_data(m, step);
    auto target_values = get_field_values(m, f);
    auto output_values =
        gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::DefaultExecutionSpace>
        (target_values, VectorPointEvaluation);
    replace_measured_field(m, f, output_values);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize();
  PCU_Comm_Init();
  print_usage(argc, argv);
  gmi_register_mesh();

  auto const geo_file = argv[1];
  auto const input_mesh_file = argv[2];
  int const num_steps = std::stoi(argv[3]);
  int const poly_order = std::stoi(argv[4]);
  int const power_kernel_exponent = std::stoi(argv[5]);
  double const epsilon_multiplier = std::stod(argv[6]);
  auto const output_mesh_file = argv[7];

  auto mesh = apf::loadMdsMesh(geo_file, input_mesh_file);
  mesh->verify();

  filter_measured_fields(mesh, num_steps,
      poly_order, power_kernel_exponent, epsilon_multiplier
  );

  mesh->writeNative(output_mesh_file);

  mesh->destroyNative();
  apf::destroyMesh(mesh);

  PCU_Comm_Free();
  Kokkos::finalize();
  MPI_Finalize();
}
