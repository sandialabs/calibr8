#include <gmi_mesh.h>
#include <gmi_null.h>
#include <apf.h>
#include <apfMesh2.h>
#include <apfMDS.h>
#include <PCU.h>
#include <lionPrint.h>
#include <parma.h>
#include <pcu_util.h>
#include <sstream>
#include <cstdlib>

const char* modelFile = 0;
const char* meshFile = 0;
const char* nodeSets = 0;
int numVectorFields = 0;
const char* outFile = 0;

void getConfig(int argc, char** argv) {
  if (argc != 6) {
    if (!PCU_Comm_Self()) {
      printf("Usage: %s <model> <mesh> <nodeset1,nodeset2,...,nodesetN> <outMesh>\n", argv[0]);
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  modelFile = argv[1];
  meshFile = argv[2];
  nodeSets = argv[3];
  numVectorFields = atoi(argv[4]);
  outFile = argv[5];
}

void vectorize_scalar_fields(apf::Mesh2* m) {
  for (int step = 0; step < numVectorFields; ++step) {
    std::string const ux_name = "ux_" + std::to_string(step);
    std::string const uy_name = "uy_" + std::to_string(step);
    std::string const uz_name = "uz_" + std::to_string(step);
    std::string const u_name = "measured_" + std::to_string(step+1);
    apf::Field* u = apf::createFieldOn(m, u_name.c_str(), apf::VECTOR);
    apf::Field* ux = m->findField(ux_name.c_str());
    apf::Field* uy = m->findField(uy_name.c_str());
    apf::Field* uz = m->findField(uz_name.c_str());
    apf::MeshEntity* vtx;
    apf::MeshIterator* vertices = m->begin(0);
    while ((vtx = m->iterate(vertices))) {
      double ux_val = apf::getScalar(ux, vtx, 0);
      double uy_val = apf::getScalar(uy, vtx, 0);
      double uz_val = apf::getScalar(uz, vtx, 0);
      apf::Vector3 u_val(ux_val, uy_val, uz_val);
      apf::setVector(u, vtx, 0, u_val);
    }
    m->end(vertices);
    apf::destroyField(ux);
    apf::destroyField(uy);
    apf::destroyField(uz);
  }
  { // create a zero initial condition
    std::string const u_name = "measured_0";
    apf::Field* u = apf::createFieldOn(m, u_name.c_str(), apf::VECTOR);
    apf::zeroField(u);
  }
  std::string const comma_delimited_node_sets(nodeSets);
  std::stringstream ss(comma_delimited_node_sets);
  std::vector<std::string> node_set_names;
  std::string node_set_name;
  while (std::getline(ss, node_set_name, ',')) {
    node_set_names.push_back(node_set_name);
  }
  printf("Removing extraneous node set fields:\n");
  for (auto node_set_name : node_set_names) {
    printf("  %s\n", node_set_name.c_str());
    for (int step = 1; step < numVectorFields; ++step) {
      std::string const fname = node_set_name + "_" + std::to_string(step);
      apf::Field* node_set_field = m->findField(fname.c_str());
      apf::destroyField(node_set_field);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc,&argv);
  PCU_Comm_Init();
  {
    getConfig(argc, argv);
    lion_set_verbosity(1);
    gmi_register_null();
    gmi_register_mesh();
    apf::Mesh2* m = apf::loadMdsMesh(modelFile, meshFile);
    vectorize_scalar_fields(m);
    m->writeNative(outFile);
  }
  PCU_Comm_Free();
  MPI_Finalize();
}
