#include <apf.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <gmi_mesh.h>
#include <cassert>
#include <PCU.h>

#include <iostream>
#include <memory>
#include <random>

static void print_usage(int argc, char** argv) {
  if (PCU_Comm_Self()) return;
  if (argc == 7) return;
  std::cout << "usage: " << argv[0]
    << " <geom.dmg> <mesh.smb> <num steps> <seed> <random factor> <outmesh.smb>" << std::endl;
  abort();
}

class NormalNoise {
  public:
    NormalNoise(int seed, double factor);
    double get_random();
  private:
    double m_factor;
    std::unique_ptr<std::mt19937> m_mt;
    std::normal_distribution<double> m_dist {0.0, 1.0};
};

NormalNoise::NormalNoise(int seed, double factor) {
  m_mt = std::make_unique<std::mt19937>(seed + PCU_Comm_Self());
  m_factor = factor;
}

double NormalNoise::get_random() {
  return m_dist(*m_mt) * m_factor;
}

static apf::Field* get_step_data(apf::Mesh2* m, int step) {
  auto name = "measured_" + std::to_string(step);
  auto measured_data = m->findField(name.c_str());
  assert(measured_data);
  return measured_data;
}

static void perturb_measured_data(
    apf::Mesh2* m,
    int step,
    NormalNoise& noise) {
  int const ndim = m->getDimension();
  auto f = get_step_data(m, step);
  apf::Vector3 measured_vert;
  apf::MeshEntity* vert;
  apf::MeshIterator* verts = m->begin(0);
  while ((vert = m->iterate(verts))) {
    if (!m->isOwned(vert)) continue;
    apf::getVector(f, vert, 0, measured_vert);
    for (int d = 0; d < ndim; ++d) {
      measured_vert[d] += noise.get_random();
    }
    apf::setVector(f, vert, 0, measured_vert);
  }
  m->end(verts);
  apf::synchronize(f);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  PCU_Comm_Init();
  print_usage(argc, argv);
  gmi_register_mesh();
  auto gfile = argv[1];
  auto mfile = argv[2];
  int nsteps = std::stoi(argv[3]);
  int seed = std::stoi(argv[4]);
  double factor = std::stod(argv[5]);
  auto ofile = argv[6];
  auto mesh = apf::loadMdsMesh(gfile, mfile);
  mesh->verify();
  NormalNoise noise(seed, factor);
  for (int step = 0; step <= nsteps; ++step) {
    perturb_measured_data(mesh, step, noise);
  }
  mesh->writeNative(ofile);
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  PCU_Comm_Free();
  MPI_Finalize();
}
