#include <PCU.h>
#include <spr.h>
#include "control.hpp"
#include "error.hpp"
#include "error_adjoint.hpp"
#include "physics.hpp"
#include "cspr.hpp"

namespace calibr8 {

int get_ndofs(int space, RCP<Physics> physics) {
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  int const neqs = disc->num_eqs();
  int dofs = neqs * disc->owned_nodes(space).getSize();
  dofs = PCU_Add_Double(dofs);
  return dofs;
}

int get_nelems(RCP<Physics> physics) {
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  int nelems = mesh->count(disc->num_dims());
  nelems = PCU_Add_Double(nelems);
  return nelems;
}

void write_stream(
    std::string const& path,
    std::stringstream const& stream) {
  std::ofstream file_stream(path.c_str());
  if (!file_stream.is_open()) {
    throw std::runtime_error("write_stream - could not open: " + path);
  }
  file_stream << stream.rdbuf();
  file_stream.close();
}

void Error::write_mesh(RCP<Physics> physics, std::string const& file, int ctr) {
  std::string names[NUM_SPACE];
  names[COARSE] = file + "/" + "coarse_" + std::to_string(ctr);
  names[FINE] = file + "/" + "fine_" + std::to_string(ctr);
  RCP<Disc> disc = physics->disc();
  apf::Mesh* mesh = disc->apf_mesh();
  for (int space = 0; space < NUM_SPACE; ++space) {
    disc->change_shape(space);
    apf::writeVtkFiles(names[space].c_str(), mesh);
  }
}

static void write_pvd(
    std::stringstream& stream,
    std::string const& name,
    int nctr) {
  stream << "<VTKFile type=\"Collection\" version=\"0.1\">" << std::endl;
  stream << "  <Collection>" << std::endl;
  for (int ctr = 1; ctr <= nctr; ++ctr) {
    std::string const out = name + "_" + std::to_string(ctr);
    std::string const pvtu = out + "/" + out;
    stream << "    <DataSet timestep=\"" << ctr << "\" group=\"\" ";
    stream << "part=\"0\" file=\"" << pvtu;
    stream << ".pvtu\"/>" << std::endl;
  }
  stream << "  </Collection>" << std::endl;
  stream << "</VTKFile>" << std::endl;
}

void Error::write_pvd(std::string const& file, int nctr) {
  if (PCU_Comm_Self()) return;
  std::stringstream coarse_stream;
  std::stringstream fine_stream;
  calibr8::write_pvd(coarse_stream, "coarse", nctr);
  calibr8::write_pvd(fine_stream, "fine", nctr);
  write_stream(file + "/" + "coarse.pvd", coarse_stream);
  write_stream(file + "/" + "fine.pvd", fine_stream);
}

apf::Field* interp_error_to_cells(apf::Field* eta, std::string const& n) {
  print("interpolating error field to cell centers");
  apf::Mesh* mesh = apf::getMesh(eta);
  apf::Field* error = apf::createStepField(mesh, n.c_str(), apf::SCALAR);
  int const neqs = apf::countComponents(eta);
  Array1D<double> values(neqs, 0.);
  apf::MeshEntity* ent;
  apf::MeshIterator* elems = mesh->begin(mesh->getDimension());
  while ((ent = mesh->iterate(elems))) {
    apf::Vector3 xi;
    apf::MeshElement* me = apf::createMeshElement(mesh, ent);
    apf::Element* e = apf::createElement(eta, me);
    apf::getIntPoint(me, 1, 0, xi);
    apf::getComponents(e, xi, &(values[0]));
    double error_val = 0.;
    for (int eq = 0; eq < neqs; ++eq) {
      error_val += std::abs(values[eq]);
    }
    apf::setScalar(error, ent, 0, error_val);
    apf::destroyElement(e);
    apf::destroyMeshElement(me);
  }
  mesh->end(elems);
  return error;
}

RCP<Error> create_error(ParameterList const& params) {
  return rcp(new Adjoint(params));
}

}
