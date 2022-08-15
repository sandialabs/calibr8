#include "adapt.hpp"
#include "control.hpp"
#include "physics.hpp"

namespace calibr8 {

static apf::Field* interp_error_to_cells(apf::Mesh* mesh) {
  apf::Field* eta = mesh->findField("eta");
  apf::Field* error = apf::createStepField(mesh, "error", apf::SCALAR);
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
  apf::writeVtkFiles("debug", mesh);
  apf::destroyField(eta);
  return error;
}

class Test : public Adapt {
  void adapt(ParameterList const& params, RCP<Physics> physics);
};

void Test::adapt(ParameterList const& params, RCP<Physics> physics) {
  print("im adapting");
  apf::Field* error  = interp_error_to_cells(physics->disc()->apf_mesh());
  (void)params;
  (void)physics;
}

RCP<Adapt> create_adapt(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "test") {
    return rcp(new Test);
  } else {
    throw std::runtime_error("invalid adapt");
  }
}

}
