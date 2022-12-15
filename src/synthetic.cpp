#include "synthetic.hpp"

namespace calibr8 {

// TODO: this copies every field from the base mesh to the copied
// mesh. a text file format might make the memory usage more
// palatable.
void write_synthetic(
    std::string const& problem_name,
    RCP<Disc> disc,
    int num_steps) {

  // copy the mesh to a whole other mesh
  apf::Mesh* base_mesh = disc->apf_mesh();
  auto model = base_mesh->getModel();
  apf::Mesh2* synth_mesh = apf::createMdsMesh(model, base_mesh);
  apf::disownMdsModel(synth_mesh);

  for (int step = 0; step <= num_steps; ++step) {

    Fields& fields = disc->primal(step);
    // BASE MODEL
    int const model_form = 0;
    int const nlocal_fields = fields.local[model_form].size();
    int const nglobal_fields = fields.global.size();

    // delete non-measured local fields
    for (int i = 0; i < nlocal_fields; ++i) {
      const char* name = apf::getName(fields.local[model_form][i]);
      apf::Field* synth_local = synth_mesh->findField(name);
      apf::destroyField(synth_local);
    }

    // delete non-measured global fields
    // here is the assumption that the DIC data is index 0
    for (int i = 1; i < nglobal_fields; ++i) {
      const char* name = apf::getName(fields.global[i]);
      apf::Field* synth_local = synth_mesh->findField(name);
      apf::destroyField(synth_local);
    }

    // rename the measured global field
    const char* meas_old_name = apf::getName(fields.global[0]);
    apf::Field* meas_local = synth_mesh->findField(meas_old_name);
    std::string const meas_name = "measured_" + std::to_string(step);
    apf::renameField(meas_local, meas_name.c_str());

  }

  // write the synthetic data on a mesh that can be used for
  // inversion
  std::string const native_name = problem_name + "_synthetic/";
  synth_mesh->writeNative(native_name.c_str());

  // clean up the copied mesh
  synth_mesh->destroyNative();
  apf::destroyMesh(synth_mesh);

}

}
