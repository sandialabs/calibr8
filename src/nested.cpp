#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfShape.h>
#include <ma.h>
#include <PCU.h>
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "nested.hpp"
#include "macros.hpp"
#include "state.hpp"

namespace calibr8 {

NestedDisc::NestedDisc(RCP<Disc> disc, int type) {
  m_is_base = false;
  m_disc_type = type;
  m_base_mesh = disc->apf_mesh();
  m_sets = disc->sets();
  m_is_null_model = disc->is_null();
  number_elems();
  copy_mesh();
  tag_old_verts();
  refine();
  store_old_verts();
  initialize();
  create_primal(disc);
  create_adjoint_for_spr(disc);
}

void NestedDisc::number_elems() {
  int i = 0;
  int const vt = apf::SCALAR;
  int const dim = m_base_mesh->getDimension();
  m_base_elems.resize(m_base_mesh->count(dim));
  apf::FieldShape* s = apf::getIPFitShape(dim, 1);
  apf::Field* f = apf::createField(m_base_mesh, "elems", vt, s);
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m_base_mesh->begin(dim);
  while ((elem = m_base_mesh->iterate(elems))) {
    m_base_elems[i] = elem;
    apf::setScalar(f, elem, 0, i++);
  }
  m_base_mesh->end(elems);
}

NestedDisc::~NestedDisc() {
  apf::Field* e1 = m_base_mesh->findField("elems");
  apf::Field* e2 = m_mesh->findField("elems");
  apf::destroyField(e1);
  apf::destroyField(e2);
  apf::removeTagFromDimension(m_mesh, m_old_vtx_tag, 0);
  apf::removeTagFromDimension(m_mesh, m_new_vtx_tag, 0);
  m_mesh->destroyTag(m_old_vtx_tag);
  m_mesh->destroyTag(m_new_vtx_tag);
}

void NestedDisc::copy_mesh() {
  auto model = m_base_mesh->getModel();
  m_mesh = apf::createMdsMesh(model, m_base_mesh);
  apf::disownMdsModel(m_mesh);
}

void NestedDisc::tag_old_verts() {
  int i = -1;
  apf::MeshEntity* vtx;
  apf::MeshIterator* it = m_mesh->begin(0);
  m_old_vtx_tag = m_mesh->createIntTag("ovt", 1);
  m_new_vtx_tag = m_mesh->createIntTag("nvt", 2);
  while ((vtx = m_mesh->iterate(it))) {
    m_mesh->setIntTag(vtx, m_old_vtx_tag, &(++i));
  }
  m_mesh->end(it);
  m_old_vertices.resize(m_mesh->count(0));
}

class Transfer : public ma::SolutionTransfer {
  public:
    Transfer(apf::Mesh* m, apf::MeshTag* o, apf::MeshTag* n);
    bool hasNodesOn(int dim);
    void onVertex(apf::MeshElement* p, ma::Vector const&, ma::Entity* vtx);
  private:
    apf::Mesh* mesh;
    apf::MeshTag* old_vtx_tag;
    apf::MeshTag* new_vtx_tag;
    int num_dims;
};

Transfer::Transfer(apf::Mesh* m, apf::MeshTag* o, apf::MeshTag* n) {
  mesh = m;
  old_vtx_tag = o;
  new_vtx_tag = n;
  num_dims = mesh->getDimension();
}

bool Transfer::hasNodesOn(int dim) {
  if (dim == 0) return true;
  if (dim == num_dims) return true;
  return false;
}

void Transfer::onVertex(
    apf::MeshElement* p,
    ma::Vector const&,
    ma::Entity* vtx) {
  int tags[2];
  apf::MeshEntity* verts[2];
  auto edge = apf::getMeshEntity(p);
  mesh->getDownward(edge, 0, verts);
  mesh->getIntTag(verts[0], old_vtx_tag, &(tags[0]));
  mesh->getIntTag(verts[1], old_vtx_tag, &(tags[1]));
  mesh->setIntTag(vtx, new_vtx_tag, &(tags[0]));
}

void NestedDisc::refine() {
  ma::AutoSolutionTransfer transfers(m_mesh);
  auto mytransfer = new Transfer(m_mesh, m_old_vtx_tag, m_new_vtx_tag);
  transfers.add(mytransfer);
  auto in = ma::makeAdvanced(ma::configureUniformRefine(m_mesh, 1, &transfers));
  in->shouldFixShape = false;
  in->shouldSnap = false;
  ma::adapt(in);
}

void NestedDisc::store_old_verts() {
  apf::MeshEntity* vtx;
  apf::MeshIterator* it = m_mesh->begin(0);
  while ((vtx = m_mesh->iterate(it))) {
    if (! m_mesh->hasTag(vtx, m_old_vtx_tag)) continue;
    int tag;
    m_mesh->getIntTag(vtx, m_old_vtx_tag, &tag);
    m_old_vertices[tag] = vtx;
  }
  m_mesh->end(it);
}

void NestedDisc::create_primal(RCP<Disc> disc) {
  Array1D<Fields> const& base_primal = disc->primal();
  int const nsteps = base_primal.size();
  for (int step = 0; step < nsteps; ++step) {
    int const ngr = base_primal[0].global.size();
    int const model_form = BASE_MODEL;
    int const nlr = base_primal[0].local[model_form].size();
    Fields fields;
    for (int i = 0; i < ngr; ++i) {
      const char* name = apf::getName(base_primal[step].global[i]);
      apf::Field* f = m_mesh->findField(name);
      ALWAYS_ASSERT(f);
      fields.global.push_back(f);
    }
    for (int i = 0; i < nlr; ++i) {
      const char* name = apf::getName(base_primal[step].local[model_form][i]);
      apf::Field* f = m_mesh->findField(name);
      ALWAYS_ASSERT(f);
      fields.local[model_form].push_back(f);
    }
    m_primal.push_back(fields);
  }
}

void NestedDisc::create_adjoint_for_spr(RCP<Disc> disc) {
  Array1D<Fields> const& base_adjoint = disc->adjoint();
  if (base_adjoint.size() == 0) return;
  int const nsteps = base_adjoint.size();
  for (int step = 0; step < nsteps; ++step) {
    int const model_form = BASE_MODEL;
    int const ngr = base_adjoint[0].global.size();
    int const nlr = base_adjoint[0].local[model_form].size();
    Fields fields;
    for (int i = 0; i < ngr; ++i) {
      const char* name = apf::getName(base_adjoint[step].global[i]);
      apf::Field* f = m_mesh->findField(name);
      ALWAYS_ASSERT(f);
      fields.global.push_back(f);
    }
    for (int i = 0; i < nlr; ++i) {
      const char* name = apf::getName(base_adjoint[step].local[model_form][i]);
      apf::Field* f = m_mesh->findField(name);
      ALWAYS_ASSERT(f);
      fields.local[model_form].push_back(f);
    }
    m_adjoint.push_back(fields);
  }
}

apf::Field* NestedDisc::get_coarse(apf::Field* f) {
  int tags[2];
  double comps0[3];
  double comps1[3];
  double new_comps[3];
  std::string const name = std::string(apf::getName(f)) + "_coarse";
  int const vt = apf::getValueType(f);
  apf::Mesh* m = apf::getMesh(f);
  apf::FieldShape* shape = apf::getShape(f);
  apf::Field* coarse = apf::createField(m, name.c_str(), vt, shape);
  apf::MeshEntity* vtx;
  apf::MeshIterator* it = m_mesh->begin(0);
  while ((vtx = m->iterate(it))) {
    if (!m->hasTag(vtx, m_new_vtx_tag)) continue;
    m_mesh->getIntTag(vtx, m_new_vtx_tag, &(tags[0]));
    auto vtx0 = m_old_vertices[tags[0]];
    auto vtx1 = m_old_vertices[tags[1]];
    apf::getComponents(f, vtx0, 0, &(comps0[0]));
    apf::getComponents(f, vtx1, 0, &(comps1[0]));
    int ncomps = apf::countComponents(f);
    for (int comp = 0; comp < ncomps; ++comp) {
      new_comps[comp] = 0.5*(comps0[comp] + comps1[comp]);
    }
    apf::setComponents(coarse, vtx, 0, &(new_comps[0]));
  }
  return coarse;
}

void NestedDisc::set_error(apf::Field* nested_error) {
  apf::Field* elem_nmbr = m_mesh->findField("elems");
  apf::Field* base_err = apf::createStepField(m_base_mesh, "error", apf::SCALAR);
  apf::zeroField(base_err);
  apf::MeshEntity* nested_elem;
  apf::MeshIterator* it = m_mesh->begin(m_num_dims);
  while ((nested_elem = m_mesh->iterate(it))) {
    int const id = int(apf::getScalar(elem_nmbr, nested_elem, 0));
    apf::MeshEntity* base_elem = m_base_elems[id];
    double base_val = apf::getScalar(base_err, base_elem, 0);
    double const nested_contrib = apf::getScalar(nested_error, nested_elem, 0);
    base_val += nested_contrib;
    apf::setScalar(base_err, base_elem, 0, base_val);
  }
  m_mesh->end(it);
}

void NestedDisc::set_error(
    apf::Field* nested_global_error,
    apf::Field* nested_local_error) {
  apf::Field* elem_nmbr = m_mesh->findField("elems");
  apf::Field* base_err = apf::createStepField(m_base_mesh, "error", apf::SCALAR);
  apf::zeroField(base_err);
  apf::MeshEntity* nested_elem;
  apf::MeshIterator* it = m_mesh->begin(m_num_dims);
  while ((nested_elem = m_mesh->iterate(it))) {
    int const id = int(apf::getScalar(elem_nmbr, nested_elem, 0));
    apf::MeshEntity* base_elem = m_base_elems[id];
    double base_val = apf::getScalar(base_err, base_elem, 0);
    double const nested_global_contrib =
      apf::getScalar(nested_global_error, nested_elem, 0);
    double const nested_local_contrib =
      apf::getScalar(nested_local_error, nested_elem, 0);
    base_val += nested_global_contrib;
    base_val += nested_local_contrib;
    apf::setScalar(base_err, base_elem, 0, base_val);
  }
  m_mesh->end(it);
}

}
