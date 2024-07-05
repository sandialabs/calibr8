#include <apf.h>
#include <apfCavityOp.h>
#include <apfMesh.h>
#include <PCU.h>

#include "control.hpp"
#include "macros.hpp"

namespace calibr8 {

struct Specification {
  apf::Mesh* mesh;
  apf::Field* error;
  int p_order;
  int target;
  double alpha;
  double beta;
  double size_factor;
  apf::Field* elem_size;
  apf::Field* vtx_size;
};

static void setup_specification(
    Specification* s,
    apf::Field* err,
    int target,
    int p_order) {
  s->mesh = apf::getMesh(err);
  s->error = err;
  s->p_order = p_order;
  s->target = target;
  s->alpha = 0.25;
  s->beta = 2.0;
  s->size_factor = 0.0;
  s->elem_size = 0;
  s->vtx_size = 0;
}

static double sum_contributions(Specification* s) {
  double r = 0.0;
  double d = s->mesh->getDimension();
  double p = s->p_order;
  apf::MeshEntity* elem;
  auto it = s->mesh->begin(d);
  while ((elem = s->mesh->iterate(it))) {
    double v = std::abs(apf::getScalar(s->error, elem, 0));
    r += std::pow(v, ((2.0 * d) / (2.0 * p + d)));
  }
  s->mesh->end(it);
  PCU_Add_Doubles(&r, 1);
  return r;
}

static void compute_size_factor(Specification* s) {
  double d = s->mesh->getDimension();
  double G = sum_contributions(s);
  double N = s->target;
  s->size_factor = std::pow((G/N), (1.0/d));
}

static double get_current_size(apf::Mesh* m, apf::MeshEntity* e) {
  double h = 0.0;
  apf::Downward edges;
  int ne = m->getDownward(e, 1, edges);
  for (int i = 0; i < ne; ++i)
    h += apf::measure(m, edges[i]) * apf::measure(m, edges[i]);
  return std::sqrt(h/ne);
}

static double get_new_size(Specification* s, apf::MeshEntity* e) {
  double p = s->p_order;
  double d = s->mesh->getDimension();
  double h = get_current_size(s->mesh, e);
  double theta_e = std::abs(apf::getScalar(s->error, e, 0));
  double r = std::pow(theta_e, ((-2.0) / (2.0*p + d)));
  double h_new = s->size_factor * r * h;
  if (h_new < s->alpha * h) h_new = s->alpha * h;
  if (h_new > s->beta * h) h_new = s->beta * h;
  return h_new;
}

static void get_elem_size(Specification* s) {
  auto e_size = apf::createStepField(s->mesh, "esize", apf::SCALAR);
  auto d = s->mesh->getDimension();
  apf::MeshEntity* elem;
  auto it = s->mesh->begin(d);
  while ((elem = s->mesh->iterate(it))) {
    double h = get_new_size(s, elem);
    apf::setScalar(e_size, elem, 0, h);
  }
  s->mesh->end(it);
  s->elem_size = e_size;
}

static void avg_to_vtx(
    apf::Field* ef,
    apf::Field* vf,
    apf::MeshEntity* ent) {
  auto m = apf::getMesh(ef);
  apf::Adjacent elems;
  m->getAdjacent(ent, m->getDimension(), elems);
  double s = 0.0;
  for (size_t i = 0; i < elems.getSize(); ++i)
    s += apf::getScalar(ef, elems[i], 0);
  s /= elems.getSize();
  apf::setScalar(vf, ent, 0, s);
}

class AverageOp : public apf::CavityOp {
  public:
    AverageOp(Specification* s) :
      apf::CavityOp(s->mesh), specs(s), entity(0) {}
    virtual Outcome setEntity(apf::MeshEntity* e) {
      entity = e;
      if (apf::hasEntity(specs->vtx_size, entity)) return SKIP;
      if (!requestLocality(&entity, 1)) return REQUEST;
      return OK;
    }
    virtual void apply() {
      avg_to_vtx(specs->elem_size, specs->vtx_size, entity);
    }
    Specification* specs;
    apf::MeshEntity* entity;
};

static void average_size_field(Specification* s) {
  s->vtx_size = apf::createLagrangeField(s->mesh, "size", apf::SCALAR, 1);
  AverageOp op(s);
  op.applyToDimension(0);
}

static void create_size_field(Specification* s) {
  compute_size_factor(s);
  get_elem_size(s);
  average_size_field(s);
  apf::destroyField(s->elem_size);
  apf::destroyField(s->error);
}

apf::Field* get_iso_target_size(
    apf::Field* e,
    int target) {
  DEBUG_ASSERT(target > 0);
  Specification s;
  setup_specification(&s, e, target, 1);
  create_size_field(&s);
  return s.vtx_size;
}

}
