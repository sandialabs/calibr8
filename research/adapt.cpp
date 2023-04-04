#include <PCU.h>
#include <ma.h>
#include <apfCavityOp.h>
#include "adapt.hpp"
#include "control.hpp"
#include "physics.hpp"

namespace calibr8 {

class Target : public Adapt {
  void adapt(
      ParameterList const& params,
      RCP<Physics> physics,
      apf::Field* error);
};

class Uniform : public Adapt {
  void adapt(
      ParameterList const& params,
      RCP<Physics> physics,
      apf::Field* error);
};

class Top : public Adapt {
  void adapt(
      ParameterList const& params,
      RCP<Physics> physics,
      apf::Field* error);
};

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
  ASSERT(target > 0);
  Specification s;
  setup_specification(&s, e, target, 1);
  create_size_field(&s);
  return s.vtx_size;
}

void Target::adapt(
    ParameterList const& params,
    RCP<Physics> physics,
    apf::Field* error) {
  print("adapting mesh with target size field");
  physics->disc()->change_shape(COARSE);
  apf::Mesh2* mesh = physics->disc()->apf_mesh();
  static int ctr = 1;
  int const base_target = params.get<int>("target");
  int const target = std::pow(2, ctr)*base_target;
  apf::Field* size_field = get_iso_target_size(error, target);
  ctr++;
  auto in = ma::makeAdvanced(ma::configure(mesh, size_field));
  in->maximumIterations = 1;
  in->shouldCoarsen = false;
  in->shouldFixShape = true;
  in->goodQuality = 0.4;
  in->shouldRunPreParma = true;
  in->shouldRunMidParma = true;
  in->shouldRunPostParma = true;
  ma::adapt(in);
  apf::destroyField(size_field);
}

void Uniform::adapt(
    ParameterList const& parmas,
    RCP<Physics> physics,
    apf::Field* error) {
  print("adapting mesh with uniform size field");
  apf::destroyField(error);
  physics->disc()->change_shape(COARSE);
  apf::Mesh2* mesh = physics->disc()->apf_mesh();
  auto in = ma::configureUniformRefine(mesh, 1);
  ma::adapt(in);
}

// this will only work in serial
static apf::Field* get_top_elems(
    apf::Mesh2* m,
    apf::Field* error,
    int percent) {
  std::vector<std::pair<double, apf::MeshEntity*>> errors;
  apf::MeshEntity* elem;
  auto elem_iterator = m->begin(m->getDimension());
  while ((elem = m->iterate(elem_iterator))) {
    double const value = apf::getScalar(error, elem, 0);
    errors.push_back(std::make_pair(value, elem));
  }
  m->end(elem_iterator);
  std::sort(errors.begin(), errors.end());
  double const nelems = m->count(m->getDimension());
  double const factor = percent/100.;
  int const nrefine_elems = int(factor*nelems);
  apf::Field* top = apf::createStepField(m, "top", apf::SCALAR);
  apf::zeroField(top);
  for (int i = 1; i <= nrefine_elems; ++i) {
    apf::setScalar(top, errors[nelems-i].second, 0, 1.);
  }
  return top;
}

static apf::Field* get_top_elem_size(apf::Field* top) {
  apf::Mesh* m = apf::getMesh(top);
  auto size = apf::createStepField(m, "esize", apf::SCALAR);
  apf::MeshEntity* elem;
  auto it = m->begin(m->getDimension());
  while ((elem = m->iterate(it))) {
    double const h = get_current_size(m, elem);
    double const should_refine = apf::getScalar(top, elem, 0);
    double h_new = h;
    if (std::abs(should_refine) > 1.e-8) h_new *= 0.5;
    apf::setScalar(size, elem, 0, h_new);
  }
  m->end(it);
  return size;
}

// this will only work in serial
static apf::Field* avg_top_to_vtx(apf::Field* elem_size) {
  apf::Mesh* m = apf::getMesh(elem_size);
  auto size = apf::createFieldOn(m, "size", apf::SCALAR);
  apf::MeshEntity* vtx;
  auto it = m->begin(0);
  apf::Adjacent elems;
  while ((vtx = m->iterate(it))) {
    m->getAdjacent(vtx, m->getDimension(), elems);
    double s = 0.;
    for (size_t i = 0; i < elems.getSize(); ++i) {
      s += apf::getScalar(elem_size, elems[i], 0);
    }
    s /= elems.getSize();
    apf::setScalar(size, vtx, 0, s);
  }
  m->end(it);
  return size;
}

void Top::adapt(
    ParameterList const& params,
    RCP<Physics> physics,
    apf::Field* error) {
  int const percent = params.get<int>("percent");
  print("adapting mesh with top %d%% size field", percent);
  physics->disc()->change_shape(COARSE);
  apf::Mesh2* mesh = physics->disc()->apf_mesh();
  auto top_elems = get_top_elems(mesh, error, percent);
  auto elem_size = get_top_elem_size(top_elems);
  auto vtx_size = avg_top_to_vtx(elem_size);
  apf::destroyField(top_elems);
  apf::destroyField(elem_size);
  apf::destroyField(error);
  auto in = ma::makeAdvanced(ma::configure(mesh, vtx_size));
  in->maximumIterations = 1;
  in->shouldCoarsen = false;
  in->shouldFixShape = false;
  physics->disc()->change_shape(COARSE);
  ma::adapt(in);
  apf::destroyField(vtx_size);
}

RCP<Adapt> create_adapt(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "target") {
    return rcp(new Target);
  } else if (type == "top") {
    return rcp(new Top);
  } else if (type == "uniform") {
    return rcp(new Uniform);
  } else {
    throw std::runtime_error("invalid adapt");
  }
}

}
