#include <list>
#include <fstream>
#include <gmi_mesh.h>
#include <gmi_null.h>
#include <iomanip>
#include <PCU.h>
#include <Tpetra_Core.hpp>
#include "arrays.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "state.hpp"

namespace calibr8 {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("geom file", "");
  p.set<std::string>("mesh file", "");
  p.set<std::string>("assoc file", "");
  p.set<int>("num steps", 1);
  p.set<double>("step size", 1.);
  p.set<std::string>("time file", "");
  p.set<double>("initial time", 0.);
  return p;
}

static void load_mesh(apf::Mesh2** mesh, ParameterList const& p) {
  gmi_register_mesh();
  gmi_register_null();
  std::string const geom_file = p.get<std::string>("geom file");
  std::string const mesh_file = p.get<std::string>("mesh file");
  char const* g = geom_file.c_str();
  char const* m = mesh_file.c_str();
  *mesh = apf::loadMdsMesh(g, m);
}

static bool is_null_model(ParameterList const& p) {
  std::string const geom_file = p.get<std::string>("geom file");
  if (geom_file == ".null") {
    return true;
  } else {
    return false;
  }
}

static void destroy_existing_numberings(apf::Mesh2* m) {
  while (m->countNumberings()) {
    apf::destroyNumbering(m->getNumbering(0));
  }
}

apf::StkModels* read_sets(apf::Mesh* m, std::string const& assoc_file) {
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

Disc::Disc(ParameterList const& params) {
  ParameterList disc_params(params);
  disc_params.validateParametersAndSetDefaults(get_valid_params(), 0);
  load_mesh(&m_mesh, disc_params);
  create_time(disc_params);
  m_is_null_model = is_null_model(disc_params);
  destroy_existing_numberings(m_mesh);
  std::string const assoc_file = disc_params.get<std::string>("assoc file");
  m_sets = read_sets(m_mesh, assoc_file);
  // lol - be aware that this is called
  apf::reorderMdsMesh(m_mesh);
  // skip because of problems with non-manifold geometries
  if (!m_is_null_model) {
    m_mesh->verify();
  }
  initialize();
}

Disc::~Disc() {
  destroy_data();
  destroy_primal(false);
  destroy_adjoint();
  destroy_virtual();
  m_mesh->destroyNative();
  apf::destroyMesh(m_mesh);
  if (m_is_base) delete m_sets;
  m_time = {};
}

static int count_nodes(apf::FieldShape* s, int type) {
  apf::EntityShape* ent_shape = s->getEntityShape(type);
  return ent_shape->countNodes();
}

void Disc::create_time(ParameterList const& params) {
  std::string const time_file = params.get<std::string>("time file");
  if (time_file == "") {
    int const nsteps = params.get<int>("num steps");
    double const dt = params.get<double>("step size");
    m_time.resize(nsteps + 1);
    for (int step = 0; step <=nsteps; ++step) {
      m_time[step] = step * dt;
    }
  } else {
    double const initial_time = params.get<double>("initial time");
    m_time.resize(1);
    m_time[0] = initial_time;
    std::ifstream in_file(time_file);
    std::string line;
    while (getline(in_file, line)) {
      m_time.push_back(std::stod(line));
    }
  }
}

void Disc::initialize() {
  m_num_dims = m_mesh->getDimension();
  m_num_elems = m_mesh->count(m_num_dims);
  m_num_elem_sets = m_sets->models[m_num_dims].size();
  m_num_side_sets = m_sets->models[m_num_dims-1].size();
  m_num_node_sets = m_sets->models[0].size();
  m_gv_shape = apf::getLagrange(1);
  m_lv_shape = apf::getIPFitShape(m_num_dims, 1);
  m_elem_type = (m_num_dims == 3) ? apf::Mesh::TET : apf::Mesh::TRIANGLE;
  m_num_gv_nodes = count_nodes(m_gv_shape, m_elem_type);
  m_num_lv_nodes = count_nodes(m_lv_shape, m_elem_type);
  m_comm = Tpetra::getDefaultComm();
}

std::string Disc::elem_set_name(int es_idx) const {
  DEBUG_ASSERT(es_idx < m_num_elem_sets);
  return m_sets->models[m_num_dims][es_idx]->stkName;
}

std::string Disc::side_set_name(int ss_idx) const {
  DEBUG_ASSERT(ss_idx < m_num_side_sets);
  return m_sets->models[m_num_dims-1][ss_idx]->stkName;
}

std::string Disc::node_set_name(int ns_idx) const {
  DEBUG_ASSERT(ns_idx < m_num_node_sets);
  return m_sets->models[0][ns_idx]->stkName;
}

int Disc::elem_set_idx(std::string const& esn) const {
  int idx = -1;
  for (int i =0 ; i < m_num_elem_sets; ++i) {
    if (esn == m_sets->models[m_num_dims][i]->stkName) {
      idx = i;
    }
  }
  DEBUG_ASSERT(idx > -1);
  return idx;
}

int Disc::side_set_idx(std::string const& ssn) const {
  int idx = -1;
  for (int i = 0; i < m_num_side_sets; ++i) {
    if (ssn == m_sets->models[m_num_dims-1][i]->stkName) {
      idx = i;
    }
  }
  DEBUG_ASSERT(idx > -1);
  return idx;
}

int Disc::node_set_idx(std::string const& nsn) const {
  int idx = -1;
  for (int i = 0; i < m_num_node_sets; ++i) {
    if (nsn == m_sets->models[0][i]->stkName) {
      idx = i;
    }
  }
  DEBUG_ASSERT(idx > -1);
  return idx;
}

ElemSet const& Disc::elems(std::string const& name) {
  ALWAYS_ASSERT(m_elem_sets.count(name));
  return m_elem_sets[name];
}

SideSet const& Disc::sides(std::string const& name) {
  ALWAYS_ASSERT(m_side_sets.count(name));
  return m_side_sets[name];
}

NodeSet const& Disc::nodes(std::string const& name) {
  ALWAYS_ASSERT(m_node_sets.count(name));
  return m_node_sets[name];
}

void Disc::compute_node_map() {
  ALWAYS_ASSERT(! m_owned_nmbr);
  ALWAYS_ASSERT(! m_global_nmbr);
  m_owned_nmbr = apf::numberOwnedNodes(m_mesh, "owned");
  m_global_nmbr = apf::makeGlobal(m_owned_nmbr, false);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_global_nmbr, owned);
  size_t const num_owned = owned.size();
  Teuchos::Array<GO> indices(num_owned);
  for (size_t n = 0; n < num_owned; ++n) {
    indices[n] = apf::getNumber(m_global_nmbr, owned[n]);
  }
  m_node_map = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
}

void Disc::compute_coords() {
  m_coords = rcp(new MultiVectorT(m_node_map, m_num_dims, false));
  apf::Vector3 x(0., 0., 0.);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    m_mesh->getPoint(node.entity, node.node, x);
    for (int dim = 0; dim < m_num_dims; ++dim) {
      m_coords->replaceLocalValue(n, dim, x[dim]);
    }
  }
}

static LO get_dof(LO nid, int eq, int neq) {
  return nid * neq + eq;
}

static GO get_gdof(GO nid, int eq, int neq) {
  return nid * neq + eq;
}

void Disc::compute_owned_maps() {
  ALWAYS_ASSERT(m_num_residuals > 0);
  resize(m_maps[OWNED], m_num_residuals);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_global_nmbr, owned);
  size_t const num_owned = owned.size();
  Teuchos::Array<GO> indices;
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = num_eqs(i);
    ALWAYS_ASSERT(neqs > 0);
    indices.resize(neqs * num_owned);
    for (size_t node = 0; node < num_owned; ++node) {
      GO gid = apf::getNumber(m_global_nmbr, owned[node]);
      for (int eq = 0; eq < neqs; ++eq) {
        indices[get_dof(node, eq, neqs)] = get_gdof(gid, eq, neqs);
      }
    }
    m_maps[OWNED][i] = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
  }
  apf::synchronize(m_global_nmbr);
}

void Disc::compute_ghost_maps() {
  ALWAYS_ASSERT(m_num_residuals > 0);
  ALWAYS_ASSERT(! m_ghost_nmbr);
  resize(m_maps[GHOST], m_num_residuals);
  m_ghost_nmbr = apf::numberOverlapNodes(m_mesh, "ghost");
  apf::DynamicArray<apf::Node> ghost;
  apf::getNodes(m_ghost_nmbr, ghost);
  size_t const num_ghost = ghost.size();
  Teuchos::Array<GO> indices;
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = num_eqs(i);
    ALWAYS_ASSERT(neqs > 0);
    indices.resize(neqs * num_ghost);
    for (size_t node = 0; node < num_ghost; ++node) {
      GO gid = apf::getNumber(m_global_nmbr, ghost[node]);
      for (int eq = 0; eq < neqs; ++eq) {
        indices[get_dof(node, eq, neqs)] = get_gdof(gid, eq, neqs);
      }
    }
    m_maps[GHOST][i] = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
  }
}

void Disc::compute_exporters() {
  resize(m_exporters, m_num_residuals);
  for (int i = 0; i < m_num_residuals; ++i) {
    RCP<const MapT> ghost_map = m_maps[GHOST][i];
    RCP<const MapT> owned_map = m_maps[OWNED][i];
    m_exporters[i] = rcp(new ExportT(ghost_map, owned_map));
  }
}

void Disc::compute_importers() {
  resize(m_importers, m_num_residuals);
  for (int i = 0; i < m_num_residuals; ++i) {
    RCP<const MapT> owned_map = m_maps[OWNED][i];
    RCP<const MapT> ghost_map = m_maps[GHOST][i];
    m_importers[i] = rcp(new ImportT(owned_map, ghost_map));
  }
}

Array1D<size_t> Disc::compute_nentries(int i, int j) {
  int const num_i_eqs = num_eqs(i);
  int const num_j_eqs = num_eqs(j);
  RCP<const MapT> map_i = m_maps[GHOST][i];
  RCP<const MapT> map_j = m_maps[GHOST][j];
  Array1D<size_t> num_entries_per_row(map_i->getLocalNumElements(), 0);
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m_mesh->begin(m_num_dims);
  while ((elem = m_mesh->iterate(elems))) {
    apf::NewArray<int> lids;
    int const num_nodes = apf::getElementNumbers(m_ghost_nmbr, elem, lids);
    for (int n = 0; n < num_nodes; ++n) {
      for (int eq = 0; eq < num_i_eqs; ++eq) {
        int lid = get_dof(lids[n], eq, num_i_eqs);
        num_entries_per_row[lid] += num_nodes * num_j_eqs;
      }
    }
  }
  m_mesh->end(elems);
  return num_entries_per_row;
}

void Disc::compute_ghost_graph(int i, int j) {
  ALWAYS_ASSERT(i < m_num_residuals);
  ALWAYS_ASSERT(j < m_num_residuals);
  int const num_i_eqs = num_eqs(i);
  int const num_j_eqs = num_eqs(j);
  RCP<const MapT> map_i = m_maps[GHOST][i];
  RCP<const MapT> map_j = m_maps[GHOST][j];
  Array1D<size_t> nentries = compute_nentries(i, j);
  Teuchos::ArrayView<const size_t> nentries_per_row(nentries);
  m_graphs[GHOST][i][j] = rcp(new GraphT(map_i, map_j, nentries_per_row));
  RCP<GraphT> graph = m_graphs[GHOST][i][j];
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m_mesh->begin(m_num_dims);
  while ((elem = m_mesh->iterate(elems))) {
    apf::NewArray<long> gids;
    int num_nodes = apf::getElementNumbers(m_global_nmbr, elem, gids);
    for (int node_i = 0; node_i < num_nodes; ++node_i) {
      for (int eq_i = 0; eq_i < num_i_eqs; ++eq_i) {
        GO row = get_gdof(gids[node_i], eq_i, num_i_eqs);
        for (int node_j = 0; node_j < num_nodes; ++node_j) {
          for (int eq_j = 0; eq_j < num_j_eqs; ++eq_j) {
            GO col = get_gdof(gids[node_j], eq_j, num_j_eqs);
            Teuchos::ArrayView<GO> tcol = Teuchos::arrayView(&col, 1);
            graph->insertGlobalIndices(row, tcol);
          }
        }
      }
    }
  }
  m_mesh->end(elems);
  graph->fillComplete(m_maps[OWNED][j], m_maps[OWNED][i]);
}

void Disc::compute_owned_graph(int i, int j) {
  RCP<const MapT> owned_map_i = m_maps[OWNED][i];
  RCP<const MapT> owned_map_j = m_maps[OWNED][j];
  RCP<const ExportT> exporter = m_exporters[i];
  RCP<const GraphT> ghost_graph = m_graphs[GHOST][i][j];
  m_graphs[OWNED][i][j] = rcp(new GraphT(owned_map_i, 0));
  RCP<GraphT> owned_graph = m_graphs[OWNED][i][j];
  owned_graph->doExport(*ghost_graph, *exporter, Tpetra::INSERT);
  owned_graph->fillComplete(owned_map_j, owned_map_i);
}

void Disc::compute_graphs() {
  ALWAYS_ASSERT(m_num_residuals > 0);
  resize(m_graphs[OWNED], m_num_residuals, m_num_residuals);
  resize(m_graphs[GHOST], m_num_residuals, m_num_residuals);
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int j = 0; j < m_num_residuals; ++j) { 
      compute_ghost_graph(i, j);
      compute_owned_graph(i, j);
    }
  }
  apf::destroyGlobalNumbering(m_global_nmbr);
  m_global_nmbr = 0;
}

void Disc::compute_elem_sets() {
  for (int i = 0; i < m_num_elem_sets; ++i) {
    resize(m_elem_sets[ elem_set_name(i) ], 0);
  }
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m_mesh->begin(m_num_dims);
  while ((elem = m_mesh->iterate(elems))) {
    apf::ModelEntity* me = m_mesh->toModel(elem);
    apf::StkModel* stkm = m_sets->invMaps[m_num_dims][me];
    std::string const name = stkm->stkName;
    m_elem_sets[name].push_back(elem);
  }
  m_mesh->end(elems);
}

void Disc::compute_side_sets() {
  for (int i = 0; i < m_num_side_sets; ++i) {
    resize(m_side_sets[ side_set_name(i) ], 0);
  }
  apf::MeshEntity* side;
  apf::MeshIterator* sides = m_mesh->begin(m_num_dims - 1);
  while ((side = m_mesh->iterate(sides))) {
    apf::ModelEntity* me = m_mesh->toModel(side);
    if (!m_sets->invMaps[m_num_dims - 1].count(me)) {
      continue;
    }
    apf::StkModel* stkm = m_sets->invMaps[m_num_dims - 1][me];
    std::string const name = stkm->stkName;
    m_side_sets[name].push_back(side);
  }
  m_mesh->end(sides);
}

void Disc::compute_node_sets() {
  for (int i = 0; i < m_num_node_sets; ++i) {
    resize(m_node_sets[ node_set_name(i) ], 0);
  }
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    apf::MeshEntity* ent = node.entity;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(
        m_mesh, m_sets->invMaps[0], m_mesh->toModel(ent), mset);
    if (mset.empty()) continue;
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      apf::StkModel* stkm = *mit;
      std::string const name = stkm->stkName;
      m_node_sets[name].push_back(node);
    }
  }
}

void Disc::compute_field_node_sets() {
  for (int i = 0; i < m_num_node_sets; ++i) {
    resize(m_node_sets[ node_set_name(i) ], 0);
  }
  for (int i = 0; i < m_num_node_sets; ++i) {
    std::string const name = node_set_name(i);
    std::string const fname = name;
    apf::Field* ns_field = m_mesh->findField(fname.c_str());
    ALWAYS_ASSERT(ns_field);
    apf::DynamicArray<apf::Node> owned;
    apf::getNodes(m_owned_nmbr, owned);
    for (size_t n = 0; n < owned.size(); ++n) {
      apf::Node const node = owned[n];
      apf::MeshEntity* ent = node.entity;
      double const val = apf::getScalar(ns_field, ent, 0);
      if (std::abs(val - 1.0) < 1.0e-12) {
        m_node_sets[name].push_back(node);
      }
    }
    //apf::destroyField(ns_field);
  }
}

void Disc::build_data(int num_residuals, Array1D<int> const& num_eqs) {
  destroy_data();
  m_num_residuals = num_residuals;
  m_num_eqs = num_eqs;
  compute_node_map();
  compute_coords();
  compute_owned_maps();
  compute_ghost_maps();
  compute_exporters();
  compute_importers();
  compute_graphs();
  compute_elem_sets();
  compute_side_sets();
  if (!m_is_null_model) {
    compute_node_sets();
  } else {
    compute_field_node_sets();
  }
}

void Disc::destroy_data() {
  if (m_owned_nmbr) {
    apf::destroyNumbering(m_owned_nmbr);
  }
  if (m_ghost_nmbr) {
    apf::destroyNumbering(m_ghost_nmbr);
  }
  if (m_global_nmbr) {
    apf::destroyGlobalNumbering(m_global_nmbr);
  }
  for (int i = 0; i < num_elem_sets(); ++i) {
    resize(m_elem_sets[elem_set_name(i)], 0);
  }
  for (int i = 0; i < num_side_sets(); ++i) {
    resize(m_side_sets[side_set_name(i)], 0);
  }
  for (int i = 0; i < num_node_sets(); ++i) {
    resize(m_node_sets[node_set_name(i)], 0);
  }
  resize(m_maps[OWNED], 0);
  resize(m_maps[GHOST], 0);
  resize(m_graphs[OWNED], 0, 0);
  resize(m_graphs[GHOST], 0, 0);
  m_node_map = Teuchos::null;
  m_owned_nmbr = nullptr;
  m_ghost_nmbr = nullptr;
  m_global_nmbr = nullptr;
  m_num_residuals = -1;
  m_num_eqs = {};
}

Array2D<LO> Disc::get_element_lids(apf::MeshEntity* e, int i) {
  Array2D<LO> lids;
  apf::NewArray<int> node_ids;
  int const num_i_eqs = num_eqs(i);
  int const num_nodes = apf::getElementNumbers(m_ghost_nmbr, e, node_ids);
  resize(lids, num_nodes, num_i_eqs);
  for (int n = 0; n < num_nodes; ++n) {
    for (int eq = 0; eq < num_i_eqs; ++eq) {
      lids[n][eq] = get_dof(node_ids[n], eq, num_i_eqs);
    }
  }
  return lids;
}

LO Disc::get_lid(apf::Node const& n, int i, int eq) {
  LO const nid = apf::getNumber(m_owned_nmbr, n.entity, n.node, 0);
  int const num_i_eqs = num_eqs(i);
  return get_dof(nid, eq, num_i_eqs);
}

LO Disc::get_lid(apf::MeshEntity* ent, int i, int n, int eq) {
  apf::NewArray<int> node_ids;
  int const num_i_eqs = num_eqs(i);
  apf::getElementNumbers(m_ghost_nmbr, ent, node_ids);
  return get_dof(node_ids[n], eq, num_i_eqs);
}

int Disc::get_value_type(int neqs, int ndims) {
  if (neqs == 1) {
    return apf::SCALAR;
  } else if ((neqs == 2) && (ndims == 2)) {
    return apf::VECTOR;
  } else if ((neqs == 3) && (ndims == 3)) {
    return apf::VECTOR;
  } else {
    return apf::MATRIX;
  }
}

void Disc::create_primal(
    RCP<Residuals<double>> R,
    int step,
    bool use_measured) {
  DEBUG_ASSERT(m_primal.size() == size_t(step));
  Fields fields;
  int const ngr = R->global->num_residuals();
  int const model_form = BASE_MODEL;
  int const nlr = R->local[model_form]->num_residuals();
  resize(fields.global, ngr);
  resize(fields.local[model_form], nlr);
  for (int i = 0; i < ngr; ++i) {
    std::string const name = R->global->resid_name(i);
    std::string const fname = name + "_" + std::to_string(step);
    int const vtype = get_value_type(R->global->num_eqs(i), m_num_dims);
    fields.global[i] = apf::createField(m_mesh, fname.c_str(), vtype, m_gv_shape);
    if (step == 0) {
      apf::zeroField(fields.global[i]);
    } else if (use_measured && i == 0) {
      std::string name = "measured_" + std::to_string(step);
      apf::Field* f_meas = m_mesh->findField(name.c_str());
      ALWAYS_ASSERT(f_meas);
      apf::copyData(fields.global[i], f_meas);
    } else {
      apf::copyData(fields.global[i], m_primal[step - 1].global[i]);
    }
  }
  for (int i = 0; i < nlr; ++i) {
    std::string const name = R->local[model_form]->resid_name(i);
    std::string const fname = name + "_" + std::to_string(step);
    int const vtype = get_value_type(R->local[model_form]->num_eqs(i), m_num_dims);
    fields.local[model_form][i] = apf::createField(m_mesh, fname.c_str(), vtype, m_lv_shape);
    apf::zeroField(fields.local[model_form][i]);
    if (step == 0) {
      apf::zeroField(fields.local[model_form][i]);
    } else {
      apf::copyData(fields.local[model_form][i], m_primal[step - 1].local[model_form][i]);
    }
  }
  m_primal.push_back(fields);
}

// TODO: add in local residual prolong operation
void Disc::create_primal_fine_model(
    RCP<Residuals<double>> R,
    int num_steps) {
  int const base_model_form = BASE_MODEL;
  int const fine_model_form = FINE_MODEL;
  int const nlr = R->local[base_model_form]->num_residuals();
  for (int step = 0; step < num_steps; ++step ) {
    resize(m_primal[step].local[fine_model_form], nlr);
    for (int i = 0; i < nlr; ++i) {
      std::string const name = R->local[base_model_form]->resid_name(i);
      std::string const fname = "prolonged_" + name + "_" + std::to_string(step);
      int const vtype = get_value_type(R->local[base_model_form]->num_eqs(i), m_num_dims);
      m_primal[step].local[fine_model_form][i] = apf::createField(m_mesh,
          fname.c_str(), vtype, m_lv_shape);
      apf::copyData(m_primal[step].local[fine_model_form][i],
          m_primal[step].local[base_model_form][i]);
    }
  }
}

static Array1D<std::string> get_vf_expressions(
    ParameterList const& vf_list) {
  Array1D<std::string> vf_expressions;
  for (auto it = vf_list.begin(); it != vf_list.end(); ++it) {
    auto pentry = vf_list.entry(it);
    std::string const val = Teuchos::getValue<std::string>(pentry);
    vf_expressions.push_back(val);
  }
  return vf_expressions;
}

Array1D<double> Disc::get_vals(
    Array1D<std::string> const& val_expressions,
    apf::Node const& node) {
  Array1D<double> vals;
  apf::Vector3 x;
  apf::MeshEntity* e = node.entity;
  int const ent_node = node.node;
  m_mesh->getPoint(e, ent_node, x);
  for (auto& val : val_expressions) {
    double const v = eval(val, x[0], x[1], x[2], 0.);
    vals.push_back(v);
  }
  return vals;
}

void Disc::create_virtual(
    RCP<Residuals<double>> R,
    ParameterList const& vf_list) {
  Fields fields;
  int const ngr = R->global->num_residuals();
  ALWAYS_ASSERT(ngr == 1);
  resize(fields.virtual_field, ngr);
  std::string const name = R->global->resid_name(0);
  std::string const fname = "virtual_" + name;
  int const vtype = get_value_type(R->global->num_eqs(0), m_num_dims);
  fields.virtual_field[0] = apf::createField(
      m_mesh, fname.c_str(), vtype, m_gv_shape);
  Array1D<std::string> const vf_expressions = get_vf_expressions(vf_list);
  ALWAYS_ASSERT(vf_expressions.size() == m_num_dims);
  Array1D<double> vf_vals;
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    apf::MeshEntity* ent = node.entity;
    int const ent_node = node.node;
    vf_vals = get_vals(vf_expressions, node);
    apf::setComponents(fields.virtual_field[0], ent, ent_node, &(vf_vals[0]));
  }
  apf::synchronize(fields.virtual_field[0]);
  m_virtual.push_back(fields);
}

void Disc::set_virtual_field_from_node_set(
    std::string const node_set_name,
    int vf_component) {
  apf::Field* node_set_field = m_mesh->findField(node_set_name.c_str());
  ALWAYS_ASSERT(node_set_field);
  auto vf = m_virtual[0].virtual_field[0];
  int const neqs = apf::countComponents(vf);
  double node_set_field_val;
  Array1D<double> vf_vals(neqs, 0.);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr, owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    apf::MeshEntity* ent = node.entity;
    int const ent_node = node.node;
    node_set_field_val = apf::getScalar(node_set_field, ent, ent_node);
    for (int d = 0; d < m_num_dims; ++d) {
      if (d == vf_component) {
        vf_vals[d] = node_set_field_val;
      } else {
        vf_vals[d] = 0.;
      }
    }
    apf::setComponents(vf, ent, ent_node, &(vf_vals[0]));
  }
  apf::synchronize(vf);
}

static void destroy_fields(Fields& fields) {
  for (size_t i = 0; i < fields.global.size(); ++i) {
    apf::destroyField(fields.global[i]);
  }
  for (size_t m = 0; m < 2; ++m) {
    for (size_t i = 0; i < fields.local[m].size(); ++i) {
      apf::destroyField(fields.local[m][i]);
    }
  }
  for (size_t i = 0; i < fields.virtual_field.size(); ++i) {
    apf::destroyField(fields.virtual_field[i]);
  }
}

void Disc::destroy_primal(bool keep_ic) {
  int const num_steps = m_primal.size();
  // Keep the initial condition
  int start = 0;
  if (keep_ic) start = 1;
  for (int n = start; n < num_steps; ++n) {
    destroy_fields(m_primal[n]);
  }
  if (keep_ic) {
    m_primal.resize(1);
  } else {
    m_primal.resize(0);
  }

  int const num_steps_fine = m_primal_fine.size();
  // Keep the initial condition
  start = 0;
  if (keep_ic) start = 1;
  for (int n = start; n < num_steps_fine; ++n) {
    destroy_fields(m_primal_fine[n]);
  }
  if (keep_ic) {
    m_primal_fine.resize(1);
  } else {
    m_primal_fine.resize(0);
  }
}

void Disc::create_adjoint(
    RCP<Residuals<double>> R,
    int const num_steps,
    int const model_form) {
  for (int step = 0; step <= num_steps; ++step) {
    Fields fields;
    int const ngr = R->global->num_residuals();
    int const nlr = R->local[model_form]->num_residuals();
    resize(fields.global, ngr);
    resize(fields.local[model_form], nlr);
    for (int i = 0; i < ngr; ++i) {
      std::string const name = R->global->resid_name(i);
      std::string const fname = "adjoint_" + name + "_" + std::to_string(step);
      int const vtype = get_value_type(R->global->num_eqs(i), m_num_dims);
      fields.global[i] = apf::createField(m_mesh, fname.c_str(), vtype, m_gv_shape);
      apf::zeroField(fields.global[i]);
    }
    for (int i = 0; i < nlr; ++i) {
      std::string const name = R->local[model_form]->resid_name(i);
      std::string const fname = "adjoint_" + name + "_" + std::to_string(step);
      int const vtype = get_value_type(R->local[model_form]->num_eqs(i), m_num_dims);
      fields.local[model_form][i] = apf::createField(m_mesh, fname.c_str(), vtype, m_lv_shape);
      apf::zeroField(fields.local[model_form][i]);
    }
    m_adjoint.push_back(fields);
  }
}

void Disc::destroy_adjoint() {
  int const num_steps = m_adjoint.size();
  for (int n = 0; n < num_steps; ++n) {
    destroy_fields(m_adjoint[n]);
  }
  m_adjoint.resize(0);

  int const num_steps_fine = m_adjoint_fine.size();
  for (int n = 0; n < num_steps_fine; ++n) {
    destroy_fields(m_adjoint_fine[n]);
  }
  m_adjoint_fine.resize(0);
}

void Disc::destroy_virtual() {
  int const num_steps = m_virtual.size();
  for (int n = 0; n < num_steps; ++n) {
    destroy_fields(m_virtual[n]);
  }
  m_virtual.resize(0);
}

apf::DynamicArray<apf::Node> Disc::get_owned_nodes() {
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(m_owned_nmbr, nodes);
  return nodes;
}

void Disc::add_to_soln(
    Array1D<apf::Field*>& x,
    Array1D<RCP<VectorT>> const& dx,
    double const alpha) {

  // sanity check
  int const num_resids = m_num_residuals;
  DEBUG_ASSERT(dx.size() == size_t(num_resids));

  // get the nodes associated with the nodes in the mesh
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(m_owned_nmbr, nodes);

  // grab data from the blocked vector
  Array1D<Teuchos::ArrayRCP<double>> dx_data(m_num_residuals);
  for (int i = 0; i < num_resids; ++i) {
    dx_data[i] = dx[i]->get1dViewNonConst();
  }

  // storage used below
  Array1D<double> sol_comps(3);

  // loop over all the nodes in the discretization
  for (size_t n = 0; n < nodes.size(); ++n) {

    // get information about the current node
    apf::Node node = nodes[n];
    apf::MeshEntity* ent = node.entity;
    int const ent_node = node.node;

    // loop over the global residuals
    for (int i = 0; i < num_resids; ++i) {

      // get the field corresponding to this residual at this ste
      apf::Field* f = x[i];

      // get the solution for the current residual at the node
      apf::getComponents(f, ent, ent_node, &(sol_comps[0]));

      // add the increment to the current solution
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        LO row = get_lid(node, i, eq);
        sol_comps[eq] += alpha * dx_data[i][row];
      }

      // set the added solution for the current residual at the node
      apf::setComponents(f, ent, ent_node, &(sol_comps[0]));

    }
  }

  // synchronize the fields in parallel
  for (int i =0 ; i < num_resids; ++i) {
    apf::synchronize(x[i]);
  }

}

void Disc::populate_vector(
    Array1D<apf::Field*> const& fields,
    Array1D<RCP<VectorT>> vec[]) {

  auto& vec_owned = vec[OWNED];
  auto& vec_ghost = vec[GHOST];
  int const num_comps = fields.size();
  DEBUG_ASSERT(vec_owned.size() == num_comps);

  // get the nodes associated with the nodes in the mesh
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(m_owned_nmbr, nodes);

  // grab data from the blocked vector
  Array1D<Teuchos::ArrayRCP<double>> vec_data(num_comps);
  for (int i = 0; i < num_comps; ++i) {
    vec_data[i] = vec_owned[i]->get1dViewNonConst();
  }

  // loop over all the nodes in the discretization
  for (size_t n = 0; n < nodes.size(); ++n) {

    // get information about the current node
    apf::Node node = nodes[n];
    apf::MeshEntity* ent = node.entity;
    int const ent_node = node.node;

    // loop over the global residuals
    for (int i = 0; i < num_comps; ++i) {

      // get the field corresponding to this residual at this step
      apf::Field* f = fields[i];

      // get the solution for the current residual at the node
      int const neqs = apf::countComponents(f);
      Array1D<double> sol_comps(neqs, 0.);
      apf::getComponents(f, ent, ent_node, &(sol_comps[0]));

      // set the data in the parallel vector
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        LO row = get_lid(node, i, eq);
        vec_data[i][row] = sol_comps[eq];
      }
    }
  }

  // scatter to ghosted nodes
  for (int i = 0; i < num_comps; ++i) {
    RCP<VectorT> vec_i_owned = vec_owned[i];
    RCP<VectorT> vec_i_ghost = vec_ghost[i];
    vec_i_ghost->doImport(*vec_i_owned, *(m_importers[i]), Tpetra::INSERT);
  }

}

void Disc::populate_field(
    Array1D<RCP<VectorT>> const& owned_vec,
    Array1D<apf::Field*>& f) {

  int const num_comps = f.size();
  DEBUG_ASSERT(owned_vec.size() == num_comps);

  // get the nodes associated with the nodes in the mesh
  apf::DynamicArray<apf::Node> nodes;
  apf::getNodes(m_owned_nmbr, nodes);

  for (int i = 0; i < num_comps; ++i) {

    int const neqs = apf::countComponents(f[i]);
    auto vec_data = owned_vec[i]->get1dView();
    Array1D<double> vals(neqs, 0.);

    for (auto& node : nodes) {
      apf::MeshEntity* ent = node.entity;
      int const local_node = node.node;
      for (int eq = 0; eq < neqs; ++eq) {
        LO const row = get_lid(node, i, eq);
        vals[eq] = vec_data[row];
      }
      apf::setComponents(f[i], ent, local_node, &(vals[0]));
    }

    apf::synchronize(f[i]);
  }
}

void Disc::create_verification_data(int model_form) {

  // gather some data
  int const nsteps = m_primal.size();
  int const ngr = m_primal[0].global.size();
  int const nlr = m_primal[0].local[model_form].size();

  // create the fine fields by copying the prolonged coarse fields
  for (int step = 0; step < nsteps; ++step) {
    Fields fields;
    for (int i = 0; i < ngr; ++i) {
      std::string name = apf::getName(m_primal[step].global[i]);
      apf::Field* coarse = m_mesh->findField(name.c_str());
      name = "fine_" + name;
      int const vtype = apf::getValueType(coarse);
      apf::Field* fine = apf::createField(m_mesh, name.c_str(), vtype, m_gv_shape);
      apf::zeroField(fine);
      fields.global.push_back(fine);
    }
    for (int i = 0; i < nlr; ++i) {
      std::string name = apf::getName(m_primal[step].local[model_form][i]);
      apf::Field* coarse = m_mesh->findField(name.c_str());
      name = "fine_" + name;
      int const vtype = apf::getValueType(coarse);
      apf::Field* fine = apf::createField(m_mesh, name.c_str(), vtype, m_lv_shape);
      apf::zeroField(fine);
      if (name == "fine_Ie_0" || name == "fine_lambda_z_0") {
        apf::MeshEntity* elem;
        apf::MeshIterator* elems = m_mesh->begin(m_num_dims);
        while ((elem = m_mesh->iterate(elems))) {
          apf::setScalar(fine, elem, 0, 1.);
        }
        m_mesh->end(elems);
      }
      fields.local[model_form].push_back(fine);
    }
    m_primal_fine.push_back(fields);
  }

  // create the branch paths
  m_branch_paths.resize(nsteps);
  for (int step = 0; step < nsteps; ++step) {
    m_branch_paths[step].resize(m_num_elem_sets);
    for (int set = 0; set < m_num_elem_sets; ++set) {
      std::string const es_name = elem_set_name(set);
      int const nelems = elems(es_name).size();
      m_branch_paths[step][set].resize(nelems);
    }
  }

}

void Disc::initialize_primal_fine(
    RCP<Residuals<double>> R,
    int step,
    int model_form) {
  int const ngr = R->global->num_residuals();
  int const nlr = R->local[model_form]->num_residuals();
  for (int i = 0; i < ngr; ++i) {
    apf::copyData(m_primal_fine[step].global[i], m_primal_fine[step - 1].global[i]);
  }
  for (int i = 0; i < nlr; ++i) {
    apf::copyData(m_primal_fine[step].local[model_form][i],
        m_primal_fine[step - 1].local[model_form][i]);
  }
}

}
