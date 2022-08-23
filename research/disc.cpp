#include <gmi_mesh.h>
#include <gmi_null.h>
#include <PCU.h>
#include <Tpetra_Core.hpp>
#include "control.hpp"
#include "disc.hpp"

namespace calibr8 {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("geom file", "");
  p.set<std::string>("mesh file", "");
  p.set<std::string>("assoc file", "");
  p.sublist("analytic node sets");
  return p;
}

static apf::StkModels* read_sets(apf::Mesh* m, ParameterList const& p) {
  apf::StkModels* sets = new apf::StkModels;
  std::string const fn = p.get<std::string>("assoc file");
  char const* filename = fn.c_str();
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
  if (geom_file == ".null") return true;
  else return false;
}

Disc::Disc(ParameterList const& params) {
  params.validateParameters(get_valid_params(), 0);
  m_params = params;
  load_mesh(&m_mesh, params);
  m_sets = read_sets(m_mesh, params);
  m_is_null = is_null_model(params);
  apf::reorderMdsMesh(m_mesh);
  initialize();
}

Disc::~Disc() {
  destroy_data();
  m_mesh->destroyNative();
  apf::destroyMesh(m_mesh);
  delete m_sets;
}

void Disc::initialize() {
  m_num_dims = m_mesh->getDimension();
  m_num_elem_sets = m_sets->models[m_num_dims].size();
  m_num_side_sets = m_sets->models[m_num_dims-1].size();
  m_num_node_sets = m_sets->models[0].size();
  m_shape[COARSE] = apf::getLagrange(1);
  m_shape[FINE] = apf::getSerendipity();
  m_comm = Tpetra::getDefaultComm();
}

void Disc::build_data(int neqs) {
  destroy_data();
  m_num_eqs = neqs;
  for (int space = 0; space < NUM_SPACE; ++space) {
    compute_node_map(space);
    compute_coords(space);
    compute_owned_map(space);
    compute_ghost_map(space);
    compute_exporter(space);
    compute_importer(space);
    compute_ghost_graph(space);
    compute_owned_graph(space);
    compute_node_sets(space);
  }
  compute_elem_sets();
  compute_side_sets();
}

void Disc::destroy_data() {
  m_elem_sets.clear();
  m_side_sets.clear();
  for (int space=0; space < NUM_SPACE; ++space) {
    if (m_owned_nmbr[space]) apf::destroyNumbering(m_owned_nmbr[space]);
    if (m_ghost_nmbr[space]) apf::destroyNumbering(m_ghost_nmbr[space]);
    if (m_global_nmbr[space]) apf::destroyGlobalNumbering(m_global_nmbr[space]);
    m_node_sets[space].clear();
    m_owned_nmbr[space] = nullptr;
    m_ghost_nmbr[space] = nullptr;
    m_global_nmbr[space] = nullptr;
    m_num_eqs = -1;
    m_exporters[space] = Teuchos::null;
    for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
      m_maps[space][distrib] = Teuchos::null;
      m_graphs[space][distrib] = Teuchos::null;
    }
  }
}

int Disc::order(int space) {
  if (space == COARSE) return 1;
  if (space == FINE) return 2;
  return -1;
}

int Disc::get_num_nodes(int space) {
  int type = -1;
  if (m_num_dims == 2) type = apf::Mesh::TRIANGLE;
  if (m_num_dims == 3) type = apf::Mesh::TET;
  apf::FieldShape* shape = m_shape[space];
  apf::EntityShape* ent_shape = shape->getEntityShape(type);
  int const nnodes = ent_shape->countNodes();
  return nnodes;
}

int Disc::get_space(apf::FieldShape* shape) {
  if (shape == m_shape[COARSE]) return COARSE;
  if (shape == m_shape[FINE]) return FINE;
  return -1;
}

std::string Disc::space_name(int space) {
  if (space == COARSE) return "H";
  if (space == FINE) return "h";
  return "";
}

void Disc::change_shape(int space) {
  apf::FieldShape* current_shape = m_mesh->getShape();
  apf::FieldShape* desired_shape = shape(space);
  if (current_shape != desired_shape) {
    m_mesh->changeShape(desired_shape, true);
  }
}

std::string Disc::elem_set_name(int es_idx) const {
  ASSERT(es_idx < m_num_elem_sets);
  return m_sets->models[m_num_dims][es_idx]->stkName;
}

std::string Disc::side_set_name(int ss_idx) const {
  ASSERT(ss_idx < m_num_side_sets);
  return m_sets->models[m_num_dims-1][ss_idx]->stkName;
}

std::string Disc::node_set_name(int ns_idx) const {
  ASSERT(ns_idx < m_num_node_sets);
  return m_sets->models[0][ns_idx]->stkName;
}

int Disc::elem_set_idx(std::string const& esn) const {
  int idx = -1;
  for (int i =0 ; i < m_num_elem_sets; ++i) {
    if (esn == m_sets->models[m_num_dims][i]->stkName) {
      idx = i;
    }
  }
  ASSERT(idx > -1);
  return idx;
}

int Disc::side_set_idx(std::string const& ssn) const {
  int idx = -1;
  for (int i = 0; i < m_num_side_sets; ++i) {
    if (ssn == m_sets->models[m_num_dims-1][i]->stkName) {
      idx = i;
    }
  }
  ASSERT(idx > -1);
  return idx;
}

int Disc::node_set_idx(std::string const& nsn) const {
  int idx = -1;
  for (int i = 0; i < m_num_node_sets; ++i) {
    if (nsn == m_sets->models[0][i]->stkName) {
      idx = i;
    }
  }
  ASSERT(idx > -1);
  return idx;
}

ElemSet const& Disc::elems(std::string const& name) {
  ASSERT(m_elem_sets.count(name));
  return m_elem_sets[name];
}

SideSet const& Disc::sides(std::string const& name) {
  ASSERT(m_side_sets.count(name));
  return m_side_sets[name];
}

NodeSet const& Disc::nodes(int space, std::string const& name) {
  ASSERT(m_node_sets[space].count(name));
  return m_node_sets[space][name];
}

static LO get_dof(LO nid, int eq, int neq) {
  return nid*neq + eq;
}

static GO get_gdof(GO nid, int eq, int neq) {
  return nid*neq + eq;
}

std::vector<LO> Disc::get_elem_lids(int space, apf::MeshEntity* e) {
  int dof = 0;
  apf::NewArray<int> node_ids;
  apf::Numbering* nmbr = m_ghost_nmbr[space];
  int const num_nodes = apf::getElementNumbers(nmbr, e, node_ids);
  ASSERT(num_nodes == get_num_nodes(space));
  std::vector<LO> lids(num_nodes*m_num_eqs);
  for (int n = 0; n < num_nodes; ++n) {
    for (int eq = 0; eq < m_num_eqs; ++eq) {
      lids[dof++] = get_dof(node_ids[n], eq, m_num_eqs);
    }
  }
  return lids;
}

LO Disc::get_lid(int space, apf::Node const& n, int eq) {
  apf::Numbering* nmbr = m_owned_nmbr[space];
  LO const nid = apf::getNumber(nmbr, n.entity, n.node, 0);
  return get_dof(nid, eq, m_num_eqs);
}

LO Disc::get_lid(int space, apf::MeshEntity* ent, int n, int eq) {
  apf::NewArray<int> node_ids;
  apf::Numbering* nmbr = m_ghost_nmbr[space];
  apf::getElementNumbers(nmbr, ent, node_ids);
  return get_dof(node_ids[n], eq, m_num_eqs);
}

apf::DynamicArray<apf::Node> Disc::owned_nodes(int space) {
  ASSERT(m_owned_nmbr[space]);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr[space], owned);
  return owned;
}

void Disc::compute_node_map(int space) {
  ASSERT(!m_owned_nmbr[space]);
  ASSERT(!m_global_nmbr[space]);
  apf::FieldShape* shape = m_shape[space];
  std::string const name = "owned_" + space_name(space);
  m_owned_nmbr[space] = apf::numberOwnedNodes(m_mesh, name.c_str(), shape);
  m_global_nmbr[space] = apf::makeGlobal(m_owned_nmbr[space], false);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_global_nmbr[space], owned);
  size_t const num_owned = owned.size();
  Teuchos::Array<GO> indices(num_owned);
  for (size_t n = 0; n < num_owned; ++n) {
    indices[n] = apf::getNumber(m_global_nmbr[space], owned[n]);
  }
  m_node_maps[space] = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
}

void Disc::compute_coords(int space) {
  change_shape(space);
  m_coords[space] = rcp(new MultiVectorT(m_node_maps[space], m_num_dims, false));
  apf::Vector3 x(0,0,0);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr[space], owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node const node = owned[n];
    m_mesh->getPoint(node.entity, node.node, x);
    for (int dim = 0; dim < m_num_dims; ++dim) {
      m_coords[space]->replaceLocalValue(n, dim, x[dim]);
    }
  }
  m_mesh->changeShape(m_shape[COARSE], true);
}

void Disc::compute_owned_map(int space) {
  ASSERT(m_num_eqs > 0);
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_global_nmbr[space], owned);
  size_t const num_owned = owned.size();
  Teuchos::Array<GO> indices;
  indices.resize(num_owned * m_num_eqs);
  for (size_t node = 0; node < num_owned; ++node) {
    GO gid = apf::getNumber(m_global_nmbr[space], owned[node]);
    for (int eq = 0; eq < m_num_eqs; ++eq) {
      indices[get_dof(node, eq, m_num_eqs)] = get_gdof(gid, eq, m_num_eqs);
    }
  }
  m_maps[space][OWNED] = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
  apf::synchronize(m_global_nmbr[space]);
}

void Disc::compute_ghost_map(int space) {
  ASSERT(m_num_eqs > 0);
  ASSERT(!m_ghost_nmbr[space]);
  apf::FieldShape* shape = m_shape[space];
  std::string const name = "ghost_" + space_name(space);
  m_ghost_nmbr[space] = apf::numberOverlapNodes(m_mesh, name.c_str(), shape);
  apf::DynamicArray<apf::Node> ghost;
  apf::getNodes(m_ghost_nmbr[space], ghost);
  size_t const num_ghost = ghost.size();
  Teuchos::Array<GO> indices;
  indices.resize(num_ghost * m_num_eqs);
  for (size_t node = 0; node < num_ghost; ++node) {
    GO gid = apf::getNumber(m_global_nmbr[space], ghost[node]);
    for (int eq = 0; eq < m_num_eqs; ++eq) {
      indices[get_dof(node, eq, m_num_eqs)] = get_gdof(gid, eq, m_num_eqs);
    }
  }
  m_maps[space][GHOST] = Tpetra::createNonContigMap<LO, GO>(indices, m_comm);
}

void Disc::compute_exporter(int space) {
  RCP<const MapT> ghost_map = m_maps[space][GHOST];
  RCP<const MapT> owned_map = m_maps[space][OWNED];
  m_exporters[space] = rcp(new ExportT(ghost_map, owned_map));
}

void Disc::compute_importer(int space) {
  RCP<const MapT> ghost_map = m_maps[space][GHOST];
  RCP<const MapT> owned_map = m_maps[space][OWNED];
  m_importers[space] = rcp(new ImportT(owned_map, ghost_map));
}

void Disc::compute_ghost_graph(int space) {
  int const est = 500;
  RCP<const MapT> ghost_map = m_maps[space][GHOST];
  m_graphs[space][GHOST] = rcp(new GraphT(ghost_map, est));
  apf::MeshEntity* elem;
  auto elems = m_mesh->begin(m_num_dims);
  while ((elem = m_mesh->iterate(elems))) {
    apf::NewArray<long> gids;
    int nnodes = apf::getElementNumbers(m_global_nmbr[space], elem, gids);
    for (int i = 0; i < nnodes; ++i) {
      for (int j = 0; j < m_num_eqs; ++j) {
        GO row = get_gdof(gids[i], j, m_num_eqs);
        for (int k = 0; k < nnodes; ++k) {
          for (int l = 0; l < m_num_eqs; ++l) {
            GO col = get_gdof(gids[k], l, m_num_eqs);
            auto col_av = Teuchos::arrayView(&col, 1);
            m_graphs[space][GHOST]->insertGlobalIndices(row, col_av);
  }}}}}
  m_mesh->end(elems);
  m_graphs[space][GHOST]->fillComplete();
}

void Disc::compute_owned_graph(int space) {
  int const est = 500;
  RCP<const MapT> owned_map = m_maps[space][OWNED];
  RCP<const GraphT> ghost_graph = m_graphs[space][GHOST];
  RCP<const ExportT> exporter = m_exporters[space];
  m_graphs[space][OWNED] = rcp(new GraphT(owned_map, est));
  m_graphs[space][OWNED]->doExport(*ghost_graph, *exporter, Tpetra::INSERT);
  m_graphs[space][OWNED]->fillComplete();
  apf::destroyGlobalNumbering(m_global_nmbr[space]);
  m_global_nmbr[space] = nullptr;
}

void Disc::compute_model_node_sets(int space) {
  m_node_sets[space].clear();
  apf::DynamicArray<apf::Node> owned;
  apf::getNodes(m_owned_nmbr[space], owned);
  for (size_t n = 0; n < owned.size(); ++n) {
    apf::Node node = owned[n];
    apf::MeshEntity* ent = node.entity;
    std::set<apf::StkModel*> mset;
    apf::collectEntityModels(
        m_mesh, m_sets->invMaps[0], m_mesh->toModel(ent), mset);
    if (mset.empty()) continue;
    APF_ITERATE(std::set<apf::StkModel*>, mset, mit) {
      auto ns = *mit;
      auto nsn = ns->stkName;
      m_node_sets[space][nsn].push_back(node);
    }
  }
}

void Disc::compute_analytic_node_sets(int space) {
  m_node_sets[space].clear();
  change_shape(space);
  auto& analytic = m_params.sublist("analytic node sets");
  for (int i = 0; i < m_num_node_sets; ++i) {
    std::string const name = node_set_name(i);
    std::string const expr = analytic.get<std::string>(name);
    apf::Vector3 x;
    apf::DynamicArray<apf::Node> owned;
    apf::getNodes(m_owned_nmbr[space], owned);
    for (size_t n = 0; n < owned.size(); ++n) {
      apf::Node const node = owned[n];
      m_mesh->getPoint(node.entity, node.node, x);
      double const val = eval(expr, x[0], x[1], x[2], 0.);
      if (std::abs(val - 1.) < 1.0e-12) {
        m_node_sets[space][name].push_back(node);
      }
    }
  }
}

void Disc::compute_node_sets(int space) {
  if (m_is_null) compute_analytic_node_sets(space);
  else compute_model_node_sets(space);
}

void Disc::compute_elem_sets() {
  m_elem_sets.clear();
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
  m_side_sets.clear();
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

}
