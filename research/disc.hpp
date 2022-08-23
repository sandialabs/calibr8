#pragma once

#include <apf.h>
#include <apfAlbany.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfShape.h>
#include <Teuchos_ParameterList.hpp>
#include "defines.hpp"

namespace calibr8 {

using ElemSet = std::vector<apf::MeshEntity*>;
using SideSet = std::vector<apf::MeshEntity*>;
using NodeSet = std::vector<apf::Node>;
using ElemSets = std::map<std::string, ElemSet>;
using SideSets = std::map<std::string, SideSet>;
using NodeSets = std::map<std::string, NodeSet>;

class Disc {
  
  public:

    Disc() = default;
    Disc(ParameterList const& p);
    ~Disc();

    void build_data(int neqs);
    void destroy_data();

    int num_dims() const { return m_num_dims; }
    int num_eqs() const { return m_num_eqs; }
    int num_elem_sets() const { return m_num_elem_sets; }
    int num_side_sets() const { return m_num_side_sets; }
    int num_node_sets() const { return m_num_node_sets; }

    apf::Mesh2* apf_mesh() { return m_mesh; }
    apf::FieldShape* shape(int space) { return m_shape[space]; }

    int order(int space);
    int get_num_nodes(int space, apf::MeshEntity* e);
    int get_space(apf::FieldShape* shape);
    std::string space_name(int space);
    void change_shape(int space);

    std::string elem_set_name(int i) const;
    std::string side_set_name(int i) const;
    std::string node_set_name(int i) const;

    int elem_set_idx(std::string const& name) const;
    int side_set_idx(std::string const& name) const;
    int node_set_idx(std::string const& name) const;

    ElemSet const& elems(std::string const& name);
    SideSet const& sides(std::string const& name);
    NodeSet const& nodes(int space, std::string const& name);

    RCP<const MapT> map(int space, int distrib) const { return m_maps[space][distrib]; }
    RCP<const GraphT> graph(int space, int distrib) const { return m_graphs[space][distrib]; }
    RCP<const ExportT> exporter(int space) const { return m_exporters[space]; }
    RCP<const ImportT> importer(int space) const { return m_importers[space]; }
    RCP<MultiVectorT> coords(int space) const { return m_coords[space]; }

    std::vector<LO> get_elem_lids(int space, apf::MeshEntity* elem);
    LO get_lid(int space, apf::Node const& n, int eq);
    LO get_lid(int space, apf::MeshEntity* ent, int n, int eq);
    apf::DynamicArray<apf::Node> owned_nodes(int space);

  private:

    void initialize();

    void compute_node_map(int space);
    void compute_coords(int space);
    void compute_owned_map(int space);
    void compute_ghost_map(int space);
    void compute_exporter(int space);
    void compute_importer(int space);
    void compute_owned_graph(int space);
    void compute_ghost_graph(int space);
    void compute_model_node_sets(int space);
    void compute_analytic_node_sets(int space);
    void compute_node_sets(int space);
    void compute_elem_sets();
    void compute_side_sets();

  private:

    int m_num_dims = -1;
    int m_num_eqs = -1;

    int m_num_elem_sets = -1;
    int m_num_side_sets = -1;
    int m_num_node_sets = -1;

    apf::Mesh2* m_mesh = nullptr;
    apf::StkModels* m_sets = nullptr;
    apf::FieldShape* m_shape[NUM_SPACE] = {nullptr};
    apf::Numbering* m_owned_nmbr[NUM_SPACE] = {nullptr};
    apf::Numbering* m_ghost_nmbr[NUM_SPACE] = {nullptr};
    apf::GlobalNumbering* m_global_nmbr[NUM_SPACE] = {nullptr};

    RCP<const Comm> m_comm;
    RCP<const MapT> m_node_maps[NUM_SPACE];
    RCP<const MapT> m_maps[NUM_SPACE][NUM_DISTRIB];
    RCP<GraphT> m_graphs[NUM_SPACE][NUM_DISTRIB];
    RCP<const ExportT> m_exporters[NUM_SPACE];
    RCP<const ImportT> m_importers[NUM_SPACE];
    RCP<MultiVectorT> m_coords[NUM_SPACE];

    ElemSets m_elem_sets;
    SideSets m_side_sets;
    NodeSets m_node_sets[NUM_SPACE];

    bool m_is_null = false;
    ParameterList m_params;

};

}
