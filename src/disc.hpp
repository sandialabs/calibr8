#pragma once

//! \file disc.hpp
//! \brief A discretization container

#include <apf.h>
#include <apfAlbany.h>
#include <apfMDS.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include <apfShape.h>
#include <Teuchos_ParameterList.hpp>
#include "arrays.hpp"
#include "defines.hpp"

namespace calibr8 {

//! \cond
// forward declarations
template <typename T> class Residuals;
//! \endcond

//! \brief Parallel distribution type
enum { OWNED, GHOST, NUM_DISTRIB };

//! \brief Discretization type
enum { COARSE, NESTED, VERIFICATION, TRUTH };

//! \brief Element set definition
using ElemSet = Array1D<apf::MeshEntity*>;

//! \brief Side set definition
using SideSet = Array1D<apf::MeshEntity*>;

//! \brief Node set definition
using NodeSet = Array1D<apf::Node>;

//! \brief A collection of element sets defined by name
using ElemSets = std::map<std::string, ElemSet>;

//! \brief A collection of side sets defined by name
using SideSets = std::map<std::string, SideSet>;

//! \brief A collection of node sets defined by name
using NodeSets = std::map<std::string, NodeSet>;

//! \brief Data containers for fields
class Fields {

  public:

    //! \brief The global fields stored by residual index
    Array1D<apf::Field*> global;

    //! \brief The local fields stored by residual index
    Array1D<apf::Field*> local;

};

//! \brief A discretization object
//! \details The discretization object couples a mesh data structure to
//! Tpetra linear algebra data structures through user-specified global
//! residual information. In particular, the number of global residuals
//! and the number of equations for those global residuals.
class Disc {

  public:

    //! \brief Default constructor
    Disc() = default;

    //! \brief Construct the discretization from a parameterlist
    //! \param p The highest level parameterlist
    //! \details The input parameterlist expects the following parameters
    //! to exist:
    //!   geom file: example_geom.dmg
    //!   mesh file: example_mesh.smb
    //!   assoc file: example_assoc.txt
    Disc(ParameterList const& p);

    //! \brief Destroy the discretization
    ~Disc();

    //! \brief Build the data required for coarse-space solves given the mesh
    //! \param num_residuals The number of PDE residuals
    //! \param num_eqs The number of equations per each PDE residual
    //! \details For use on initialization and after mesh adaptation
    void build_data(int num_residuals, std::vector<int> const& num_eqs);

    //! \brief Get the number of spatial dimensions of the mesh
    int num_dims() const { return m_num_dims; }

    //! \brief Get the number of element sets in the mesh
    int num_elem_sets() const { return m_num_elem_sets; }

    //! \brief Get the number of side sets in the mesh
    int num_side_sets() const { return m_num_side_sets; }

    //! \brief Get the number of node sets in the mesh
    int num_node_sets() const { return m_num_node_sets; }

    //! \brief Get the discretization type (COARSE/NESTED/VERIFICATION)
    int type() const { return m_disc_type; }

    //! \brief Get the underlying apf mesh
    apf::Mesh2* apf_mesh() { return m_mesh; }

    //! \brief Get the owned numbering of the nodes on the apf mesh
    apf::Numbering* owned_numbering() { return m_owned_nmbr; }

    //! \brief Get the name of the ith element set
    //! \param i The index of the element set
    std::string elem_set_name(int i) const;

    //! \brief Get the name of the ith side set
    //! \param i The index of the side set
    std::string side_set_name(int i) const;

    //! \brief Get the name of the ith node set
    //! \param i The index of the node set
    std::string node_set_name(int i) const;

    //! \brief Get the index of an element set by name
    //! \param name The name of the element set of interest
    int elem_set_idx(std::string const& name) const;

    //! \brief Get the index of a side set by name
    //! \param name The name of the side set of interest
    int side_set_idx(std::string const& name) const;

    //! \brief Get the index of a node set by name
    //! \param name The name of the node set of interest
    int node_set_idx(std::string const& name) const;

    //! \brief Get the elements in an element set by name
    //! \param name The name of the element set of interest
    ElemSet const& elems(std::string const& name);

    //! \brief Get the sides in a side set by name
    //! \param name The name of the side set of interest
    SideSet const& sides(std::string const& name);

    //! \brief Get the nodes in a node set by name
    //! \param name The name of the node set of interest
    NodeSet const& nodes(std::string const& name);

    //! \brief Get the number of global variable nodes
    int num_gv_nodes_per_elem() const { return m_num_gv_nodes; }

    //! \brief Get the number of local variable nodes
    int num_lv_nodes_per_elem() const { return m_num_lv_nodes; }

    //! \brief Get the global variable shape functions
    apf::FieldShape* gv_shape() { return m_gv_shape; }

    //! \brief Get the local variable shape functions
    apf::FieldShape* lv_shape() { return m_lv_shape; }

    //! \brief Get the nodal coordinates for nodal fields for the current
    RCP<MultiVectorT> coords() const { return m_coords; }

    //! \brief Get the map for a given residual
    //! \param distrib The parallel distribution type (OWNED/GHOST)
    //! \param i The residual index of interest
    RCP<const MapT> map(int distrib, int i) const {
      return m_maps[distrib][i];
    }

    //! \brief Get the graph for a residual pair
    //! \param distrib The parallel distribution type (OWNED/GHOST)
    //! \param i The first residual index
    //! \param j The second residual index
    RCP<const GraphT> graph(int distrib, int i, int j) {
      return m_graphs[distrib][i][j];
    }

    //! \brief Get the exporter for a given residual
    //! \param i The residual index of interest
    RCP<const ExportT> exporter(int i) const { return m_exporters[i]; }

    //! \brief Get the number of PDE residuals associated with this disc
    //! \details Only available after build_data has been called
    int num_residuals() const { return m_num_residuals; }

    //! \brief Get the number of equations associated with a residual
    //! \param residual The PDE residual index
    //! \details Only available after build_data has been called
    int num_eqs(int residual) const { return m_num_eqs[residual]; }

    //! \brief Get the local IDs associated with a PDE residual
    //! \param elem The mesh entity corresponding to an element
    //! \param i The residual index of interest
    Array2D<LO> get_element_lids(apf::MeshEntity* elem, int i);

    //! \brief Get the local ID associated with a PDE residual
    //! \param n The node object
    //! \param i The residual index of interest
    //! \param eq The equation component of the residual
    LO get_lid(apf::Node const& n, int i, int eq);

    //! \brief Get the local ID associated with a PDE residual
    //! \param ent The mesh entity
    //! \param i The residual index of interest
    //! \param n The node index wrt the mesh entity
    //! \param eq The equation index of the residual
    LO get_lid(apf::MeshEntity* ent, int i, int n, int eq);

    //! \brief Get the sets associated with this discretization
    apf::StkModels* sets() { return m_sets; }

    // !\brief Destroy Mesh specific data
    void destroy_data();

    //! \brief Create the primal fields at a step
    //! \param R The global/local residuals defining the problem
    //! \param step The current load/time step
    void create_primal(
        RCP<Residuals<double>> R,
        int step);

    //! \brief Destroy the primal fields
    //! \param keep_ic Keep the initial condition?
    void destroy_primal(bool keep_ic = true);

    //! \brief Create all adjoint fields
    //! \parma R The global/local residuals defining the problem
    //! \param num_steps The number of load/time steps
    void create_adjoint(
        RCP<Residuals<double>> R,
        int num_steps);

    //! \brief Destroy the adjoint fields
    void destroy_adjoint();

    //! \brief Get the primal fields stored on this discretization
    Array1D<Fields>& primal() { return m_primal; }

    //! \brief Get the adjoint fields stored on this discretization
    Array1D<Fields>& adjoint() { return m_adjoint; }

    //! \brief Get the primal fields at a step
    Fields& primal(int step) { return m_primal[step]; }

    //! \brief Get the adjoint fields at a step
    Fields& adjoint(int step) { return m_adjoint[step]; }

    //! \brief Add a solution increment at the current step
    //! \param x The solution fields (of global variables)
    //! \param dx The solution increment (of global variables)
    void add_to_soln(
        Array1D<apf::Field*>& x,
        Array1D<RCP<VectorT>> const& dx);

  protected:

    int m_num_dims = -1;
    int m_num_elems = -1;

    int m_num_elem_sets = -1;
    int m_num_side_sets = -1;
    int m_num_node_sets = -1;

    int m_elem_type = -1;
    int m_num_gv_nodes = -1;
    int m_num_lv_nodes = -1;

    apf::Mesh2* m_mesh = nullptr;
    apf::StkModels* m_sets = nullptr;

    apf::FieldShape* m_gv_shape = nullptr;
    apf::FieldShape* m_lv_shape = nullptr;

    apf::Numbering* m_owned_nmbr = nullptr;
    apf::Numbering* m_ghost_nmbr = nullptr;
    apf::GlobalNumbering* m_global_nmbr = nullptr;

    int m_num_residuals = -1;
    Array1D<int> m_num_eqs;

    RCP<const Comm> m_comm;
    RCP<const MapT> m_node_map;
    RCP<MultiVectorT> m_coords;

    Array1D<RCP<const MapT>> m_maps[NUM_DISTRIB];
    Array2D<RCP<GraphT>> m_graphs[NUM_DISTRIB];
    Array1D<RCP<const ExportT>> m_exporters;

    ElemSets m_elem_sets;
    SideSets m_side_sets;
    NodeSets m_node_sets;

    Array1D<Fields> m_primal;
    Array1D<Fields> m_adjoint;

    bool m_is_base = true;
    int m_disc_type = COARSE;

  protected:

    void initialize();

    void compute_node_map();
    void compute_coords();
    void compute_owned_maps();
    void compute_ghost_maps();
    void compute_exporters();
    void compute_graphs();
    void compute_elem_sets();
    void compute_side_sets();
    void compute_node_sets();

    std::vector<size_t> compute_nentries(int i, int j);
    void compute_ghost_graph(int i, int j);
    void compute_owned_graph(int i, int j);


};

}
