#pragma once

#include <apf.h>
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
class State;
template <typename T> class GlobalResidual;
template <typename T> class LocalResidual;
//! \endcond

//! \brief A class for QoI evaluations and their derivatives
//! \param T The underlying scalar type
template <typename T>
class QoI {

  public:

    //! \brief The QoI constructor
    QoI();

    //! \brief The QoI destructor
    virtual ~QoI();

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    //! \param step The current load/time step
    virtual void before_elems(RCP<Disc> disc, int step);

    //! \brief Set element data on element input
    //! \param mesh_elem The current mesh element to operate on
    void set_elem(apf::MeshElement* mesh_elem);

    //! \brief Evaluate the qoi at an integration point
    //! \param elem_set The index of the current element set
    //! \param elem_idx The index of the current element in the element set
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    virtual void evaluate(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv) = 0;

    //! \brief Scatter the integration point value into the total QoI
    void scatter(double& J);

    //! \brief Evaluate a preprocessing quantity at an integration point
    //! \param elem_set The index of the current element set
    //! \param elem_idx The index of the current element in the element set
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    virtual void preprocess(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

    //! \brief Finalize the QoI preprocessing computation
    //! \param step load step
    virtual void preprocess_finalize(int step);

    //! \brief Add preprocessing contributions to the QoI
    //! \param J QoI value
    virtual void postprocess(double& J);

    //! \brief Gather the derivative vector dJ / d(seeded_vars)
    EVector eigen_dvector(int nderivs) const;

    //! \brief Reset element-specific data after processing an element
    void unset_elem();

    //! \brief Do some post-processing of the linear systems
    virtual void modify_state(RCP<State> state);

    //! \brief Reset the class after processing all elements
    virtual void after_elems();

  protected:


    //! \cond

    //! \brief Initialize the mapping from elements to facets in a side set
    //! \param side_set The name of the side set
    //! \param disc Discretization object
    //! \param mapping The element to side set facet mapping array
    //! \returns Flag for whether the mapping array exists
    bool setup_side_set_mapping(std::string const& side_set,
        RCP<Disc> disc,
        Array2D<int>& mapping);

    //! \brief Initialize the mapping from elements to nodes in a side set
    //! \param side_set The name of the side set
    //! \param disc Discretization object
    //! \param mapping The element to nodes in the sideset mapping array
    //! \returns Flag for whether the mapping array exists
    bool setup_side_set_to_node_mapping(std::string const& side_set,
        RCP<Disc> disc,
        Array3D<int>& mapping);

    //! \brief Initialize the mapping from elements to nodes in a node set
    //! \param side_set The name of the node set
    //! \param disc Discretization object
    //! \param mapping The element to node set node mapping array
    //! \returns Flag for whether the mapping array exists
    bool setup_node_set_mapping(std::string const& node_set,
        RCP<Disc> disc,
        Array3D<int>& mapping);

    //! \brief Initialize the mapping from elements to nodes with a fixed coordinate value
    //! \param coord_idx Index of the coordinate
    //! \param coord_value Value for the coordinate
    //! \param disc Discretization object
    //! \param mapping The element to nodes mapping array
    //! \returns Flag for whether the mapping array exists
    bool setup_coord_based_node_mapping(
        int coord_idx,
        double coord_value,
        RCP<Disc> disc,
        Array3D<int>& mapping);

    //! \brief Zero out the value (and derivatives) of the QoI at a point
    void initialize_value_pt();

    int m_num_dims = -1;
    int m_step = -1;
    apf::Mesh* m_mesh = nullptr;
    apf::FieldShape* m_shape = nullptr;
    apf::MeshElement* m_mesh_elem = nullptr;

    T value_pt = T(0.);

    //! \endcod

};

//! \brief Create a specific quantity of interest
//! \tparam T The underlying scalar type
//! \param params The paramterlist used to build the QoI
template <typename T>
RCP<QoI<T>> create_qoi(ParameterList const& params);

}
