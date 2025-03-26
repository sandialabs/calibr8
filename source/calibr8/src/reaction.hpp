#pragma once

//! \file reaction.hpp
//! \brief The interface for reaction/torque QoIs

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of a reaction/torque over a node set
template <typename T>
class Reaction : public QoI<T> {

  public:

    //! \brief The reaction/torque constructor
    //! \param params The QoI parameter list
    Reaction(ParameterList const& params);

    //! \brief The reaction/torque destructor
    ~Reaction();

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    void before_elems(RCP<Disc> disc, int step);

    //! \brief Evaluate the qoi at an integration point
    //! \param global The global residual object
    //! \param local The local residual object
    void evaluate(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);


    //! \brief Evaluate a preprocessing quantity at an integration point
    //! \param elem_set The index of the current element set
    //! \param elem_idx The index of the current element in the element set
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    T compute_load(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

    //! \brief Compute a component of the torque vector at a node according to m_reaction_force_comp
    //! \param global The global residual object
    //! \param r The position vector for the node
    //! \param node_id The node id of the node
    T compute_torque(
        RCP<GlobalResidual<T>> global,
        apf::Vector3 const& r,
        int const node_id);

  private:

    bool is_initd = false;
    Array3D<int> m_mapping; // m_mapping[es_idx][elem_idx][node_idx]
    int m_coord_idx = -1;
    double m_coord_value = 0.;
    int m_reaction_force_comp = -1;
    bool m_compute_torque = false;

};

}
