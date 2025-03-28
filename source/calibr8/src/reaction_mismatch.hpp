#pragma once

//! \file reaction_mismatch.hpp
//! \brief The interface for reaction mismatch QoIs

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of a reaction mismatch over a node set
// TODO: change this name to displacement mismatch
template <typename T>
class ReactionMismatch : public QoI<T> {

  public:

    //! \brief The average displacement constructor
    //! \param params The QoI parameter list
    ReactionMismatch(ParameterList const& params);

    //! \brief The average displacement destructor
    ~ReactionMismatch();

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
    void preprocess(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

    //! \brief Finalize the QoI preprocessing computation
    //! \param step load step
    void preprocess_finalize(int step);

    //! \brief Add preprocessing contributions to the QoI
    //! \param J QoI value
    void postprocess(double& J);

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
    bool m_write_load = false;
    std::string m_load_out_file = "";
    bool m_read_load = false;
    std::string m_load_in_file = "";
    double m_total_load = 0.;
    double m_load_mismatch = 0.;
    Array1D<double> m_load_data;
    Array3D<int> m_mapping; // m_mapping[es_idx][elem_idx][node_idx]
    int m_coord_idx = -1;
    double m_coord_value = 0.;
    int m_reaction_force_comp = -1;
    bool m_compute_torque = false;

};

}
