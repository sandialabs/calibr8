#pragma once

//! \file surface_mismatch.hpp
//! \brief The interface for displacement mismatch QoIs

#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of a displacement mismatch over a surface
// TODO: change this name to displacement mismatch
template <typename T>
class LoadMismatch : public QoI<T> {

  public:

    //! \brief The average displacement constructor
    //! \param params The QoI parameter list
    LoadMismatch(ParameterList const& params);

    //! \brief The average displacement destructor
    ~LoadMismatch();

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
        apf::Vector3 const&,
        double,
        double);

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
        apf::Vector3 const& iota_input);

  private:

    bool is_initd = false;
    std::string m_side_set = "";
    bool m_write_load = false;
    std::string m_load_out_file = "";
    bool m_read_load = false;
    std::string m_load_in_file = "";
    double m_total_load = 0.;
    double m_load_mismatch = 0.;
    Array1D<double> m_load_data;
    Array2D<int> m_mapping; // m_mapping[es_idx][elem_idx]

};

}
