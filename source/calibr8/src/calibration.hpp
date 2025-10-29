#pragma once

//! \file calibration.hpp
//! \brief The interface for displacement (distance-based inclusion) and load mismatch QoI

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of a displacement and load mismatches over two surfaces
template <typename T>
class Calibration : public QoI<T> {

  public:

    //! \brief The calibration constructor
    //! \param params The QoI parameter list
    Calibration(ParameterList const& params);

    //! \brief The calibration destructor
    ~Calibration();

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
        apf::Vector3 const& iota_input,
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

  private:

    // 2D
    T compute_disp_mismatch(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota_input);

    // 3D
    T compute_surface_mismatch(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota_input);

    T compute_load(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

    double m_balance_factor = 1.;

    bool is_initd_load = false;
    Array3D<int> m_mapping_load; // m_mapping[es_idx][elem_idx]

    bool m_write_load = false;
    std::string m_load_out_file = "";
    bool m_read_load = false;
    std::string m_load_in_file = "";
    double m_total_load = 0.;
    double m_load_mismatch = 0.;
    Array1D<double> m_load_data;
    int m_coord_idx = -1;
    double m_coord_value = 0.;
    double m_coord_tol = 1e-12;
    int m_reaction_force_comp = -1;

    // only in 3D
    std::string m_side_set_disp = "";

    bool is_initd_disp = false;
    std::string m_distance_field_name = "distance";
    Array2D<int> m_mapping_disp; // m_mapping[es_idx][elem_idx]

    Array1D<double> m_weights {1., 1., 1.};
    bool m_has_weights = false;
    bool m_has_distance_threshold = false;
    double m_distance_threshold = 0.;

    bool m_write_obj_at_step = false;
    std::string m_obj_out_file = "";

    double m_area = 1.;
    double m_dt = 1.;
    bool is_initd_total_time = false;
    double m_total_time = 1.;
};

}
