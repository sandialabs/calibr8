#pragma once

#include <ROL_Objective.hpp>
#include <ROL_StdVector.hpp>
#include "arrays.hpp"
#include "defines.hpp"
#include "primal.hpp"
#include "state.hpp"

namespace calibr8 {

using V = ROL::Vector<double>;
using SV = ROL::StdVector<double>;

class Objective : public ROL::Objective<double> {

  public:

    Objective(RCP<ParameterList> params);
    ~Objective();

    double value(ROL::Vector<double> const& p, double&) = 0;

    Array1D<double> transform_params(Array1D<double> const& params,
        bool scale_to_canonical) const;

    Array1D<double> active_params() const;

    Array2D<std::string> active_param_names() const;

    Array1D<std::string> elem_set_names() const;

  protected:

    void setup_opt_params(ParameterList const& inverse_params);
    Array1D<double> transform_gradient(Array1D<double> const& gradient) const;

    ROL::Ptr<Array1D<double> const> getVector(V const& vec);
    ROL::Ptr<Array1D<double>> getVector(V& vec);

    int m_num_problems = 0;
    RCP<ParameterList> m_params;
    Array1D<RCP<Primal>> m_primal;
    Array1D<RCP<State>> m_state;

    bool param_diff(std::vector<double> const&);

    Array1D<double> m_lower_bounds;
    Array1D<double> m_upper_bounds;
    int m_num_opt_params;

    double m_J_old = 0.;
    std::vector<double> m_p_old;
    double const m_difftol = 1.0e-15;

    int const m_model_form = 0;

};

}
