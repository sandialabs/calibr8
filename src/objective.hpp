#pragma once

#include <ROL_Objective.hpp>
#include <ROL_StdVector.hpp>
#include "arrays.hpp"
#include "defines.hpp"

namespace calibr8 {

class Primal;
class State;

using V = ROL::Vector<double>;
using SV = ROL::StdVector<double>;

class Objective : public ROL::Objective<double> {

  public:

    Objective(RCP<ParameterList> params);
    ~Objective();

    Array2D<double> model_params() const;
    Array1D<double> active_params() const;
    Array1D<double> transform_params(Array1D<double> const& params,
        bool scale_to_canonical);

    double value(ROL::Vector<double> const& p, double&) = 0;

    Array2D<std::string> active_param_names() const {
      return m_active_param_names;
    }

    Array1D<std::string> elem_set_names() const {
      return m_elem_set_names;
    }

  protected:

    void setup_opt_params(ParameterList const& inverse_params);
    Array1D<double> transform_gradient(Array1D<double> const& gradient);

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Primal> m_primal;

    Array2D<int> m_active_indices;
    Array1D<std::string> m_elem_set_names;
    Array2D<std::string> m_active_param_names;
    Array1D<double> m_lower_bounds;
    Array1D<double> m_upper_bounds;
    int m_num_opt_params;

    ROL::Ptr<Array1D<double> const> getVector(V const& vec);
    ROL::Ptr<Array1D<double>> getVector(V& vec);

};

}
