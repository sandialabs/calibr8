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

class FEMU_Objective : public ROL::Objective<double> {

  public:

    FEMU_Objective(RCP<ParameterList> params);
    ~FEMU_Objective();

    Array1D<double> opt_params() const;
    double value(ROL::Vector<double> const& p, double&);


#if 0
    Array1D<double> scale_params(Array1D<double> const& p);
    Array1D<double> unscale_params(Array1D<double> const& p);
    Array1D<double> scale_gradient(Array1D<double> const& g);
    Array1D<double> get_scaled_ig() { return scaled_ig; }
#endif

  protected:

    void set_params(ROL::Vector<double> const& p);

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Primal> m_primal;

    int m_num_opt_params;

    ROL::Ptr<Array1D<double> const> getVector(V const& vec);
    ROL::Ptr<Array1D<double>> getVector(V& vec);

};

}
