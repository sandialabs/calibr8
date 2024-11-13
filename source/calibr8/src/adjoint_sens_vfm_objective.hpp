#pragma once

#include "objective.hpp"
#include "virtual_power.hpp"

namespace calibr8 {

class Adjoint_VFM_Objective : public Objective {

  public:

    Adjoint_VFM_Objective(RCP<ParameterList> params);
    ~Adjoint_VFM_Objective();

    double value(ROL::Vector<double> const& p, double&);

    void gradient(ROL::Vector<double>& g,
        ROL::Vector<double> const& p,
        double&);

  private:
    RCP<VirtualPower> m_virtual_power;
    std::string m_load_in_file;
    Array1D<double> m_load_data;

};

}
