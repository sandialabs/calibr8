#pragma once

#include "adjoint.hpp"
#include "objective.hpp"

namespace calibr8 {

class Adjoint_Objective: public Objective {

  public:

    Adjoint_Objective(RCP<ParameterList> params);
    ~Adjoint_Objective();

    double value(ROL::Vector<double> const& p, double&);

    void gradient(ROL::Vector<double>& g,
        ROL::Vector<double> const& p,
        double&);

  private:
    RCP<Adjoint> m_adjoint;

};

}
