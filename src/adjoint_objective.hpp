#pragma once

#include "femu_objective.hpp"

namespace calibr8 {

class Adjoint;

class Adjoint_Objective: public FEMU_Objective {

  public:

    Adjoint_Objective(RCP<ParameterList> params);
    ~Adjoint_Objective();

    void gradient(ROL::Vector<double>& g,
        ROL::Vector<double> const& p,
        double&);

  private:
    RCP<Adjoint> m_adjoint;

};

}
