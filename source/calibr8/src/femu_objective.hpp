#pragma once

#include "objective.hpp"

namespace calibr8 {

class FEMU_Objective : public Objective {

  public:

    FEMU_Objective(RCP<ParameterList> params);
    ~FEMU_Objective();

    double value(ROL::Vector<double> const& p, double&);

};

}
