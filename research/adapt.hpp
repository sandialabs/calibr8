#pragma once

#include <apf.h>
#include "defines.hpp"

namespace calibr8 {

class Physics;

class Adapt {
  public:
    virtual void adapt(
        ParameterList const& params,
        RCP<Physics> physics,
        apf::Field* error) = 0;
};

RCP<Adapt> create_adapt(ParameterList const& params);

}
