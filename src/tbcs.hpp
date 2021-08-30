#pragma once

//! \file tbcs.hpp
//! \brief Methods for traction boundary condtions

#include "arrays.hpp"
#include "defines.hpp"

//! \cond
// forward declarations
namespace apf {
class Field;
}
//! \endcond

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
//! \endcond

//! \brief Apply traction BCs to the primal system
//! \param tbcs The traction BC parameter list
//! \param disc The discretization object
//! \param R The OWNED residual vectors
//! \param t The current time
//! \details This is specific to mechanics quantities,
//! specifically displacements
void apply_primal_tbcs(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<RCP<VectorT>>& R,
    double t);

}
