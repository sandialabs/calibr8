#pragma once

//! \file state.hpp
//! \brief A data container for the code state

#include "defines.hpp"
#include "arrays.hpp"
#include "disc.hpp"
#include "linear_alg.hpp"

//! \cond
// forward declarations
namespace apf {
  class Field;
}
//! \endcond

namespace calibr8 {

//! \cond
// forward declarations
template <typename T> class GlobalResidual;
template <typename T> class LocalResidual;
template <typename T> class QoI;
//! \endcond

//! \brief A container to group local and global residuals
//! \tparam T The underlying scalar type used for evaluations
template <typename T>
class Residuals {

  public:

    //! \brief The global residual objects
    RCP<GlobalResidual<T>> global;

    //! \brief The local residual objects
    RCP<LocalResidual<T>> local;

};

//! \brief A data container for the the code state
class State {

  public:

    //! \brief The discretization data structure
    RCP<Disc> disc;

    //! \brief The residual data structures
    RCP<Residuals<double>> residuals;

    //! \brief The residual data structures with derivative information
    RCP<Residuals<FADT>> d_residuals;

    //! \brief The quantity of interest data structure
    RCP<QoI<double>> qoi;

    //! \brief The derivative of the QoI data structure
    RCP<QoI<FADT>> d_qoi;

    //! \brief The linear algebra data structure
    RCP<LinearAlg> la;

  public:

    //! \brief State constructor
    //! \param params The top-level parameterlist
    State(ParameterList const& params);

};

}
