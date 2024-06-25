#pragma once

//! \file synthetic.hpp
//! \brief Methods to write synthetic DIC data

#include "disc.hpp"

namespace calibr8 {

//! \brief Write synthetic DIC data
//! \param problem_name The name of the FEM problem (from the YAML input)
//! \param disc The discretization
//! \param num_steps The total number of forward steps in the simulation
//! \details This assumes the disc.primal(step)[0] is the displacement field
void write_synthetic(
    std::string const& problem_name,
    RCP<Disc> disc,
    int num_steps);

}
