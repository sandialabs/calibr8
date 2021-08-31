#pragma once

//! \file mesh_size.hpp
//! \brief Mesh size field specification methods for adaptivity

#include <apf.h>

namespace calibr8 {

//! \brief Get an isotropic size field for a target number of elems
//! \param error The error distribution over elements
//! \param target The target number of elements in the adapted mesh
apf::Field* get_iso_target_size(apf::Field* error, int target);

}
