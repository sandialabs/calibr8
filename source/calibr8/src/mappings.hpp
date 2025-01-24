#pragma once

//! \file mappings.hpp
//! \brief Helper methods for mapping side and node sets to other entities

#include "arrays.hpp"
#include "disc.hpp"

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
//! \endcond

//! \brief Initialize the mapping from elements to downward entitity (1 level) in a side set
//! \param side_set The name of the side set
//! \param disc Discretization object
//! \param mapping The element to side set downward entity array
//! \returns Flag for whether the mapping array exists
bool setup_side_set_mapping(
    std::string const& side_set,
    RCP<Disc> disc,
    Array2D<int>& mapping);

//! \brief Initialize the mapping from elements to downward entitity (1 level) in an array of side sets
//! \param side_set Array of side set names
//! \param disc Discretization object
//! \param mapping The element to side set downward entity array
//! \returns Flag for whether the mapping array exists
bool setup_side_sets_mapping(
    Array1D<std::string> const& side_sets,
    RCP<Disc> disc,
    Array2D<int>& mapping);

//! \brief Initialize the mapping from elements to nodes in a side set
//! \param side_set The name of the side set
//! \param disc Discretization object
//! \param mapping The element to nodes in the sideset mapping array
//! \returns Flag for whether the mapping array exists
bool setup_side_set_to_node_mapping(
    std::string const& side_set,
    RCP<Disc> disc,
    Array3D<int>& mapping);

//! \brief Initialize the mapping from elements to nodes in a node set
//! \param side_set The name of the node set
//! \param disc Discretization object
//! \param mapping The element to node set node mapping array
//! \returns Flag for whether the mapping array exists
bool setup_node_set_mapping(std::string const& node_set,
    RCP<Disc> disc,
    Array3D<int>& mapping);

//! \brief Initialize the mapping from elements to nodes with a fixed coordinate value
//! \param coord_idx Index of the coordinate
//! \param coord_value Value for the coordinate
//! \param disc Discretization object
//! \param mapping The element to nodes mapping array
//! \returns Flag for whether the mapping array exists
bool setup_coord_based_node_mapping(
    int coord_idx,
    double coord_value,
    RCP<Disc> disc,
    Array3D<int>& mapping);

}
