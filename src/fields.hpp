#pragma once

//! \file fields.hpp
//! \brief Methods to help with field extraction / setting

#include <apf.h>
#include "arrays.hpp"
#include "defines.hpp"

namespace calibr8 {

//! \brief Type of variable
enum VariableType {
  SCALAR,
  VECTOR,
  SYM_TENSOR,
  TENSOR
};

//! \brief Get the number of residual equations given a variable type
//! \param type The variable type of interest
//! \param ndims The number of spatial dimensions of the discretization
int get_num_eqs(int type, int ndims);

//! \brief Get the element-wise nodal components of a field
//! \param f The field of interest
//! \param me The mesh element of interest
//! \param type The (calibr8) variable type
//! \param num_nodes The number of nodes in the element
Array2D<double> get_nodal_components(
    apf::Field* f,
    apf::MeshElement* me,
    int type,
    int num_nodes);

//! \brief Get the components of a field at a specific node
//! \param f The field of interest
//! \param e The entity that the node is specific to
//! \param node The node index
//! \param type The (calibr8) variable type
//! \details We currently use this for local residuals, where a
//! 'node' is an integration point
Array1D<double> get_node_components(
    apf::Field* f,
    apf::MeshEntity* e,
    int node,
    int type);

//! \brief Get the components of a variable in APF form
//! \param xi The compact (calibr8) 1D representation of our variable
//! \param ndims The number of spatial dimensions
//! \param type The variable type
Array1D<double> get_components(
    Array1D<FADT> const& xi,
    int ndims,
    int type);

//! \brief Get the components of a variable in APF form
//! \param chi The compact (calibr8) 1D representation of our variable
//! \param ndims The number of spatial dimensions
//! \param type The variable type
Array1D<double> get_components(
    Array1D<double> const& chi,
    int ndims,
    int type);

//! \brief Enrich a nodal field
//! \param z_H The coarse nodal field to enrich
//! \returns An enriched nodal field
apf::Field* enrich_nodal_field(
    apf::Field* z_H,
    apf::FieldShape* fine_shape);

//! \brief Enrich a quadrature point field
//! \param phi_H The coarse quadrature point field to enrich
//! \returns An enriched quadrature point field
apf::Field* enrich_qp_field(
    apf::Field* phi_H,
    apf::FieldShape* fine_shape);

//! \brief Subtract two nodal fields on the same mesh
//! \returns b - a
apf::Field* subtract(apf::Field* b, apf::Field* a);

}
