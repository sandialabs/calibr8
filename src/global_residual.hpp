#pragma once

//! \file global_residual.hpp
//! \brief The global residual interface

#include <apf.h>
#include "arrays.hpp"
#include "defines.hpp"
#include "fields.hpp"
#include "weight.hpp"

namespace calibr8 {

//! \cond
//  forward declarations
class Disc;
template <typename T> class LocalResidual;
//! \endcond

//! \brief The global residual interface
//! \tparam T The underlying scalar type used for evaluations
//! \details This object is intended to evaluate local element-wise
//! contributions based on the global residual(s)
template <typename T>
class GlobalResidual {

  public:

    //! \brief The global residual constructor
    GlobalResidual();

    //! \brief The global residual destructor
    virtual ~GlobalResidual();

    //! \brief The number of distinct PDE equations
    //! \details For example, mechanics would return 2 : a momentum
    //! and pressure residual
    int num_residuals() const { return m_num_residuals; }

    //! \brief The number of equations for a given PDE residual
    //! \param i The index into the residual of interest
    //! \details For example, mechanics momentum residual would return
    //! the number of spatial dimensions in the mesh
    int num_eqs(int i) const { return m_num_eqs[i]; }

    //! \brief The name of a given residual / global variable
    //! \param i The index into the residual of interest
    std::string const& resid_name(int i) const { return m_resid_names[i]; }

    //! \brief The number of equations for all PDE residuals
    Array1D<int> const& num_eqs() const { return m_num_eqs; }

    //! \brief The number of spatial dimensions
    int num_dims() const { return m_num_dims; }

    //! \brief The number of nodes in an element
    int num_nodes() { return m_num_nodes; }

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    //! \details This will initialize the local element-level quantities
    //! that this class is responsible for using/computing
    void before_elems(RCP<Disc> disc);

    //! \brief Set element data on element input
    //! \param mesh_elem The current mesh element to operate on
    void set_elem(apf::MeshElement* mesh_elem);

    //! \brief Gather the nodal global state variables
    //! \param x The global state variable fields at the current step
    //! \param x_prev The global state variable fields at the previous step
    //! \details This will gather the element's global variable values at nodes
    void gather(
        Array1D<apf::Field*> const& x,
        Array1D<apf::Field*> const& x_prev);

    //! \brief Zero the integration point residual
    void zero_residual();

    //! \brief Seed the global variables as derivative quantities
    void seed_wrt_x();

    //! \brief Unseed the local variables as derivative quantities
    //! \details This will set the value of m_x to m_x.val()
    void unseed_wrt_x();

    //! \brief Seed the global variables from the previous step as derivative quantities
    void seed_wrt_x_prev();

    //! \brief Unseed the local variables as derivative quantities
    //! \details This will set the value of m_x_prev to m_x_prev.val()
    void unseed_wrt_x_prev();

    //! \brief Interpolate the nodal values to an integration point
    //! \param iota The integration point in the reference element space
    //! \details This calls weight->evaluate
    //! This should be called after gather + seed have been called
    //! appropriately
    void interpolate(apf::Vector3 const& iota);

    //! \brief Interpolate nodal values to an integration point in error mode
    //! \param iota The integration point in the reference element space
    //! \details This calls weight->evaluate, which should have been constructed
    //! with the ErrorWeight class
    void interpolate_with_error(apf::Vector3 const& iota);

    //! \brief Evaluate the residual at an integration point
    //! \param local The local residual
    //! \param iota The integration point in the reference element space
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    //! \param ip_set The integration point set index
    virtual void evaluate(
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv,
        int ip_set) = 0;

    //! \brief Scatter the residual into the global residual vector
    //! \param disc The discretization object
    //! \param vector The element-level vector to scatter
    //! \param RHS The global right hand side vectors
    void scatter_rhs(
        RCP<Disc> disc,
        EVector const& rhs,
        RCP<VectorT>& RHS);

    //! \brief Assign a value into a global vector
    //! \param disc The discretization object
    //! \param vector The element-level vector to assign
    //! \param RHS The global right hand side vectors
    void assign_rhs(
        RCP<Disc> disc,
        EVector const& rhs,
        RCP<VectorT>& RHS);

    //! \brief Scatter the residual into the global Jacobian matrix
    //! \param disc The discretization object
    //! \param dtotal The total element-level derivative
    //! \param LHS The global LHS matrices
    //! \details Only performs an operation if this class has been templated
    //! on FADT.
    void scatter_lhs(
        RCP<Disc> disc,
        EMatrix const& dtotal,
        RCP<MatrixT>& LHS);

    //! \brief Reset element-specific data after processing an element
    void unset_elem();

    //! \brief Reset the residual data structure after looping over elements
    void after_elems();

    //! \brief The nodal values of the global variables
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    T const& x_nodal(int i, int node, int eq) const {
      return m_x_nodal[i][node][eq];
    }

    //! \brief The nodal values of the global variables at the previous step
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    T const& x_prev_nodal(int i, int node, int eq) const {
      return m_x_prev_nodal[i][node][eq];
    }

    //! \brief The nodal residual values based on some physics
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    T& R_nodal(int i, int node, int eq) {
      return m_R_nodal[i][node][eq];
    }

    //! \brief The weighting function at the current integration point
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    double weight(int i, int n, int eq) {
      return m_weight->val(i, n, eq);
    }

    //! \brief The stabilized weighting function at the current integration point
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    double stab_weight(int i, int n, int eq) {
      return m_stab_weight->val(i, n, eq);
    }

    //! \brief The weighting function at the current integration point
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    //! \parma dim The derivative axis
    double grad_weight(int i, int n, int eq, int dim) {
      return m_weight->grad(i, n, eq, dim);
    }

    //! \brief The stabilized weighting function at the current integration point
    //! \param i The global residual index
    //! \param node The node index
    //! \param eq The global variable equation component
    //! \parma dim The derivative axis
    double grad_stab_weight(int i, int n, int eq, int dim) {
      return m_stab_weight->grad(i, n, eq, dim);
    }

    //! \brief Get a scalar variable at the current integration point
    //! \param i The residual index of interest
    T scalar_x(int i) const;

    //! \brief Get a vector variable at the current integration point
    //! \param i The residual index of interest
    Vector<T> vector_x(int i) const;

    //! \brief Get a scalar gradient at the current integration point
    //! \param i The residual index of interest
    Vector<T> grad_scalar_x(int i) const;

    //! \brief Get a vector gradient at the current integration point
    //! \param i The residual index of interest
    Tensor<T> grad_vector_x(int i) const;

    //! \brief Get a previous scalar variable at the current integration point
    //! \param i The residual index of interest
    T scalar_x_prev(int i) const;

    //! \brief Get a previous vector variable at the current integration point
    //! \param i The residual index of interest
    Vector<T> vector_x_prev(int i) const;

    //! \brief Get a previous scalar gradient at the current integration point
    //! \param i The residual index of interest
    Vector<T> grad_scalar_x_prev(int i) const;

    //! \brief Get a previous vector gradient at the current integration point
    //! \param i The residual index of interest
    Tensor<T> grad_vector_x_prev(int i) const;

    //! \brief Gather the residual vector R
    EVector eigen_residual() const;

    //! \brief Gather the Jacobian matrix dR / d (seeded_vars)
    EMatrix eigen_jacobian() const;

    //! \brief Gather the nodal global DOF values
    //! \param z The global adjoint fields
    EVector gather_adjoint(Array1D<apf::Field*> const& z) const;

    //! \brief Gather the nodal difference between two DOF fields
    //! \param x_fine The fine solution field
    //! \param x The coarse solution field
    EVector gather_difference(
        Array1D<apf::Field*> const& x_fine,
        Array1D<apf::Field*> const& x) const;

    Array1D<int> ip_sets() const;

  private:

    int dx_idx(int i, int node, int eq) const;

  protected:

    //! \cond

    int m_num_residuals = -1;
    Array1D<int> m_num_eqs;
    Array1D<int> m_var_types;
    Array1D<std::string> m_resid_names;

    int m_num_dims = -1;
    int m_num_nodes = -1;
    int m_num_dofs = -1;

    Array1D<int> m_dx_offsets;

    apf::Mesh* m_mesh = nullptr;
    apf::FieldShape* m_shape = nullptr;
    apf::MeshElement* m_mesh_elem = nullptr;

    Array3D<T> m_x_nodal;
    Array3D<T> m_x_prev_nodal;
    Array3D<T> m_R_nodal;

    Array2D<T> m_x;
    Array2D<T> m_x_prev;
    Array3D<T> m_grad_x;
    Array3D<T> m_grad_x_prev;

    Array1D<int> m_ip_sets;

    Weight* m_weight = nullptr;
    Weight* m_stab_weight = nullptr;

    //! \endcond

};

//! \brief Create a global residual given a name
//! \tparam T The underlying scalar type used for evaluations
//! \param params The global residual parameters
//! \param ndims The number of spatial dimensions
template <typename T>
RCP<GlobalResidual<T>> create_global_residual(
    ParameterList const& params,
    int ndims);

}
