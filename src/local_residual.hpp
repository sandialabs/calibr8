#pragma once

//! \file local_residual.hpp
//! \brief The local residual interface

#include <string>
#include <apf.h>
#include "arrays.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "fields.hpp"

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
template <typename T> class GlobalResidual;
class State;
//! \endcond

//! \brief The local residual interface
//! \tparam T The underlying scalar type used for evaluations
//! \brief This object is intended to evaluate local element-wise
//! contributions based on the local residuals (defined by the
//! constitutive model)
//! \details In this context, nodal values are actually interior to
//! the element.
template <typename T>
class LocalResidual {

  public:

    //! \brief The local residual constructor
    LocalResidual();

    //! \brief The local residual destructor
    virtual ~LocalResidual();

    //! \brief The number of local residual equations
    int num_residuals() const { return m_num_residuals; }

    //! \brief The variable type associated with a single local residual
    //! \param residual The residual index of interest
    int var_type(int residual) const { return m_var_types[residual]; }

    //! \brief The number of components associated with a single local residual
    //! \param residual The residual index of interest
    int num_eqs(int residual) const { return m_num_eqs[residual]; }

    //! \brief The name of a given residual / local variable
    //! \param i The index into the residual of interest
    std::string const& resid_name(int i) const { return m_resid_names[i]; }

    //! \brief Initialize the local state variables to correct values
    //! \details This will call init_variables at the integration point level
    //! \param state The application state object
    void init_variables(RCP<State> state);

    //! \brief Initialize material parameters
    virtual void init_params() = 0;

    //! \brief Initialize local state at an integration point
    virtual void init_variables_impl() = 0;

    //! \brief Perform intializations before the loop over elements
    //! \param disc The discretization object
    void before_elems(int const es, RCP<Disc> disc);

    //! \brief Process element data on element input
    //! \param mesh_elem The current mesh element to operate on
    void set_elem(apf::MeshElement* mesh_elem);

    //! \brief Interpolate local variables to the current integration point
    //! \param pt The integration point index
    //! \param xi The local state variable fields at the current step
    //! \param xi_prev The local state variable fields at the previous step
    //! \details Local fields are not 'gathered' in the same way that global
    //! fields are because they are totally uncoupled from one another.
    void gather(
        int pt,
        Array1D<apf::Field*> const& xi,
        Array1D<apf::Field*> const& xi_prev);

    //! \brief Seed the local variables as derivative quantities
    void seed_wrt_xi();

    //! \brief Unseed the local variables as derivative quantities
    //! \details This will set the value of m_xi to m_xi.val()
    void unseed_wrt_xi();

    //! \brief Seed the local variables from the previous step as derivative quantities
    void seed_wrt_xi_prev();

    //! \brief Unseed the local variables from the previous step as derivative quantities
    //! \details This will set the value of m_xi to m_xi.val()
    void unseed_wrt_xi_prev();

    //! \brief Seed the local variable derivatives wrt the global variables
    //! \param dxi_dx The matrix containing the derivative data
    void seed_wrt_x(EMatrix const& dxi_dx);

    //! \brief Seed the parameters as derivative quantities
    void seed_wrt_params();

    //! \brief Unseed the parameters as derivative quantities
    //! \details This will set the value of m_params to m_params.val()
    void unseed_wrt_params();

    //! \brief Solve the nonlinear model at the current integration point
    //! \param global The global residual equations
    //! \details This will fill in the current state m_xi and residuals m_R
    virtual int solve_nonlinear(RCP<GlobalResidual<T>> global) = 0;

    //! \brief Evaluate the local residual equations at the current integration point
    //! \param global The global residual equations
    //! \param use_path Use a flag to force a specific branch
    //! \param path The flag used to specify the branch path
    //! \details Returns an int defining which branch path the evaluation took
    virtual int evaluate(
        RCP<GlobalResidual<T>> global,
        bool use_path = false,
        int path = 0) = 0;

    //! \brief A flag to determine if these equations correspond
    //! to finite deformation
    virtual bool is_finite_deformation() = 0;

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    virtual Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global) = 0;

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    //! \param p The pressure
    virtual Tensor<T> cauchy(RCP<GlobalResidual<T>> global, T p) = 0;

    //! \brief Save the solved local variables to the current integration point
    //! \param pt The integration point index
    //! \param xi The local state variable fields at the current step
    void scatter(int pt, Array1D<apf::Field*>& xi);

    //! \brief Save the solved adjoint local variables to the current point
    //! \param pt The integration point index
    //! \param phi_pt The solved vector of all local adjoint variables
    //! \param phi The local adjoint state variable fields at the current step
    void scatter_adjoint(int pt, EVector const& phi_pt, Array1D<apf::Field*>& phi);

    //! \brief Gather the local adjoint to the current integration point
    //! \param pt The integration point index
    //! \param phi The local adjoint state variable fields at the current step
    //! \returns phi_pt The solved vector of all local adjoint variables
    EVector gather_adjoint(int pt, Array1D<apf::Field*> const& phi) const;

    //! \brief Gather the difference between two local states
    //! \param pt The integration point index
    //! \param xi_fine The fine local state variable fields at the current step
    //! \param xi The local state variable fields at the current step
    EVector gather_difference(
        int pt,
        Array1D<apf::Field*> const& xi_fine,
        Array1D<apf::Field*> const& xi) const;

    //! \brief Reset the element-specific data after processing an element
    void unset_elem();

    //! \brief Reset the residual data structure after looping over elements
    void after_elems();

  private:

    int dxi_idx(int i, int eq) const;

  public:

    //! \brief Get the first local variable value at the integration point
    T first_value() const;

    //! \brief Get a scalar variable at the current integration point
    //! \param i The residual index of interest
    T scalar_xi(int i) const;

    //! \brief Get a vector variable at the current integration point
    //! \param i The residual index of interest
    Vector<T> vector_xi(int i) const;

    //! \brief Get a symmetric tensor variable at the current integration point
    //! \param i The residual index of interest
    Tensor<T> sym_tensor_xi(int i) const;

    //! \brief Get a full tensor variable at the current integration point
    //! \param i The residual index of interest
    Tensor<T> tensor_xi(int i) const;

    //! \brief Get a previous scalar variable at the current integration point
    //! \param i The residual index of interest
    T scalar_xi_prev(int i) const;

    //! \brief Get a previous vector variable at the current integration point
    //! \param i The residual index of interest
    Vector<T> vector_xi_prev(int i) const;

    //! \brief Get a previous symmetric tensor variable at the current integration point
    //! \param i The residual index of interest
    Tensor<T> sym_tensor_xi_prev(int i) const;

    //! \brief Get a previous full tensor variable at the current integration point
    //! \param i The residual index of interest
    Tensor<T> tensor_xi_prev(int i) const;

  protected:

    //! \brief Set a scalar local variable at the current integration point
    //! \param i The residual index of interest
    //! \param val The value to set
    //! \details This only sets the value, and does not update derivatives
    void set_scalar_xi(int i, T const& val);

    //! \brief Set a vector local variable at the current integration point
    //! \param i The residual index of interest
    //! \param val The value to set
    //! \details This only sets the value, and does not update derivatives
    void set_vector_xi(int i, Vector<T> const& val);

    //! \brief Set a symmetric tensor local variable at the current integration point
    //! \param i The residual index of interest
    //! \param val The value to set
    //! \details This only sets the value, and does not update derivatives
    void set_sym_tensor_xi(int i, Tensor<T> const& val);

    //! \brief Set a full tensor local variable at the current integration point
    //! \param i The residual index of interest
    //! \param val The value to set
    //! \details This only sets the value, and does not update derivatives
    void set_tensor_xi(int i, Tensor<T> const& val);

  protected:

    //! \brief Add a scalar local variable at the current integration point
    //! \param i The residual index of interest
    //! \param dxi The increment to add
    //! \details This only adds the value, and does not update derivatives
    void add_to_scalar_xi(int i, EVector const& dxi);

    //! \brief Add a vector local variable at the current integration point
    //! \param i The residual index of interest
    //! \param dxi The increment to add
    //! \details This only adds the value, and does not update derivatives
    void add_to_vector_xi(int i, EVector const& dxi);

    //! \brief Add a symmetric tensor local variable at the current integration point
    //! \param i The residual index of interest
    //! \param dxi The increment to add
    //! \details This only adds the value, and does not update derivatives
    void add_to_sym_tensor_xi(int i, EVector const& dxi);

    //! \brief Add a symmetric tensor local variable at the current integration point
    //! \param i The residual index of interest
    //! \param dxi The increment to add
    //! \details This only adds the value, and does not update derivatives
    void add_to_tensor_xi(int i, EVector const& dxi);

  protected:

    //! \brief Fill in a scalar residual at the current integration point
    //! \param i The residual index of interest
    //! \param val The residual value
    void set_scalar_R(int i, T const& val);

    //! \brief Fill in a vector residual at the current integration point
    //! \param i The residual index of interest
    //! \param val The residual value
    void set_vector_R(int i, Vector<T> const& val);

    //! \brief Fill in a symmetric tensor residual at the current integration point
    //! \param i The residual index of interest
    //! \param val The residual value
    void set_sym_tensor_R(int i, Tensor<T> const& val);

    //! \brief Fill in a full tensor residual at the current integration point
    //! \param i The residual index of interest
    //! \param val The residual value
    void set_tensor_R(int i, Tensor<T> const& val);

  public:

    //! \brief Compute the 2-norm of the local residuals
    double norm_residual() const;

    //! \brief Gather the Jacobian matrix dC / d(seeded_vars)
    EMatrix eigen_jacobian() const;

    //! \brief Gather the residual vector as an Eigen vector
    EVector eigen_residual() const;

  public:

    //! \brief Get the material model parameters
    Array1D<double> params() const {
      Array1D<double> r(m_params.size());
      for (size_t i = 0; i < m_params.size(); ++i) {
        r[i] = val(m_params[i]);
      }
      return r;
    }

    //! \brief Set the material model parameters
    void set_params(Array1D<double> const& params,
        Array1D<size_t> const& active_indices) {
      for (size_t i = 0; i < active_indices.size(); ++i) {
        m_params[active_indices[i]] = params[i];
      }
    }

    //! \brief Get the material model parameters by index
    //! \param p The parameter index
    T const& params(int const p) { return m_params[p]; }

    //! \brief Get the parameter names for the local residual model
    Array1D<std::string> const& param_names() { return m_param_names; }

  protected:

    //! \cond

    int m_num_residuals = -1;
    std::vector<int> m_num_eqs;
    std::vector<int> m_var_types;
    std::vector<std::string> m_resid_names;

    ParameterList m_params_list;

    Array1D<T> m_params;
    Array2D<double> m_param_values;
    Array1D<std::string> m_param_names;

    Array1D<std::string> m_elem_set_names;

    int m_num_dims = -1;
    int m_num_nodes = -1;
    int m_num_dofs = -1;

    std::vector<int> m_dxi_offsets;

    apf::FieldShape* m_shape = nullptr;
    apf::MeshElement* m_mesh_elem = nullptr;

    Array2D<T> m_xi;
    Array2D<T> m_xi_prev;
    Array2D<T> m_R;

    //! \endcond

};

//! \brief Create a local residual given a name
//! \tparam T The underlying scalar type used for evaluations
//! \param params The local residual parameters
//! \param ndims The number of spatial dimensions
template <typename T>
RCP<LocalResidual<T>> create_local_residual(
    ParameterList const& params,
    int ndims);

}
