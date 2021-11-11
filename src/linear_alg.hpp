#include "disc.hpp"

namespace calibr8 {

//! \brief Storage for linear algebra objects
class LinearAlg {

  private:

    RCP<Disc> m_disc;

  public:

    //! \brief Global matrices
    Array2D<RCP<MatrixT>> A[NUM_DISTRIB];

    //! \brief Global solution vectors
    Array1D<RCP<VectorT>> x[NUM_DISTRIB];

    //! \brief Global right hand side vectors
    Array1D<RCP<VectorT>> b[NUM_DISTRIB];

  public:

    //! \brief Build the linear algebra data for a given disc
    //! \param disc The input discretization object
    void build_data(RCP<Disc> disc);

    //! \brief Destroy the linear algebra data
    void destroy_data();

    //! \brief Resume filling the matrix A
    void resume_fill_A();

    //! \brief Complete filling the matrix A
    void complete_fill_A();

    //! \brief Perform an MPI reduction of the matrices A
    //! \details Used with the directive ADD
    void gather_A();

    //! \brief Perform an MPI reduction of the vectors b
    //! \details Used with the directive ADD
    void gather_x();

    //! \brief Perform an MPI reduction of the vectors b
    //! \details Used with the directive ADD
    void gather_b();

    //! \brief Zero the solution vector
    void zero_all();

    //! \brief Scale the b vectors by a value
    //! \param val The value to scale the vectors by
    //! \details This only scales the OWNED b vectors
    void scale_b(double val);

    //! \brief Get the total norm of the blocked vector b
    double norm_b();

};

}
