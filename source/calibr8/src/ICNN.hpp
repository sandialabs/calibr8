#include "defines.hpp"
#include "NN.hpp"

#include "Eigen/Core"

using RAD_SFADT = Sacado::Rad::ADvar<SFADT>;
using RAD_DFADT = Sacado::Rad::ADvar<DFADT>;

namespace Eigen {

template <> struct NumTraits<RAD_SFADT> : NumTraits<double>
{
  typedef RAD_SFADT Real;
  typedef RAD_SFADT NonInteger;
  typedef RAD_SFADT Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template <> struct NumTraits<RAD_DFADT> : NumTraits<double>
{
  typedef RAD_DFADT Real;
  typedef RAD_DFADT NonInteger;
  typedef RAD_DFADT Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

}

namespace Sacado {

inline const RAD_SFADT& conj(const RAD_SFADT& x)  { return x; }
inline const RAD_SFADT& real(const RAD_SFADT& x)  { return x; }
inline RAD_SFADT imag(const RAD_SFADT&)    { return 0.; }
inline RAD_SFADT abs(const RAD_SFADT&  x)  { return fabs(x); }
inline RAD_SFADT abs2(const RAD_SFADT& x)  { return x*x; }

inline const RAD_DFADT& conj(const RAD_DFADT& x)  { return x; }
inline const RAD_DFADT& real(const RAD_DFADT& x)  { return x; }
inline RAD_DFADT imag(const RAD_DFADT&)    { return 0.; }
inline RAD_DFADT abs(const RAD_DFADT&  x)  { return fabs(x); }
inline RAD_DFADT abs2(const RAD_DFADT& x)  { return x*x; }

}

namespace ML {


/* \brief a feed forward input convex neural network */
template <class ScalarT>
class FICNN
{
  public:
    using Vector = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>;
  public:
    FICNN(
        const char* activation,
        std::vector<int> const& topology);
    Vector const& get_params();
    void set_params(Vector const& p);
    Vector evaluate(Vector const& y);
  private:
    ScalarT (*activation)(ScalarT const&);
    Vector f(Vector const& v);
  private:
    int num_vectors;
    std::vector<Matrix> Wy; // standard operators
    std::vector<Matrix> Wz; // pass-through operators
    std::vector<Vector> b;  // biases
    std::vector<Vector> x;
  private:
    int num_params;
    Vector params;
};

}
