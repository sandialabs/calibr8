#include "defines.hpp"

#include "Eigen/Core"

#include <Sacado_Fad_DFad.hpp>
#include <Sacado_Fad_SLFad.hpp>

#include <array>
#include <vector>

using SFADT = Sacado::Fad::SLFad<double, calibr8::nmax_derivs>;
using DFADT = Sacado::Fad::DFad<double>;

namespace Eigen {

template<> struct NumTraits<DFADT> : NumTraits<double>
{
  typedef DFADT Real;
  typedef DFADT NonInteger;
  typedef DFADT Nested;
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

template <> struct NumTraits<SFADT> : NumTraits<double>
{
  typedef SFADT Real;
  typedef SFADT NonInteger;
  typedef SFADT Nested;
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

inline const DFADT& conj(const DFADT& x)  { return x; }
inline const DFADT& real(const DFADT& x)  { return x; }
inline DFADT imag(const DFADT&)    { return 0.; }
inline DFADT abs(const DFADT&  x)  { return fabs(x); }
inline DFADT abs2(const DFADT& x)  { return x*x; }
inline const SFADT& conj(const SFADT& x)  { return x; }
inline const SFADT& real(const SFADT& x)  { return x; }
inline SFADT imag(const SFADT&)    { return 0.; }
inline SFADT abs(const SFADT&  x)  { return fabs(x); }
inline SFADT abs2(const SFADT& x)  { return x*x; }

}

namespace ML {

using dVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using sfadVector = Eigen::Matrix<SFADT, Eigen::Dynamic, 1>;
using sfadMatrix = Eigen::Matrix<SFADT, Eigen::Dynamic, Eigen::Dynamic>;
using dfadVector = Eigen::Matrix<DFADT, Eigen::Dynamic, 1>;
using dfadMatrix = Eigen::Matrix<DFADT, Eigen::Dynamic, Eigen::Dynamic>;

template <class ScalarT>
class FFNN
{
  public:
    using Vector = Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>;
  public:
    FFNN(const char* activation, std::vector<int> const& topology);
    Vector const& get_params();
    void set_params(Vector const& p);
    Vector evaluate(Vector const& x);
  private:
    ScalarT (*activation)(ScalarT const&);
    Vector f(Vector const& v);
  private:
    int num_vectors;
    std::vector<Matrix> W;
    std::vector<Vector> b;
    std::vector<Vector> x;
  private:
    int num_params;
    Vector params;
};

}
