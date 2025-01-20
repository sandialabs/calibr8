#include "defines.hpp"
#include "NN.hpp"

#include "Eigen/Core"

using RFAD_SFADT = Sacado::Rad::ADvar<SFADT>;
using RFAD_DFADT = Sacado::Rad::ADvar<DFADT>;

namespace Eigen {

template <> struct NumTraits<RFAD_SFADT> : NumTraits<double>
{
  typedef RFAD_SFADT Real;
  typedef RFAD_SFADT NonInteger;
  typedef RFAD_SFADT Nested;
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

template <> struct NumTraits<RFAD_DFADT> : NumTraits<double>
{
  typedef RFAD_DFADT Real;
  typedef RFAD_DFADT NonInteger;
  typedef RFAD_DFADT Nested;
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

inline const RFAD_SFADT& conj(const RFAD_SFADT& x)  { return x; }
inline const RFAD_SFADT& real(const RFAD_SFADT& x)  { return x; }
inline RFAD_SFADT imag(const RFAD_SFADT&)    { return 0.; }
inline RFAD_SFADT abs(const RFAD_SFADT&  x)  { return fabs(x); }
inline RFAD_SFADT abs2(const RFAD_SFADT& x)  { return x*x; }

inline const RFAD_DFADT& conj(const RFAD_DFADT& x)  { return x; }
inline const RFAD_DFADT& real(const RFAD_DFADT& x)  { return x; }
inline RFAD_DFADT imag(const RFAD_DFADT&)    { return 0.; }
inline RFAD_DFADT abs(const RFAD_DFADT&  x)  { return fabs(x); }
inline RFAD_DFADT abs2(const RFAD_DFADT& x)  { return x*x; }

}

namespace ML {

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
    Vector evaluate(Vector const& x);
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
