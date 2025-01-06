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

}

namespace Sacado {

inline const RFAD_SFADT& conj(const RFAD_SFADT& x)  { return x; }
inline const RFAD_SFADT& real(const RFAD_SFADT& x)  { return x; }
inline RFAD_SFADT imag(const RFAD_SFADT&)    { return 0.; }
inline RFAD_SFADT abs(const RFAD_SFADT&  x)  { return fabs(x); }
inline RFAD_SFADT abs2(const RFAD_SFADT& x)  { return x*x; }

}

namespace ML {

class Dummy
{
  Dummy();
};

}
