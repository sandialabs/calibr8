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


namespace DEBUG {

template <class T>
class Vector
{

  private:

    std::vector<T> v;

  public:

    Vector() = default;

    Vector(int sz) {
      v.resize(sz);
      std::fill(v.begin(), v.end(), T(0));
    }

    Vector(Vector<T> const& other)
    {
      v = other.v;
    }

    Vector<T>& operator=(Vector<T> const& other)
    {
      v = other.v;
      return *this;
    }

    int size() const { return v.size(); }

    T& operator()(int i) { return v[i]; }
    T const& operator()(int i) const { return v[i]; }

    Vector<T> operator+(Vector<T> const& other) const
    {
      if (size() != other.size()) {
        throw std::runtime_error("Vector[+] invalid\n");
      }
      Vector<T> r(size());
      for (int i = 0; i < size(); ++i) {
        r.v[i] = v[i] + other.v[i];
      }
      return r;
    }

    Vector<T> operator-(Vector<T> const& other) const
    {
      if (size() != other.size()) {
        throw std::runtime_error("Vector[+] invalid\n");
      }
      Vector<T> r(size());
      for (int i = 0; i < size(); ++i) {
        r.v[i] = v[i] - other.v[i];
      }
      return r;
    }

    Vector<T> operator*(T const& scalar) const
    {
      Vector<T> r(size());
      for (int i = 0; i < size(); ++i) {
        r.v[i] = v[i] * scalar;
      }
      return r;
    }

    Vector<T> operator/(T const& scalar) const
    {
      Vector<T> r(size());
      for (int i = 0; i < size(); ++i) {
        r.v[i] = v[i] / scalar;
      }
      return r;
    }

    void operator*=(T const& scalar)
    {
      for (int i = 0; i < size(); ++i) {
        v[i] *= scalar;
      }
    }

    void operator/=(T const& scalar)
    {
      for (int i = 0; i < size(); ++i) {
        v[i] /= scalar;
      }
    }

    void setOnes()
    {
      for (int i = 0; i < size(); ++i) {
        v[i] = T(1);
      }
    }

};

template <class T>
class Matrix
{

  private:

    int ni = 0;
    int nj = 0;
    std::vector<T> v;

  public:

    Matrix() = default;

    Matrix(int ni_in, int nj_in)
    {
      ni = ni_in;
      nj = nj_in;
      v.resize(ni*nj);
      std::fill(v.begin(), v.end(), T(0));
    }

    Matrix(Matrix<T> const& other)
    {
      ni = other.ni;
      nj = other.nj;
      v = other.v;
    }

    Matrix<T>& operator=(Matrix<T> const& other)
    {
      ni = other.ni;
      nj = other.nj;
      v = other.v;
      return *this;
    }

    int rows() const { return ni; }
    int cols() const { return nj; }
    int size() const { return ni*nj; }

    T& operator()(int i, int j) { return v[i*nj + j]; }
    T const& operator()(int i, int j) const { return v[i*nj + j]; }

    Matrix<T> operator+(Matrix<T> const& other) const
    {
      if (ni != other.ni) throw std::runtime_error("Matrix[+] ni\n");
      if (nj != other.nj) throw std::runtime_error("Matrix[+] nj\n");
      Matrix<T> r(ni, nj);
      for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
          r(i,j) = (*this)(i,j) + other(i,j);
        }
      }
      return r;
    }

    Matrix<T> operator-(Matrix<T> const& other) const
    {
      if (ni != other.ni) throw std::runtime_error("Matrix[+] ni\n");
      if (nj != other.nj) throw std::runtime_error("Matrix[+] nj\n");
      Matrix<T> r(ni, nj);
      for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
          r(i,j) = (*this)(i,j) - other(i,j);
        }
      }
      return r;
    }

    Vector<T> operator*(Vector<T> const& x) const
    {
      if (nj != x.size()) {
        printf("nj: %d\n", nj);
        printf("x.size(): %d\n", x.size());
        throw std::runtime_error("Matrix[*] nj\n");
      }
      Vector<T> r(ni);
      for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
          r(i) += (*this)(i,j) * x(j);
        }
      }
      return r;
    }

};

}

/* \brief a feed forward input convex neural network */
template <class ScalarT>
class FICNN
{
  public:
    using Vector = DEBUG::Vector<ScalarT>;
    using Matrix = DEBUG::Matrix<ScalarT>;
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
