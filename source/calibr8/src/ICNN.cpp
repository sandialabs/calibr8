#include "ICNN.hpp"

#include <stdexcept>

namespace ML {

template <class T>
T softplus(T const& x)
{
  return std::log(1. + std::exp(x));
}

static void check_topo(std::vector<int> const& topo)
{
  if (topo.size() < 3) {
    throw std::runtime_error("FICNN: not enough layers");
  }
}

template <class MatrixT, class VectorT>
int ficnn_count(
    std::vector<MatrixT> const& Wy,
    std::vector<MatrixT> const& Wz,
    std::vector<VectorT> const& b)
{
  int num = 0;
  for (size_t i =0 ; i < Wy.size(); ++i) {
    num += Wy[i].size();
    num += Wz[i].size();
    num += b[i].size();
  }
  return num;
}

template <class T>
FICNN<T>::FICNN(
    const char* act,
    std::vector<int> const& topology)
{
  std::string const act_type = act;
  if (act_type == "softplus") activation = softplus;
  else throw std::runtime_error("FICNN: unknown " + act_type);
  num_vectors = int(topology.size());
  Wy.resize(num_vectors-1);
  Wz.resize(num_vectors-1);
  b.resize(num_vectors-1);
  x.resize(num_vectors);
  int const end = num_vectors-1;
  int const ny = topology[0];
  for (int i = 0; i < end; ++i) {
    int const n_i = topology[i];
    int const n_ip1 = topology[i+1];
    b[i] = Vector(n_ip1);
    x[i] = Vector(n_i);
    Wy[i] = Matrix(n_ip1, n_i);
    if (i > 0) {
      Wz[i] = Matrix(ny, n_ip1);
    }
    for (int j = 0; j < Wy[i].rows(); ++j) {
      for (int k = 0; k < Wy[i].cols(); ++k) {
        double weight = ((double) rand() / (RAND_MAX));
        Wy[i](j,k) = std::abs(weight);
      }
    }
    for (int j = 0; j < Wz[i].rows(); ++j) {
      for (int k = 0; k < Wz[i].cols(); ++k) {
        double weight = ((double) rand() / (RAND_MAX));
        Wz[i](j,k) = std::abs(weight);
      }
    }
    b[i].setOnes();
  }
  x[end] = Vector(topology[end]);
  params = Vector(ficnn_count(Wz, Wy, b));
  num_params = params.size();
}

template <class T>
typename FICNN<T>::Vector FICNN<T>::f(Vector const& v)
{
  Vector result(v.size());
  for (int i = 0; i < int(v.size()); ++i) {
    result[i] = activation(v[i]);
  }
  return result;
}

template <class T>
typename FICNN<T>::Vector FICNN<T>::evaluate(Vector const& y)
{
  x[0] = y; /* just to keep input/output and all internal nodes in same data */
  x[1] = Wy[0]*y + b[0];
  for (int i = 1; i < num_vectors - 1; ++i) {
    Vector const tmp = Wz[i]*x[i] + Wy[i]*y + b[i];
    x[i+1] = f(tmp);
  }
  int const end = num_vectors - 1;
  return x[end];
}

template <class T>
typename FICNN<T>::Vector const& FICNN<T>::get_params()
{
  int idx = 0;
  for (size_t i = 0; i < Wy.size(); ++i) {
    for (int j = 0; j < Wy[i].rows(); ++j) {
      for (int k = 0; k < Wy[i].cols(); ++k) {
        params[idx++] = Wy[i](j,k);
      }
    }
    for (int j = 0; j < Wz[i].rows(); ++j) {
      for (int k = 0; k < Wz[i].cols(); ++k) {
        params[idx++] = Wz[i](j,k);
      }
    }
    for (int j = 0; j < b[i].size(); ++j) {
      params[idx++] = b[i][j];
    }
  }
  return params;
}

template <class T>
void FICNN<T>::set_params(Vector const& p)
{
  params = p;
  int idx = 0;
  for (size_t i = 0; i < Wy.size(); ++i) {
    for (int j = 0; j < Wy[i].rows(); ++j) {
      for (int k = 0; k < Wy[i].cols(); ++k) {
        Wy[i](j,k) = params[idx++];
      }
    }
    for (int j = 0; j < Wz[i].rows(); ++j) {
      for (int k = 0; k < Wz[i].cols(); ++k) {
        Wz[i](j,k) = params[idx++];
      }
    }
    for (int j = 0; j < b[i].size(); ++j) {
      b[i][j] = params[idx++];
    }
  }
}

template class FICNN<double>;
template class FICNN<SFADT>;
template class FICNN<DFADT>;
template class FICNN<RAD_SFADT>;
template class FICNN<RAD_DFADT>;

}
