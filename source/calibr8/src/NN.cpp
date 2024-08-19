#include "NN.hpp"

namespace ML {

template <class T>
T relu(T const& x)
{
  return (x > 0.) ? x : 0.;
}

template <class T>
T sigmoid(T const& x)
{
  return 1. / (1. + std::exp(-x));
}

template <class T>
T my_tanh(T const& x)
{
  return std::tanh(x);
}

static void ffnn_check_topo(std::vector<int> const& topo)
{
  if (topo.size() < 3) {
    throw std::runtime_error("FFNN: not enough layers");
  }
}

template <class MatrixT, class VectorT>
int ffnn_count(
    std::vector<MatrixT> const& W,
    std::vector<VectorT> const& b)
{
  int num = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    num += W[i].size();
    num += b[i].size();
  }
  return num;
}

template <class MatrixT, class VectorT>
void ffnn_seed(int, std::vector<MatrixT> const&, std::vector<VectorT> const&);
void ffnn_seed(int, std::vector<dMatrix> const&, std::vector<dVector> const&) {}
void ffnn_seed(int, std::vector<sfadMatrix> const&, std::vector<sfadVector> const&) {}

void ffnn_seed(
    int num_params,
    std::vector<dfadMatrix>& W,
    std::vector<dfadVector>& b)
{
  int idx = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    for (int j = 0; j < W[i].rows(); ++j) {
      for (int k = 0; k < W[i].cols(); ++k) {
        W[i](j,k).diff(idx++, num_params);
      }
    }
    for (int j = 0; j < b[i].size(); ++j) {
      b[i][j].diff(idx++, num_params);
    }
  }
}

template <class T>
FFNN<T>::FFNN(const char* act, std::vector<int> const& topo, bool positive_weights)
{
  ffnn_check_topo(topo);
  srand(10);
  std::string const type = act;
  if (type == "relu") activation = relu;
  else if (type == "sigmoid") activation = sigmoid;
  else if (type == "tanh") activation = my_tanh;
  else throw std::runtime_error("FFNN: unknown " + type);
  num_vectors = int(topo.size());
  W.resize(num_vectors - 1);
  b.resize(num_vectors - 1);
  x.resize(num_vectors);
  int const end = num_vectors - 1;
  for (int i = 0; i < end; ++i) {
    int const n_ip1 = topo[i+1];
    int const n_i = topo[i];
    W[i] = Matrix(n_ip1, n_i);
    b[i] = Vector(n_ip1);
    x[i] = Vector(n_i);
    for (int j = 0; j < W[i].rows(); ++j) {
      for (int k = 0; k < W[i].cols(); ++k) {
        double weight = ((double) rand() / (RAND_MAX));
        if (positive_weights) {
          W[i](j, k) = std::abs(weight);
        } else {
          W[i](j, k) = weight;
        }

        W[i](j,k) = ((double) rand() / (RAND_MAX));
      }
    }
    b[i].setOnes();
  }
  x[end] = Vector(topo[end]);
  params = Vector(ffnn_count(W, b));
  num_params = params.size();
  ffnn_seed(num_params, W, b);
}

template <class T>
typename FFNN<T>::Vector FFNN<T>::f(Vector const& v)
{
  Vector result(v.size());
  for (int i = 0; i < int(v.size()); ++i) {
    result[i] = activation(v[i]);
  }
  return result;
}

template <class T>
typename FFNN<T>::Vector FFNN<T>::evaluate(Vector const& x_in)
{
  x[0] = x_in;
  for (int i = 0; i < num_vectors - 2; ++i) {
    Vector const z = W[i] * x[i] + b[i];
    x[i+1] = f(z);
  }
  int const end = num_vectors - 1;
  x[end] = W[end-1] * x[end-1] + b[end-1];
  return x[end];
}

template <class T>
typename FFNN<T>::Vector const& FFNN<T>::get_params()
{
  int idx = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    for (int j = 0; j < W[i].rows(); ++j) {
      for (int k = 0; k < W[i].cols(); ++k) {
        params[idx++] = W[i](j,k);
      }
    }
    for (int j = 0; j < b[i].size(); ++j) {
      params[idx++] = b[i][j];
    }
  }
  return params;
}

template <class T>
void FFNN<T>::set_params(Vector const& p)
{
  params = p;
  int idx = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    for (int j = 0; j < W[i].rows(); ++j) {
      for (int k = 0; k < W[i].cols(); ++k) {
        W[i](j,k) = params[idx++];
      }
    }
    for (int j = 0; j < b[i].size(); ++j) {
      b[i][j] = params[idx++];
    }
  }
  ffnn_seed(num_params, W, b);
}

template class FFNN<double>;
template class FFNN<SFADT>;
template class FFNN<DFADT>;

}
