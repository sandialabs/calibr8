#include "NN.hpp"

#include <iostream> // debug
#include <iomanip> // debug

namespace ML {

double relu(double x)       { return (x > 0.) ? x : 0.; }
double d_relu(double x)     { return (x > 0.) ? 1. : 0.; }
double sigmoid(double x)    { return 1. / (1. + std::exp(-x)); }
double d_sigmoid(double x)  { return sigmoid(x)*(1.-sigmoid(x)); }
double my_tanh(double x)    { return std::tanh(x); }
double d_my_tanh(double x)  { return 1. - my_tanh(x)*my_tanh(x); }

static int count(
    std::vector<Matrix> const& W,
    std::vector<Vector> const& b)
{
  int num = 0;
  for (size_t i = 0; i < W.size(); ++i) {
    num += W[i].size();
    num += b[i].size();
  }
  return num;
}

static void check_topo(std::vector<int> const& topo)
{
  if (topo.size() < 3) {
    throw std::runtime_error("NN: not enough layers");
  }
}

static void check_x(Vector const& x, int num_inputs)
{
  if (x.size() != num_inputs) {
    throw std::runtime_error("NN: x -> invalid size");
  }
}

static void check_y(Vector const& y, int num_outputs)
{
  if (y.size() != num_outputs) {
    throw std::runtime_error("NN: y -> invalid size");
  }
}

NN::NN(const char* act, std::vector<int> const& topo)
{
  check_topo(topo);
  srand(10);
  std::string const type = act;
  if (type == "relu") {
    activation = relu;
    d_activation = d_relu;
  } else if (type == "sigmoid") {
    activation = sigmoid;
    d_activation = d_sigmoid;
  } else if (type == "tanh") {
    activation = tanh;
    d_activation = d_my_tanh;
  } else {
    throw std::runtime_error("NN: unkown " + type);
  }
  num_vectors = int(topo.size());
  num_inputs = topo[0];
  num_outputs = topo.back();
  W.resize(num_vectors - 1);
  b.resize(num_vectors - 1);
  x.resize(num_vectors);
  dx.resize(num_vectors);
  int const end = num_vectors - 1;
  for (int i = 0; i < end; ++i) {
    int const n_ip1 = topo[i+1];
    int const n_i = topo[i];
    W[i] = Matrix(n_ip1, n_i);
    b[i] = Vector(n_ip1);
    for (int j = 0; j < W[i].rows(); ++j) {
      for (int k = 0; k < W[i].cols(); ++k) {
        W[i](j,k) = ((double) rand() / (RAND_MAX));
//        W[i](j,k) = 0.125;
      }
    }
    b[i].setOnes();
    x[i] = Vector(n_i);
    dx[i] = Vector(n_i);
  }
  x[end] = Vector(topo[end]);
  dx[end] = Vector(topo[end]);
  params = Vector(count(W, b));
  num_params = params.size();
}

Vector const& NN::get_params()
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

void NN::set_params(Vector const& p)
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
}

Vector NN::feed_forward(Vector const& x_in)
{
  check_x(x_in, num_inputs);
  x[0] = x_in;
  dx[0] = x_in;
  for (int i = 0; i < num_vectors - 2; ++i) {
    Vector const z = W[i] * x[i] + b[i];
    x[i+1] = f(z);
    dx[i+1] = df(z);
  }
  int const end = num_vectors - 1;
  x[end] = W[end-1] * x[end-1] + b[end-1];
  dx[end].setOnes();
  return x[end];
}

static Vector flatten(Matrix const& M, Vector const& v)
{
  Vector result = Vector(M.rows() * M.cols() + v.size());
  int idx = 0;
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      result[idx++] = M(i,j);
    }
  }
  for (int i = 0; i < v.size(); ++i) {
    result[idx++] = v[i];
  }
  return result;
}

static void assign_at(Matrix& M, int i, int offset, Vector const& v)
{
  for (int j = 0; j < v.size(); ++j) {
    M(i, offset + j) = v[j];
  }
}

std::array<Matrix, 2> NN::differentiate(Vector const& y_out)
{
  check_y(y_out, num_outputs);
  std::array<Matrix, 2> derivs;
  derivs[0] = Matrix(num_outputs, num_params);
  derivs[1] = Matrix(num_outputs, num_inputs);
  Matrix U = Matrix::Identity(num_outputs, num_outputs);
  int const end = num_vectors - 1;
  int offset = num_params;
  for (int l = end-1; l >= 0; --l) {
    if (l < end-1) {
      // W and dx are offset in their indexing
      // this is why we use dx[l+1] instead of dx[l]
      Matrix V = W[l+1] * dx[l+1].asDiagonal();
      U = U*V;
    }
    offset -= (W[l].size() + b[l].size());
    for (int i = 0; i < num_outputs; ++i) {
      Vector U_i = U(i, Eigen::all);
      Vector a = x[l];
      Matrix dy_dW = U_i * a.transpose();
      Vector dy_db = U_i;
      Vector dy_dparams = flatten(dy_dW, dy_db);
      assign_at(derivs[0], i, offset, dy_dparams);
    }
  }
  derivs[1] = U * W[0];
  return derivs;
}

Vector NN::f(Vector const& v)
{
  Vector result(v.size());
  for (int i = 0; i < int(v.size()); ++i) {
    result[i] = activation(v[i]);
  }
  return result;
}

Vector NN::df(Vector const& v)
{
  Vector result(v.size());
  for (int i = 0; i < int(v.size()); ++i) {
    result[i] = d_activation(v[i]);
  }
  return result;
}

}
