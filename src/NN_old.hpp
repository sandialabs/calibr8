#pragma once

#include "Eigen/Core"

#include <array>
#include <vector>

namespace ML_old {

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

typedef double (*compwise_func)(double);

class NN
{
  public:
    NN(const char* activation, std::vector<int> const& topology);
    Vector const& get_params();
    void set_params(Vector const& p);
    Vector feed_forward(Vector const& x);
    std::array<Matrix, 2> differentiate(Vector const& y);
  private:
    compwise_func activation;
    compwise_func d_activation;
    Vector f(Vector const& v);
    Vector df(Vector const& v);
  private:
    std::vector<Matrix> W;
    std::vector<Vector> b;
    std::vector<Vector> x;
    std::vector<Vector> dx;
    int num_vectors;
    int num_inputs;
    int num_outputs;
    int num_params;
  private:
    Vector params;
};

}
