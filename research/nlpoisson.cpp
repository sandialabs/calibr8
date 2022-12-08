#include "nlpoisson.hpp"

namespace calibr8 {

double eval_sin_exp_body_force(apf::Vector3 const& pt, double alpha) {
  using std::pow;
  using std::exp;
  using std::sin;
  using std::cos;
  double const x = pt[0];
  double const y = pt[1];
  static constexpr double pi = 3.14159265358979323;
  return -2*alpha*(exp(x + y)*sin(2*pi*x)*sin(2*pi*y) + 1.0)*pow(sin(2*pi*x) +
      2*pi*cos(2*pi*x), 2)*exp(2*x + 2*y)*pow(sin(2*pi*y), 2) -
    2*alpha*(exp(x + y)*sin(2*pi*x)*sin(2*pi*y) + 1.0)*pow(sin(2*pi*y) +
        2*pi*cos(2*pi*y), 2)*exp(2*x + 2*y)*pow(sin(2*pi*x), 2) -
    (alpha*pow(exp(x + y)*sin(2*pi*x)*sin(2*pi*y) + 1.0, 2) + 1)*
    (-4*pow(pi, 2)*sin(2*pi*x) + sin(2*pi*x) + 4*pi*cos(2*pi*x))*
    exp(x + y)*sin(2*pi*y) -
    (alpha*pow(exp(x + y)*sin(2*pi*x)*sin(2*pi*y) + 1.0, 2) + 1)*
    (-4*pow(pi, 2)*sin(2*pi*y) + sin(2*pi*y) + 4*pi*cos(2*pi*y))*exp(x + y)*sin(2*pi*x);
}

void concat(
    int nverts,
    int nedges,
    apf::Downward const& verts,
    apf::Downward const& edges,
    std::vector<apf::MeshEntity*>& ents) {
  ents.resize(nverts + nedges);
  int ctr = 0;
  for (int i = 0; i < nverts; ++i) {
    ents[ctr++] = verts[i];
  }
  for (int i = 0; i < nedges; ++i) {
    ents[ctr++] = edges[i];
  }
}

}
