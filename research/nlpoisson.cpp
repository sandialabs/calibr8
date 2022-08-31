#include "nlpoisson.hpp"

namespace calibr8 {

double eval_sin_body_force(apf::Vector3 const& pt, double alpha) {
  using std::pow;
  using std::exp;
  using std::sin;
  using std::cos;
  double const x = pt[0];
  double const y = pt[1];
  static constexpr double pi = 3.14159265358979323;
  return 8*pow(pi, 2)*sin(2*pi*x)*sin(2*pi*y);
}

double eval_sin_exp_body_force(apf::Vector3 const& pt, double alpha) {
  (void)pt;
  (void)alpha;
  return 1.;
}

}
