#include "nlpoisson.hpp"

namespace calibr8 {

double eval_manufactured_force(apf::Vector3 const& pt, double alpha) {
  using std::pow;
  using std::exp;
  using std::sin;
  using std::cos;
  double const x = pt[0];
  double const y = pt[1];
  static constexpr double pi = 3.14159265358979323;
  return
    (2.5*exp(2.5*x + 2.5*y)*sin(2*pi*x)*sin(2*pi*y) + 2*pi*exp(2.5*x + 2.5*y)*
     sin(2*pi*x)*cos(2*pi*y))*(-5.0*alpha*exp(5.0*x + 5.0*y)*pow(sin(2*pi*x), 2)*
     pow(sin(2*pi*y), 2) - 4*pi*alpha*exp(5.0*x + 5.0*y)*pow(sin(2*pi*x), 2)*sin(2*pi*y)*cos(2*pi*y)) +
     (2.5*exp(2.5*x + 2.5*y)*sin(2*pi*x)*sin(2*pi*y) + 2*pi*exp(2.5*x + 2.5*y)*sin(2*pi*y)*cos(2*pi*x))*
     (-5.0*alpha*exp(5.0*x + 5.0*y)*pow(sin(2*pi*x), 2)*pow(sin(2*pi*y), 2) - 4*pi*alpha*exp(5.0*x + 5.0*y)*
      sin(2*pi*x)*pow(sin(2*pi*y), 2)*cos(2*pi*x)) + (-alpha*exp(5.0*x + 5.0*y)*pow(sin(2*pi*x), 2)*
      pow(sin(2*pi*y), 2) - 1)*(-4*pow(pi, 2)*exp(2.5*x + 2.5*y)*sin(2*pi*x)*sin(2*pi*y) + 6.25*exp(2.5*x + 2.5*y)*
      sin(2*pi*x)*sin(2*pi*y) + 10.0*pi*exp(2.5*x + 2.5*y)*sin(2*pi*x)*cos(2*pi*y)) +
      (-alpha*exp(5.0*x + 5.0*y)*pow(sin(2*pi*x), 2)*pow(sin(2*pi*y), 2) - 1)*(-4*pow(pi, 2)*exp(2.5*x + 2.5*y)*
          sin(2*pi*x)*sin(2*pi*y) + 6.25*exp(2.5*x + 2.5*y)*sin(2*pi*x)*sin(2*pi*y) +
          10.0*pi*exp(2.5*x + 2.5*y)*sin(2*pi*y)*cos(2*pi*x));
}

}
