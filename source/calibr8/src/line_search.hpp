#pragma once

//! \file line_search.hpp
//! \brief Backtracking line search with cubic (Hermite) interpolation and the
//! Armijo sufficient-decrease condition, shared by the global (primal) and local
//! (constitutive) Newton solves.
//! \details This is the standard globalization for damped Newton on a nonlinear
//! system R = 0, using the merit phi(alpha) = 1/2 ||r||^2. It accepts the largest
//! trial step that sufficiently decreases phi; Each rejected step picks the
//! next trial as the minimizer of the cubic that matches phi and phi' at the base
//! point and at the current trial (a two-point Hermite cubic). The trial slope
//! phi'(alpha) is available cheaply -- the Jacobian is assembled at every trial
//! so the interpolant uses it; only sufficient decrease is required for
//! acceptance.

#include <algorithm>
#include <cmath>
#include "control.hpp"

namespace calibr8 {

//! \brief Parameters for the backtracking line search.
struct LineSearchParams {
  double c1 = 1.0e-4;         //!< sufficient-decrease (Armijo) constant
  double backtrack_min = 0.5; //!< next step >= backtrack_min * current step
  double backtrack_max = 0.9; //!< next step <= backtrack_max * current step
  int max_evals = 10;         //!< maximum merit-function evaluations
  bool print = false;         //!< print convergence information
  char const* tag = "";       //!< prefix for printed messages ("(local) " locally)
};

namespace line_search_detail {

//! \brief Minimizer of the cubic that matches (phi, slope) at the base point
//! (0, phi_0, dphi_0) and at the trial (a, phi, slope_a). Standard two-point
//! cubic interpolation; falls back to halving when there is no real interior
//! minimizer. The caller safeguards the returned value to a fraction of a.
inline double cubic_min(
    double phi_0, double dphi_0,
    double a, double phi, double slope_a) {
  double const d1 = dphi_0 + slope_a - 3. * (phi_0 - phi) / (0. - a);
  double const radicand = d1 * d1 - dphi_0 * slope_a;
  if (radicand < 0.) return 0.5 * a;            // no real interior minimizer
  double const d2 = std::sqrt(radicand);        // a > 0, so the sign is +
  double const denom = slope_a - dphi_0 + 2. * d2;
  if (denom == 0.) return 0.5 * a;
  return a - a * (slope_a + d2 - d1) / denom;
}

}  // namespace line_search_detail

//! \brief Backtracking Armijo line search with Hermite-cubic interpolation.
//! \param p The line search parameters.
//! \param phi_0 Merit value at the base point, phi(0) = 1/2 ||r_0||^2.
//! \param dphi_0 Slope at the base point, phi'(0) = -||r_0||^2 < 0.
//! \param eval Callable bool eval(double alpha, double& phi, double& slope):
//! advance the problem to step \p alpha, set phi = phi(alpha) and
//! slope = phi'(alpha), and return false if the residual evaluation failed (the
//! search then contracts the step). The problem is left at the last evaluated
//! step; the caller moves it to the returned step.
//! \returns The accepted step (the first satisfying sufficient decrease), or the
//! most contracted step tried if none did within max_evals.
template <class Evaluator>
double line_search(
    LineSearchParams const& p,
    double phi_0,
    double dphi_0,
    Evaluator&& eval) {

  using line_search_detail::cubic_min;
  double const armijo_slope = p.c1 * dphi_0;

  double alpha = 1.;            // start at the full Newton step
  double last_eval_alpha = 1.;  // most contracted step actually evaluated

  for (int n = 1; n <= p.max_evals; ++n) {

    double phi, slope;
    if (!eval(alpha, phi, slope)) {
      // Failed residual evaluation: contract and retry.
      alpha *= 0.5;
      continue;
    }
    last_eval_alpha = alpha;

    if (phi <= phi_0 + alpha * armijo_slope) {
      if (p.print)
        print(" > %sline search: alpha = %.3e (%d evals)", p.tag, alpha, n);
      return alpha;
    }

    // Minimize the Hermite cubic through the base point and this trial, then
    // safeguard the next step to a fraction of the current one.
    double const alpha_model = cubic_min(phi_0, dphi_0, alpha, phi, slope);
    alpha = std::min(std::max(alpha_model, p.backtrack_min * alpha),
                     p.backtrack_max * alpha);
  }

  if (p.print)
    print(" > %sline search: reached max evals, alpha = %.3e", p.tag, last_eval_alpha);
  return last_eval_alpha;
}

}
