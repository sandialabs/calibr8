#include <PCU.h>
#include "control.hpp"
#include "fd_vfm_objective.hpp"
#include "local_residual.hpp"

namespace calibr8 {

FD_VFM_Objective::FD_VFM_Objective(RCP<ParameterList> params) : Objective(params) {
  m_virtual_power = rcp(new VirtualPower(m_params, m_state, m_state->disc));
  ParameterList& inverse_params = m_params->sublist("inverse", true);
  std::string load_in_file = inverse_params.get<std::string>("load input file");
  std::ifstream in_file(load_in_file);
  std::string line;
  while (getline(in_file, line)) {
    m_load_data.push_back(std::stod(line));
  }
}

FD_VFM_Objective::~FD_VFM_Objective() {}

double FD_VFM_Objective::value(ROL::Vector<double> const& p, double&) {

  ROL::Ptr<Array1D<double> const> xp = getVector(p);

  if (param_diff(*xp)) {
    Array1D<double> const unscaled_params = transform_params(*xp, false);
    m_state->residuals->local->set_params(unscaled_params);
    m_state->d_residuals->local->set_params(unscaled_params);

    ParameterList& problem_params = m_params->sublist("problem", true);
    ParameterList& inverse_params = m_params->sublist("inverse", true);
    int const nsteps = problem_params.get<int>("num steps");
    double const dt = problem_params.get<double>("step size");
    double const thickness = inverse_params.get<double>("thickness", 1.);
    double const internal_power_scale_factor
        = inverse_params.get<double>("internal power scale factor", 1.);
    bool const print_vfm_mismatch
        = inverse_params.get<bool>("print vfm mismatch", false);
    double t = 0.;
    double J = 0.;
    double internal_virtual_power;
    double load_at_step;
    double volume_internal_virtual_power;
    m_state->disc->destroy_primal();
    for (int step = 1; step <= nsteps; ++step) {
      t += dt;
      internal_virtual_power = m_virtual_power->compute_at_step(step, t, dt)
          * internal_power_scale_factor;
      load_at_step = m_load_data[step - 1];
      PCU_Add_Double(internal_virtual_power);
      volume_internal_virtual_power = thickness * internal_virtual_power;
      if (print_vfm_mismatch) {
        print("\nstep = %d", step);
        print("  internal_power = %e", volume_internal_virtual_power);
        print("  load = %e", load_at_step);
      }
      J += 0.5 * std::pow(volume_internal_virtual_power - load_at_step, 2);

    }
    m_J_old = J;

  }

  return m_J_old;
}

}
