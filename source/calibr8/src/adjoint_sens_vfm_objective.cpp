#include <PCU.h>
#include "control.hpp"
#include "adjoint_sens_vfm_objective.hpp"
#include "local_residual.hpp"

namespace calibr8 {

Adjoint_VFM_Objective::Adjoint_VFM_Objective(RCP<ParameterList> params) : Objective(params) {
  //TODO: generalize to multiple states
  m_virtual_power = rcp(new VirtualPower(m_params, m_state[0], m_state[0]->disc,
        m_num_opt_params));
  ParameterList& inverse_params = m_params->sublist("inverse", true);
  std::string load_in_file = inverse_params.get<std::string>("load input file");
  std::ifstream in_file(load_in_file);
  std::string line;
  while (getline(in_file, line)) {
    m_load_data.push_back(std::stod(line));
  }
}

Adjoint_VFM_Objective::~Adjoint_VFM_Objective() {}

double Adjoint_VFM_Objective::value(ROL::Vector<double> const& p, double&) {

  //TODO: generalize to multiple states
  ROL::Ptr<Array1D<double> const> xp = getVector(p);

  if (param_diff(*xp)) {
    Array1D<double> const unscaled_params = transform_params(*xp, false);
    m_state[0]->residuals->local[m_model_form]->set_params(unscaled_params);
    m_state[0]->d_residuals->local[m_model_form]->set_params(unscaled_params);

    ParameterList& inverse_params = m_params->sublist("inverse", true);
    double const obj_scale_factor = inverse_params.get<double>("objective scale factor");
    int const nsteps = m_state[0]->disc->num_time_steps();
    double const total_time = m_state[0]->disc->time(nsteps) - m_state[0]->disc->time(0);
    double const thickness = inverse_params.get<double>("thickness", 1.);
    bool const print_vfm_mismatch
      = inverse_params.get<bool>("print vfm mismatch", false);
    double dt = 0.;
    double J = 0.;
    double internal_virtual_power;
    double load_at_step;
    double volume_internal_virtual_power;
    m_state[0]->disc->destroy_primal();
    for (int step = 1; step <= nsteps; ++step) {
      dt = m_state[0]->disc->dt(step);
      internal_virtual_power = m_virtual_power->compute_at_step(step);
      load_at_step = m_load_data[step - 1];
      PCU_Add_Double(internal_virtual_power);
      volume_internal_virtual_power = thickness * internal_virtual_power;
      if (print_vfm_mismatch) {
        print("\nstep = %d", step);
        print("  internal_power = %e", volume_internal_virtual_power);
        print("  load = %e", load_at_step);
      }
      J += 0.5 * obj_scale_factor * dt / total_time
        * std::pow(volume_internal_virtual_power - load_at_step, 2);

    }
    m_J_old = J;

  }

  return m_J_old;
}

void Adjoint_VFM_Objective::gradient(
    ROL::Vector<double>& g,
    ROL::Vector<double> const& p,
    double&) {

  //TODO: generalize to multiple states

  ROL::Ptr<Array1D<double>> gp = getVector(g);
  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  Array1D<double> const unscaled_params = transform_params(*xp, false);
  m_state[0]->residuals->local[m_model_form]->set_params(unscaled_params);
  m_state[0]->d_residuals->local[m_model_form]->set_params(unscaled_params);

  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);

  ParameterList& inverse_params = m_params->sublist("inverse", true);
  double const obj_scale_factor = inverse_params.get<double>("objective scale factor");
  int const nsteps = m_state[0]->disc->num_time_steps();
  double const total_time = m_state[0]->disc->time(nsteps) - m_state[0]->disc->time(0);
  double const thickness = inverse_params.get<double>("thickness", 1.);
  double dt = 0.;
  double J = 0.;
  double internal_virtual_power;
  double load_at_step;
  double virtual_power_mismatch;
  double volume_internal_virtual_power;
  m_state[0]->disc->destroy_primal();
  Array1D<double> internal_virtual_power_vec(nsteps);
  double scaled_virtual_power_mismatch;

  for (int step = 1; step <= nsteps; ++step) {
    dt = m_state[0]->disc->dt(step);
    internal_virtual_power = m_virtual_power->compute_at_step(step);
    internal_virtual_power_vec[step - 1] = internal_virtual_power;
  }

  for (int step = nsteps; step > 0; --step) {
    dt = m_state[0]->disc->dt(step);
    load_at_step = m_load_data[step - 1];

    virtual_power_mismatch = internal_virtual_power_vec[step - 1] * thickness - load_at_step;
    scaled_virtual_power_mismatch = virtual_power_mismatch * obj_scale_factor * dt / total_time;
    J += 0.5 * virtual_power_mismatch * scaled_virtual_power_mismatch;

    m_virtual_power->compute_at_step_adjoint(step, scaled_virtual_power_mismatch, grad_at_step);

    for (int i = 0; i < m_num_opt_params; ++i) {
      grad[i] += grad_at_step[i];
    }
  }
  m_J_old = J;

  Array1D<double> const canonical_grad = transform_gradient(grad);
  for (int i = 0; i < m_num_opt_params; ++i) {
    (*gp)[i] = canonical_grad[i];
  }
}

}
