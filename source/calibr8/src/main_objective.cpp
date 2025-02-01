#include <PCU.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "state.hpp"
#include "virtual_power.hpp"

using namespace calibr8;

static ParameterList get_valid_pdeco_params()
{
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("dirichlet bcs");
  p.sublist("traction bcs");
  p.sublist("linear algebra");
  p.sublist("quantity of interest");
  p.sublist("inverse");
  return p;
}

static ParameterList get_valid_virtual_fields_params()
{
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("inverse");
  p.sublist("virtual fields");
  return p;
}

class Objective
{
  public:
    Objective(RCP<ParameterList> params, bool evaluate_gradient);
    ~Objective() {};
    virtual void evaluate() = 0;
    void write_output();
    void cleanup();
  protected:
    void setup_opt_params(ParameterList const& inverse_params);
    RCP<ParameterList> m_params;
    Array1D<RCP<State>> m_state;
    Array1D<RCP<Primal>> m_primal;
    Array1D<RCP<ParameterList>> m_prob_params;
    int m_model_form = 0;
    int m_num_problems;
    int m_num_opt_params;
    bool m_evaluate_gradient;
    double m_objective_value = 0.;
    std::vector<double> m_objective_gradient;
};

Objective::Objective(RCP<ParameterList> params, bool evaluate_gradient)
{
  m_params = params;
  m_evaluate_gradient = evaluate_gradient;

  bool const is_multiple = m_params->isSublist("problems");
  if (is_multiple) {
    auto problems_list = m_params->sublist("problems");
    for (auto problem_entry : problems_list) {
      auto problem_list = Teuchos::getValue<ParameterList>(problem_entry.second);
      auto rcp_prob_list = RCP(new ParameterList);
      *rcp_prob_list = problem_list;
      m_prob_params.push_back(rcp_prob_list);
      auto state = rcp(new State(problem_list));
      auto primal = rcp(new Primal(rcp_prob_list, state, state->disc));
      m_state.push_back(state);
      m_primal.push_back(primal);
    }
    m_num_problems = m_state.size();
  } else {
    m_num_problems = 1;
    m_state.resize(1);
    m_primal.resize(1);
    m_prob_params.resize(1);
    m_state[0] = rcp(new State(*m_params));
    m_primal[0] = rcp(new Primal(m_params, m_state[0], m_state[0]->disc));
    m_prob_params[0] = m_params;
  }
  setup_opt_params(m_params->sublist("inverse", true));
}

void Objective::write_output()
{
  if (PCU_Comm_Self() == 0) {
    std::ofstream obj_file;
    std::string const obj_filename = "objective_value.txt";
    obj_file.open(obj_filename);
    obj_file << std::scientific << std::setprecision(17);
    obj_file << m_objective_value << "\n";
    obj_file.close();
  }
  if (m_evaluate_gradient) {
    std::ofstream grad_file;
    std::string const grad_filename = "objective_gradient.txt";
    grad_file.open(grad_filename);
    grad_file << std::scientific << std::setprecision(17);
    for (int i = 0; i < m_num_opt_params; ++i) {
      grad_file << m_objective_gradient[i] << "\n";
    }
    grad_file.close();
  }
}

void Objective::cleanup()
{
  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->disc->destroy_primal();
  }
}

void Objective::setup_opt_params(ParameterList const& inverse_params)
{
  Array1D<std::string> const& elem_set_names =
      m_state[0]->residuals->local[m_model_form]->elem_set_names();
  Array1D<std::string> const& param_names =
      m_state[0]->residuals->local[m_model_form]->param_names();

  ParameterList const& all_material_params =
      inverse_params.sublist("materials");

  int const num_elem_sets = elem_set_names.size();
  int const num_params_total = param_names.size();

  Array2D<int> active_indices(num_elem_sets);
  m_num_opt_params = 0;

  for (int es = 0; es < num_elem_sets; ++es) {
    ParameterList material_params =
        all_material_params.sublist(elem_set_names[es]);
    for (int i = 0; i < num_params_total; ++i) {
      std::string param_name = param_names[i];
      if (material_params.isParameter(param_name)) {
        active_indices[es].push_back(i);
        m_num_opt_params++;
      }
    }
  }

  int const num_dfad_params = m_state[0]->residuals->local[m_model_form]->num_dfad_params();
  m_num_opt_params += num_dfad_params;

  m_objective_gradient.resize(m_num_opt_params, 0.);

  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_state[prob]->residuals->local[m_model_form]->set_active_indices(active_indices);
    m_state[prob]->d_residuals->local[m_model_form]->set_active_indices(active_indices);
  }
}

class PDECO_Objective : public Objective
{
  public:
    PDECO_Objective(RCP<ParameterList> params, bool evaluate_gradient);
    ~PDECO_Objective() {};
    void evaluate() override;
  private:
    Array1D<RCP<Adjoint>> m_adjoint;
};

PDECO_Objective::PDECO_Objective(
    RCP<ParameterList> params,
    bool evaluate_gradient) : Objective(params, evaluate_gradient)
{
  if (m_evaluate_gradient) {
    m_adjoint.resize(m_num_problems);
    for (int prob = 0; prob < m_num_problems; ++prob) {
      m_adjoint[prob] = rcp(new Adjoint(m_prob_params[prob], m_state[prob],
          m_state[prob]->disc));
    }
  }
}

void PDECO_Objective::evaluate()
{
  for (int prob = 0; prob < m_num_problems; ++prob) {
    double J_prob = 0.;
    int const nsteps = m_state[prob]->disc->num_time_steps();
    for (int step = 1; step <= nsteps; ++step) {
      m_primal[prob]->solve_at_step(step);
      J_prob += eval_qoi(m_state[prob], m_state[prob]->disc, step);
    }
    J_prob = PCU_Add_Double(J_prob);
    m_objective_value += J_prob;
  }
  if (m_evaluate_gradient) {
    for (int prob = 0; prob < m_num_problems; ++prob) {
      Array1D<double> grad_at_step(m_num_opt_params);
      int const nsteps = m_state[prob]->disc->num_time_steps();
      m_state[prob]->disc->create_adjoint(m_state[prob]->residuals, nsteps);
      for (int step = nsteps; step > 0; --step) {
        m_adjoint[prob]->solve_at_step(step);
        grad_at_step = eval_qoi_gradient(m_state[prob], step);
        for (int i = 0; i < m_num_opt_params; ++i) {
          m_objective_gradient[i] += grad_at_step[i];
        }
      }
      m_state[prob]->disc->destroy_adjoint();
    }
    PCU_Add_Doubles(m_objective_gradient.data(), m_num_opt_params);
  }
}

class VFM_Objective : public Objective
{
  public:
    VFM_Objective(RCP<ParameterList> params, bool evaluate_gradient);
    ~VFM_Objective() {};
    void evaluate() override;
  protected:
    RCP<VirtualPower> m_virtual_power;
    std::string m_load_in_file;
    Array1D<double> m_load_data;
};

VFM_Objective::VFM_Objective(
    RCP<ParameterList> params,
    bool evaluate_gradient) : Objective(params, evaluate_gradient)
{
  m_virtual_power = rcp(new VirtualPower(m_params, m_state[m_model_form],
      m_state[m_model_form]->disc, m_num_opt_params));
  ParameterList& inverse_params = m_params->sublist("inverse", true);
  std::string load_in_file = inverse_params.get<std::string>("load input file");
  std::ifstream in_file(load_in_file);
  std::string line;
  while (getline(in_file, line)) {
    m_load_data.push_back(std::stod(line));
  }
}

void VFM_Objective::evaluate()
{
  ParameterList& inverse_params = m_params->sublist("inverse", true);
  double const obj_scale_factor = inverse_params.get<double>("objective scale factor", 1.);
  double dt;
  double internal_virtual_power;
  double load_at_step;
  double virtual_power_mismatch;

  double scaled_virtual_power_mismatch;
  Array1D<double> internal_virtual_power_vec;

  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);

  for (int prob = 0; prob < m_num_problems; ++prob) {
    double J_prob = 0.;
    int const nsteps = m_state[prob]->disc->num_time_steps();
    internal_virtual_power_vec.resize(nsteps, 0.);
    double const total_time = m_state[prob]->disc->time(nsteps) - m_state[prob]->disc->time(0);

    m_state[prob]->disc->destroy_primal();

      for (int step = 1; step <= nsteps; ++step) {
        dt = m_state[prob]->disc->dt(step);
        internal_virtual_power = m_virtual_power->compute_at_step(step);
        internal_virtual_power_vec[step - 1] = internal_virtual_power;
      }

      for (int step = nsteps; step > 0; --step) {
        dt = m_state[prob]->disc->dt(step);
        load_at_step = m_load_data[step - 1];

        virtual_power_mismatch = internal_virtual_power_vec[step - 1] - load_at_step;
        scaled_virtual_power_mismatch = virtual_power_mismatch * obj_scale_factor * dt / total_time;
        J_prob += 0.5 * virtual_power_mismatch * scaled_virtual_power_mismatch;

        if (m_evaluate_gradient) {
          m_virtual_power->compute_at_step_adjoint(
              step, scaled_virtual_power_mismatch, grad_at_step
          );

          for (int i = 0; i < m_num_opt_params; ++i) {
            m_objective_gradient[i] += grad_at_step[i];
          }
        }
      }
    m_objective_value += J_prob;
  }
  PCU_Add_Doubles(m_objective_gradient.data(), m_num_opt_params);
}

class EUCLID_Objective : public VFM_Objective
{
  public:
    EUCLID_Objective(RCP<ParameterList> params, bool evaluate_gradient);
    ~EUCLID_Objective() {};
    void evaluate() override;
};

EUCLID_Objective::EUCLID_Objective(
    RCP<ParameterList> params,
    bool evaluate_gradient) : VFM_Objective(params, evaluate_gradient) {}

void EUCLID_Objective::evaluate()
{

  ParameterList& inverse_params = m_params->sublist("inverse", true);
  std::string const ns_names_str = "node set names";
  std::string const vf_components_str = "virtual field components";
  std::string const obj_scale_factors_str = "objective scale factors";
  std::string const load_scale_factors_str = "load scale factors";
  ALWAYS_ASSERT(inverse_params.isParameter(ns_names_str));
  ALWAYS_ASSERT(inverse_params.isParameter(obj_scale_factors_str));
  ALWAYS_ASSERT(inverse_params.isParameter(load_scale_factors_str));

  auto const node_set_names = (inverse_params.get<Teuchos::Array<std::string>>(ns_names_str)).toVector();
  auto const vf_components = (inverse_params.get<Teuchos::Array<int>>(vf_components_str)).toVector();
  auto const obj_scale_factors = (inverse_params.get<Teuchos::Array<double>>(obj_scale_factors_str)).toVector();
  auto const load_scale_factors = (inverse_params.get<Teuchos::Array<double>>(load_scale_factors_str)).toVector();

  int const num_virtual_fields = node_set_names.size();
  ALWAYS_ASSERT(vf_components.size() == num_virtual_fields);
  ALWAYS_ASSERT(obj_scale_factors.size() == num_virtual_fields);
  ALWAYS_ASSERT(load_scale_factors.size() == num_virtual_fields);

  std::string node_set_name;
  int vf_component;
  double obj_scale_factor;
  double load_scale_factor;

  double dt;
  double internal_virtual_power;
  double load_at_step;
  double virtual_power_mismatch;

  double scaled_virtual_power_mismatch;
  Array2D<double> internal_virtual_power_vec;
  internal_virtual_power_vec.resize(num_virtual_fields);

  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);

  for (int prob = 0; prob < m_num_problems; ++prob) {

    double J_prob = 0.;
    int const nsteps = m_state[prob]->disc->num_time_steps();
    double const total_time = m_state[prob]->disc->time(nsteps) - m_state[prob]->disc->time(0);
    for (int vf_idx = 0; vf_idx < num_virtual_fields; ++vf_idx) {
      internal_virtual_power_vec[vf_idx].resize(nsteps, 0.);
    }

    m_state[prob]->disc->destroy_primal();

      for (int step = 1; step <= nsteps; ++step) {
        dt = m_state[prob]->disc->dt(step);
        m_virtual_power->compute_residual_at_step(step);
        for (int vf_idx = 0; vf_idx < num_virtual_fields; ++vf_idx) {
          node_set_name = node_set_names[vf_idx];
          vf_component = vf_components[vf_idx];
          m_state[prob]->disc->set_virtual_field_from_node_set(node_set_name, vf_component);
          m_virtual_power->populate_vf_vector();
          internal_virtual_power = m_virtual_power->compute_internal_virtual_power();
          internal_virtual_power_vec[vf_idx][step - 1] = internal_virtual_power;
        }
      }

      for (int vf_idx = 0; vf_idx < num_virtual_fields; ++vf_idx) {
        node_set_name = node_set_names[vf_idx];
        vf_component = vf_components[vf_idx];
        obj_scale_factor = obj_scale_factors[vf_idx];
        load_scale_factor = load_scale_factors[vf_idx];
        m_state[prob]->disc->set_virtual_field_from_node_set(node_set_name, vf_component);

        for (int step = nsteps; step > 0; --step) {
          dt = m_state[prob]->disc->dt(step);
          load_at_step = load_scale_factor * m_load_data[step - 1];
          virtual_power_mismatch = internal_virtual_power_vec[vf_idx][step - 1] - load_at_step;
          scaled_virtual_power_mismatch = virtual_power_mismatch * obj_scale_factor * dt / total_time;
          J_prob += 0.5 * virtual_power_mismatch * scaled_virtual_power_mismatch;
          if (m_evaluate_gradient) {
            m_virtual_power->compute_at_step_adjoint(
                step, scaled_virtual_power_mismatch, grad_at_step
            );
            for (int i = 0; i < m_num_opt_params; ++i) {
              m_objective_gradient[i] += grad_at_step[i];
            }
          }
        }
      }
    m_objective_value += J_prob;
  }
  PCU_Add_Doubles(m_objective_gradient.data(), m_num_opt_params);
}

static RCP<Objective> create_objective(
    RCP<ParameterList> params,
    bool evaluate_gradient)
{
  ParameterList const& inverse_params = params->sublist("inverse", true);
  auto const& obj_type = inverse_params.get<std::string>("objective type");
  if (obj_type == "pdeco") {
    params->validateParameters(get_valid_pdeco_params(), 0);
    return rcp(new PDECO_Objective(params, evaluate_gradient));
  } else if (obj_type == "vfm") {
    params->validateParameters(get_valid_virtual_fields_params(), 0);
    return rcp(new VFM_Objective(params, evaluate_gradient));
  } else if (obj_type == "euclid") {
    params->validateParameters(get_valid_virtual_fields_params(), 0);
    return rcp(new EUCLID_Objective(params, evaluate_gradient));
  } else {
    fail("objective type not implemented");
  }
}

int main(int argc, char** argv)
{
  initialize();
  ALWAYS_ASSERT(argc == 3);
  {
    std::string const yaml_input = argv[1];
    std::string const compute_gradient = argv[2];
    bool evaluate_gradient;
    if (compute_gradient == "true") {
      evaluate_gradient = true;
    } else {
      evaluate_gradient = false;
    }

    auto params = rcp(new ParameterList);
    Teuchos::updateParametersFromYamlFile(yaml_input, params.ptr());

    RCP<Objective> objective = create_objective(params, evaluate_gradient);
    objective->evaluate();
    objective->cleanup();
    objective->write_output();
  }
  finalize();
}
