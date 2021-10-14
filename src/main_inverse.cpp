#include <ROL_Algorithm.hpp>
#include <ROL_Bounds.hpp>
#include <ROL_LineSearchStep.hpp>
#include <ROL_Objective.hpp>
#include <ROL_ParameterList.hpp>
#include <ROL_Stream.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint_objective.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "femu_objective.hpp"
#include "macros.hpp"

using namespace calibr8;

void set_default_rol_params(Teuchos::RCP<ParameterList> rol_params) {
  rol_params->sublist("General")
    .sublist("Secant")
    .set("Type", "Limited-Memory BFGS");
  rol_params->sublist("General")
    .sublist("Secant")
    .set("Maximum Storage", 20);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .set("Function Evaluation Limit", 5);
  rol_params->sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  rol_params->sublist("Status Test").set("Step Tolerance", 1.0e-12);
}

static ParameterList get_valid_params() {
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

RCP<Objective> create_rol_objective(
  RCP<ParameterList> params,
  std::string const& grad_type) {
  if (grad_type == "FEMU") {
    return rcp(new FEMU_Objective(params));
  }
  else if (grad_type == "adjoint") {
    return rcp(new Adjoint_Objective(params));
  }
  else {
    return Teuchos::null;
  }
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    const char* in = argv[1];
    auto params = rcp(new ParameterList);
    Teuchos::updateParametersFromYamlFile(in, params.ptr());
    params->validateParameters(get_valid_params(), 0);
    ROL::Ptr<std::ostream> outStream;
    outStream = ROL::makePtrFromRef(std::cout);
    auto rol_params = rcp(new ParameterList);
    set_default_rol_params(rol_params);


    ParameterList& inverse_params = params->sublist("inverse", true);
    std::string const grad_type = inverse_params.get<std::string>("gradient type");
    bool check_gradient  = inverse_params.get<bool>("check gradient", false);
    int const iteration_limit = inverse_params.get<int>("iteration limit", 20);
    rol_params->sublist("Status Test").set("Iteration Limit", iteration_limit);

    auto rol_objective = create_rol_objective(params, grad_type);

    Array1D<double> const model_params = rol_objective->model_params();
    Array1D<double> const active_model_params =
      rol_objective->extract_active_params(model_params);
    Array1D<double> const initial_guess =
      rol_objective->transform_params(model_params, true);
    size_t const dim = initial_guess.size();
    ROL::Ptr<std::vector<double> > x_ptr = ROL::makePtr<std::vector<double>>(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        (*x_ptr)[i] = initial_guess[i];
    }
    ROL::StdVector<double> x(x_ptr);

    ROL::Ptr<ROL::Bounds<double>> bound;
    ROL::Ptr<std::vector<double>> lo_ptr = ROL::makePtr<std::vector<double>>(dim, -1.);
    ROL::Ptr<std::vector<double>> hi_ptr = ROL::makePtr<std::vector<double>>(dim, 1.);
    ROL::Ptr<ROL::Vector<double>> lop = ROL::makePtr<ROL::StdVector<double>>(lo_ptr);
    ROL::Ptr<ROL::Vector<double>> hip = ROL::makePtr<ROL::StdVector<double>>(hi_ptr);
    bound = ROL::makePtr<ROL::Bounds<double>>(lop, hip);

    ROL::Ptr<ROL::Step<double>> step =
      ROL::makePtr<ROL::LineSearchStep<double>>(*rol_params);
    ROL::Ptr<ROL::StatusTest<double>>
      status = ROL::makePtr<ROL::StatusTest<double>>(*rol_params);
    ROL::Algorithm<double> algo(step, status, false);

    if (grad_type != "femu" && check_gradient) {
      ROL::Ptr<std::vector<double>> d_ptr = ROL::makePtr<std::vector<double> >(dim, 0.1);
      ROL::StdVector<double> d(d_ptr);
      (*rol_objective).checkGradient(x, d, true, *outStream, 13, 2);
    }

    std::vector<std::string> output;
    output = algo.run(x, *rol_objective, *bound, true, *outStream);
    for (size_t i = 0; i < output.size(); ++i) {
      *outStream << output[i];
    }

  }

  finalize();
}
