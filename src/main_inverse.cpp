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

// TODO: determine which of these are defaults and clean up
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
      .set("Function Evaluation Limit", 3);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .set("Sufficient Decrease Tolerance", 1.0e-4);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .set("Initial Step Size", 1.0);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Descent Method")
      .set("Type", "Quasi-Newton");
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Descent Method")
      .set("Nonlinear CG Type", "Hestenes-Stiefel");
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Curvature Condition")
      .set("Type", "Strong Wolfe Conditions");
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Curvature Condition")
      .set("General Parameter", 0.9);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Curvature Condition")
      .set("Generalized Wolfe Parameter", 0.6);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Line-Search Method")
      .set("Type", "Cubic Interpolation");
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Line-Search Method")
      .set("Backtracking Rate", 0.5);
  rol_params->sublist("General")
      .sublist("Step")
      .sublist("Line Search")
      .sublist("Line-Search Method")
      .set("Bracketing Tolerance", 1.0e-8);

  //rol_params->sublist("Status Test").set("Gradient Tolerance", 1.0e-4);
  //rol_params->sublist("Status Test").set("Step Tolerance", 1.0e-8);
  //rol_params->sublist("Status Test").set("Iteration Limit", 200);

  rol_params->sublist("Status Test").set("Gradient Tolerance", 1.0e-12);
  rol_params->sublist("Status Test").set("Step Tolerance", 1.0e-12);
  rol_params->sublist("Status Test").set("Iteration Limit", 0);
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

    auto rol_objective = create_rol_objective(params, grad_type);

    Array1D<double> const ig = rol_objective->opt_params();
    size_t const dim = ig.size();
    ROL::Ptr<std::vector<double> > x_ptr = ROL::makePtr<std::vector<double>>(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        (*x_ptr)[i] = ig[i];
    }
    ROL::StdVector<double> x(x_ptr);

    // TODO: change to [-1, 1] bounds and implement parameter scaling
    ROL::Ptr<ROL::Bounds<double>> bound;
    ROL::Ptr<std::vector<double>> lo_ptr = ROL::makePtr<std::vector<double>>(dim, 0.2);
    ROL::Ptr<std::vector<double>> hi_ptr = ROL::makePtr<std::vector<double>>(dim, 2000.);
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

#if 0
    ROL::Ptr<ROL::Step<double>> step =
      ROL::makePtr<ROL::LineSearchStep<double>>(*rol_params);
    ROL::Ptr<ROL::StatusTest<double>>
      status = ROL::makePtr<ROL::StatusTest<double>>(*rol_params);
    ROL::Algorithm<double> algo(step, status, false);
    ROL::Ptr<std::vector<double> > x_ptr = ROL::makePtr<std::vector<double>>(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        (*x_ptr)[i] = scaled_ig[i];
    }
    ROL::StdVector<double> x(x_ptr);
    ROL::Ptr<ROL::Bounds<double> > bound;
    ROL::Ptr<std::vector<double> > lo_ptr = ROL::makePtr<std::vector<double> >(dim,-1.0);
    ROL::Ptr<std::vector<double> > hi_ptr = ROL::makePtr<std::vector<double> >(dim,1.0);
    ROL::Ptr<ROL::Vector<double> > lop = ROL::makePtr<ROL::StdVector<double> >(lo_ptr);
    ROL::Ptr<ROL::Vector<double> > hip = ROL::makePtr<ROL::StdVector<double> >(hi_ptr);
    bound = ROL::makePtr<ROL::Bounds<double> >(lop,hip);
    ROL::Ptr<std::vector<double> > d_ptr = ROL::makePtr<std::vector<double> >(dim,0.1);
    ROL::StdVector<double> d(d_ptr);
    bool isProcZero = (PCU_Comm_Self() == 0);
    bool fd_check = optlist.get<bool>("fd check",false);
    std::vector<std::string> output;
    output = algo.run(x, *rol_objective, *bound, isProcZero, *outStream);
    if (isProcZero) {
      std::string outname = "ROL_out.txt";
      std::ofstream rolOut(outname);
      rolOut.precision(16);
      ROL::Ptr<const std::vector<double> > xp = (dynamic_cast<const ROL::StdVector<double>&>(x)).getVector();
      std::vector<double> optParams = rol_objective->unscale_params(*xp);
      for (size_t i = 0; i < output.size(); ++i) {
        *outStream << output[i];
        rolOut << output[i];
      }
      auto names = rol_objective->get_param_names();
      auto active = rol_objective->get_active();
      for (size_t i = 0; i < names.size(); ++i) {
        if (active[i]) {
          *outStream << names[i] << " = " << optParams[i] << std::endl;
          rolOut << names[i] << " = " << optParams[i] << std::endl;
        }
      }
      rolOut.close();
    }
#endif

  }
  finalize();
}
