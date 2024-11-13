#include <PCU.h>
#include <ROL_Algorithm.hpp>
#include <ROL_Bounds.hpp>
#include <ROL_LineSearchStep.hpp>
#include <ROL_Objective.hpp>
#include <ROL_ParameterList.hpp>
#include <ROL_Stream.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "arrays.hpp"
#include "adjoint_objective.hpp"
#include "adjoint_sens_vfm_objective.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "fd_vfm_objective.hpp"
#include "forward_sens_vfm_objective.hpp"
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
}

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("problems");
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("dirichlet bcs");
  p.sublist("traction bcs");
  p.sublist("linear algebra");
  p.sublist("quantity of interest");
  //p.sublist("regression");
  p.sublist("inverse");
  p.sublist("virtual fields");
  return p;
}

RCP<Objective> create_rol_objective(
  RCP<ParameterList> params,
  std::string const& obj_type) {
  if (obj_type == "adjoint") {
    return rcp(new Adjoint_Objective(params));
  } else if (obj_type == "FEMU") {
    return rcp(new FEMU_Objective(params));
  } else if (obj_type == "FS_VFM") {
    return rcp(new FS_VFM_Objective(params));
  } else if (obj_type == "Adjoint_VFM") {
    return rcp(new Adjoint_VFM_Objective(params));
  } else if (obj_type == "VFM") {
    return rcp(new FD_VFM_Objective(params));
  } else {
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
    //ParameterList& regression_params = params->sublist("regression", false);
    std::string const obj_type = inverse_params.get<std::string>("objective type");

    bool check_gradient  = inverse_params.get<bool>("check gradient", false);
    int const iteration_limit = inverse_params.get<int>("iteration limit", 20);
    double const gradient_tol =
        inverse_params.get<double>("gradient tolerance", 1e-12);
    double const step_tol = inverse_params.get<double>("step tolerance", 1e-12);
    int const max_line_search_evals =
        inverse_params.get<int>("max line search evals", 5);

    rol_params->sublist("Status Test").set("Iteration Limit", iteration_limit);
    rol_params->sublist("Status Test").set("Gradient Tolerance", gradient_tol);
    rol_params->sublist("Status Test").set("Step Tolerance", step_tol);
    rol_params->sublist("Step")
      .sublist("Line Search")
      .set("Function Evaluation Limit", max_line_search_evals);

    auto rol_objective = create_rol_objective(params, obj_type);

    Array1D<double> const& active_params = rol_objective->active_params();
    Array1D<double> const& initial_guess =
        rol_objective->transform_params(active_params, true);
    int const dim = initial_guess.size();
    ROL::Ptr<Array1D<double>> x_ptr = ROL::makePtr<Array1D<double>>(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        (*x_ptr)[i] = initial_guess[i];
    }
    ROL::StdVector<double> x(x_ptr);

    ROL::Ptr<ROL::Bounds<double>> bound;
    ROL::Ptr<Array1D<double>> lo_ptr = ROL::makePtr<Array1D<double>>(dim, -1.);
    ROL::Ptr<Array1D<double>> hi_ptr = ROL::makePtr<Array1D<double>>(dim, 1.);
    ROL::Ptr<ROL::Vector<double>> lop = ROL::makePtr<ROL::StdVector<double>>(lo_ptr);
    ROL::Ptr<ROL::Vector<double>> hip = ROL::makePtr<ROL::StdVector<double>>(hi_ptr);
    bound = ROL::makePtr<ROL::Bounds<double>>(lop, hip);

    ROL::Ptr<ROL::Step<double>> step =
        ROL::makePtr<ROL::LineSearchStep<double>>(*rol_params);
    ROL::Ptr<ROL::StatusTest<double>>
        status = ROL::makePtr<ROL::StatusTest<double>>(*rol_params);
    ROL::Algorithm<double> algo(step, status, false);

    bool isProcZero = (PCU_Comm_Self() == 0);

    if (((obj_type == "adjoint") || (obj_type == "FS_VFM") || (obj_type == "Adjoint_VFM")) && check_gradient) {
      int const num_steps = 13;
      Array2D<double> fd_results;
      ROL::Ptr<Array1D<double>> d_ptr = ROL::makePtr<Array1D<double>>(dim, 0.1);
      ROL::StdVector<double> d(d_ptr);
      fd_results = (*rol_objective).checkGradient(x, d, isProcZero,
          *outStream, num_steps, 2);
      double min_error = std::numeric_limits<double>::max();
      double max_error = std::numeric_limits<double>::min();
      for (int i = 0; i < num_steps; ++i) {
        min_error = std::min(min_error, fd_results[i][3]);
        max_error = std::max(max_error, fd_results[i][3]);
      }
      double log10_mag_drop = std::log10(max_error / min_error);
      print("log10 of FD error magnitude drop = %.16e", log10_mag_drop);
    }

    Array1D<std::string> output;
    output = algo.run(x, *rol_objective, *bound, isProcZero, *outStream);
    for (int i = 0; i < output.size(); ++i) {
      *outStream << output[i];
    }

    if (isProcZero) {
      std::string outname = "ROL_out.txt";
      std::ofstream rolOut(outname);
      rolOut.precision(16);
      ROL::Ptr<Array1D<double> const> xp =
          (dynamic_cast<ROL::StdVector<double> const&>(x)).getVector();
      Array1D<double> const& opt_params =
          rol_objective->transform_params(*xp, false);
      for (int i = 0; i < output.size(); ++i) {
        *outStream << output[i];
        rolOut << output[i];
      }
      Array1D<std::string> const& elem_set_names =
          rol_objective->elem_set_names();
      Array2D<std::string> const& active_param_names =
          rol_objective->active_param_names();
      int const num_elem_sets = elem_set_names.size();
      int p = 0;
      for (int es = 0; es < num_elem_sets; ++es) {
        int const num_es_active_params = active_param_names[es].size();
        for (int i = 0; i < num_es_active_params; ++i) {
          *outStream << elem_set_names[es] << ": " << active_param_names[es][i]
              << " = " << opt_params[p] << "\n";
          rolOut << elem_set_names[es] << ": " << active_param_names[es][i]
              << " = " << opt_params[p] << "\n";
          ++p;
        }
      }
      rolOut.close();
    }

  }

  finalize();
}
