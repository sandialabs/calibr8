#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "disc.hpp"
#include "linear_solve.hpp"
#include "macros.hpp"

namespace calibr8 {

using MV = Tpetra::MultiVector<double, LO, GO>;
using OP = Tpetra::Operator<double, LO, GO>;
using RM = Tpetra::RowMatrix<double, LO, GO>;
using LinearProblem = Belos::LinearProblem<double, MV, OP>;
using Solver = Belos::SolverManager<double, MV, OP>;
using GmresSolver = Belos::BlockGmresSolMgr<double, MV, OP>;
using Prec = Tpetra::Operator<double, LO, GO>;

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<int>("krylov size", 0);
  p.set<int>("max iters", 0);
  p.set<double>("tolerance", 0.0);
  p.set<int>("output frequency", 0);
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tolerance", 0.0);
  p.set<double>("nonlinear relative tolerance", 0.0);
  p.sublist("multigrid");
  return p;
}

static ParameterList get_belos_params(ParameterList const& params) {
  ParameterList p;
  int max_iters = params.get<int>("max iters");
  int krylov = params.get<int>("krylov size");
  double tol = params.get<double>("tolerance");
  p.set<int>("Block Size" , 1);
  p.set<int>("Num Blocks", krylov);
  p.set<int>("Maximum Iterations", max_iters);
  p.set<double>("Convergence Tolerance", tol);
  p.set<std::string>("Orthogonalization", "DGKS");
  if (params.isType<int>("output frequency")) {
    int f = params.get<int>("output frequency");
    p.set<int>("Verbosity", 33);
    p.set<int>("Output Style", 1);
    p.set<int>("Output Frequency", f);
  }
  return p;
}

static RCP<Solver> build_solver(
    ParameterList const& params,
    RCP<Disc> d,
    RCP<MatrixT> A,
    RCP<MultiVectorT> x,
    RCP<MultiVectorT> b) {
  Teuchos::ParameterList mg_params(params.sublist("multigrid"));
  auto belos_params = get_belos_params(params);
  /* causing an error when a vector in dRdp is zero */
  //belos_params.set<int>("Block Size", b->getNumVectors());
  auto AA = (RCP<OP>)A;
  auto coords = d->coords();
  auto& user_data = mg_params.sublist("user data");
  user_data.set<RCP<MultiVectorT>>("Coordinates", coords);
  auto P = MueLu::CreateTpetraPreconditioner(AA, mg_params);
  auto problem = rcp(new LinearProblem(A, x, b));
  //problem->setLeftPrec(P);
  /* right preconditioner seems to work better with new
   * MueLu settings */
  problem->setRightPrec(P);
  problem->setProblem();
  return rcp(new GmresSolver(problem, rcpFromRef(belos_params)));
}

void solve(
    ParameterList& params,
    RCP<Disc> disc,
    RCP<MatrixT>& A,
    RCP<VectorT>& x,
    RCP<VectorT>& b) {

  params.validateParameters(get_valid_params(), 0);
  auto solver = build_solver(params, disc, A, x, b);
  solver->solve();

}

}
