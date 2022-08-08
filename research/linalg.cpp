#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"

namespace calibr8 {

using MV = Tpetra::MultiVector<double, LO, GO>;
using OP = Tpetra::Operator<double, LO, GO>;
using RM = Tpetra::RowMatrix<double, LO, GO>;
using LinearProblem = Belos::LinearProblem<double, MV, OP>;
using Solver = Belos::SolverManager<double, MV, OP>;
using GmresSolver = Belos::BlockGmresSolMgr<double, MV, OP>;
using Prec = Tpetra::Operator<double, LO, GO>;

Vector::Vector(int space, RCP<Disc> disc) :
  m_space(space),
  m_disc(disc) {
    RCP<const MapT> owned_map = m_disc->map(m_space, OWNED);
    RCP<const MapT> ghost_map = m_disc->map(m_space, GHOST);
    val[OWNED] = rcp(new VectorT(owned_map));
    val[GHOST] = rcp(new VectorT(ghost_map));
}

void Vector::zero() {
  val[OWNED]->putScalar(0.);
  val[GHOST]->putScalar(0.);
}

void Vector::gather(Tpetra::CombineMode mode) {
  RCP<const ExportT> exporter = m_disc->exporter(m_space);
  val[OWNED]->doExport(*(val[GHOST]), *exporter, mode);
}

void Vector::scatter(Tpetra::CombineMode mode) {
  RCP<const ImportT> importer = m_disc->importer(m_space);
  val[GHOST]->doExport(*(val[OWNED]), *importer, mode);
}

Matrix::Matrix(int space, RCP<Disc> disc) :
  m_space(space),
  m_disc(disc) {
    RCP<const GraphT> owned_graph = m_disc->graph(m_space, OWNED);
    RCP<const GraphT> ghost_graph = m_disc->graph(m_space, GHOST);
    val[OWNED] = rcp(new MatrixT(owned_graph));
    val[GHOST] = rcp(new MatrixT(ghost_graph));
}

void Matrix::begin_fill() {
  val[OWNED]->resumeFill();
  val[GHOST]->resumeFill();
}

void Matrix::zero() {
  val[OWNED]->setAllToScalar(0.);
  val[GHOST]->setAllToScalar(0.);
}

void Matrix::gather(Tpetra::CombineMode mode) {
  RCP<const ExportT> exporter = m_disc->exporter(m_space);
  val[OWNED]->doExport(*(val[GHOST]), *exporter, mode);
}

System::System(int distrib, Matrix& A_in, Vector& x_in, Vector& b_in) {
  A = A_in.val[distrib];
  x = x_in.val[distrib];
  b = b_in.val[distrib];
}

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

static ParameterList get_belos_params(ParameterList const& in) {
  ParameterList p;
  int max_iters = in.get<int>("max iters");
  int krylov = in.get<int>("krylov size");
  double tol = in.get<double>("tolerance");
  p.set<int>("Block Size" , 1);
  p.set<int>("Num Blocks", krylov);
  p.set<int>("Maximum Iterations", max_iters);
  p.set<double>("Convergence Tolerance", tol);
  p.set<std::string>("Orthogonalization", "DGKS");
  if (in.isType<int>("output frequency")) {
    int f = in.get<int>("output frequency");
    p.set<int>("Verbosity", 33);
    p.set<int>("Output Style", 1);
    p.set<int>("Output Frequency", f);
  }
  return p;
}

static RCP<Solver> build_solver(
    ParameterList const& in,
    int space,
    RCP<Disc> d,
    RCP<MatrixT> A,
    RCP<MultiVectorT> x,
    RCP<MultiVectorT> b) {
  Teuchos::ParameterList mg_params(in.sublist("multigrid"));
  auto belos_params = get_belos_params(in);
  belos_params.set<int>("Block Size", b->getNumVectors());
  auto AA = (RCP<OP>)A;
  auto coords = d->coords(space);
  auto& user_data = mg_params.sublist("user data");
  user_data.set<RCP<MultiVectorT>>("Coordinates", coords);
  auto P = MueLu::CreateTpetraPreconditioner(AA, mg_params);
  auto problem = rcp(new LinearProblem(A, x, b));
  problem->setRightPrec(P);
  problem->setProblem();
  return rcp(new GmresSolver(problem, rcpFromRef(belos_params)));
}

void solve(
    ParameterList& in,
    int space,
    RCP<Disc> disc,
    System& sys) {
  in.validateParameters(get_valid_params(), 0);
  auto solver = build_solver(in, space, disc, sys.A, sys.x, sys.b);
  solver->solve();
  auto dofs = solver->getProblem().getRHS()->getGlobalLength();
  print(" > linear system: num dofs %zu", dofs);
  auto iters = solver->getNumIters();
  print(" > linear system: solved in %d iterations", iters);
  if (iters >= in.get<int>("max iters")) {
    print(" >  but solve was incomplete!");
  }
}

}
