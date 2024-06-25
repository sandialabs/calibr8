#include <Teko_Utilities.hpp>
#include <Teko_StratimikosFactory.hpp>
#include <Thyra_DefaultBlockedLinearOp.hpp>
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_MueLuPreconditionerFactory.hpp>
#include <Thyra_PreconditionerFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraVectorSpace.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include "disc.hpp"
#include "linear_solve.hpp"
#include "macros.hpp"

namespace calibr8 {

using BlockOpT = Thyra::DefaultBlockedLinearOp<double>;
using ThyraVecT = Thyra::MultiVectorBase<double>;
using LinearOpT = Thyra::LinearOpBase<double>;
using NodeT = VectorT::node_type;

void solve(
    ParameterList& params,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& A,
    Array1D<RCP<VectorT>>& x,
    Array1D<RCP<VectorT>>& b) {

  int const nblocks = disc->num_residuals();

  DEBUG_ASSERT(A.size() == size_t(nblocks));
  DEBUG_ASSERT(x.size() == size_t(nblocks));
  DEBUG_ASSERT(b.size() == size_t(nblocks));

  // set up a thyra block operator
  Teuchos::RCP<BlockOpT> A_thyra = Thyra::defaultBlockedLinearOp<double>();
  A_thyra->beginBlockFill(nblocks, nblocks);
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      RCP<MatrixT> A_ij = A[i][j];
      auto range = Thyra::tpetraVectorSpace<double>(A_ij->getRangeMap());
      auto domain = Thyra::tpetraVectorSpace<double>(A_ij->getDomainMap());
      auto crs_as_op = Teuchos::rcp_implicit_cast<OpT>(A_ij);
      auto tpetra_op = Thyra::tpetraLinearOp<double>(range, domain, crs_as_op);
      A_thyra->setBlock(i, j, tpetra_op);
    }
  }
  A_thyra->endBlockFill();

  // set up a thyra right hand side
  Array1D<Teuchos::RCP<ThyraVecT>> b_thyra(nblocks);
  for (int i = 0; i < nblocks; ++i) {
    Teuchos::RCP<const Thyra::TpetraVectorSpace<double, LO, GO, NodeT>> range =
      Thyra::tpetraVectorSpace<double>(A[i][i]->getRangeMap());
    b_thyra[i] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<VectorT>(b[i]));
  }

  // set up a teko solution multivector
  Array1D<Teko::MultiVector> x_teko(nblocks);
  for (int i = 0; i < nblocks; ++i) {
    Teuchos::RCP<const Thyra::TpetraVectorSpace<double, LO, GO, NodeT>> domain =
      Thyra::tpetraVectorSpace<double>(A[i][i]->getDomainMap());
    x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<VectorT>(x[i]));
    DEBUG_ASSERT(! x_teko[i].is_null());
  }

  // set up final teko form of vectors
  Thyra::SolveStatus<double> status;
  Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(b_thyra);
  Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

  // proceed with the ickyness

  Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> linearSolverBuilder =
    Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);

  typedef Thyra::PreconditionerFactoryBase<double> PreconditionerBase;
  typedef Thyra::Ifpack2PreconditionerFactory<MatrixT> Ifpack2Impl;

  linearSolverBuilder->setPreconditioningStrategyFactory(
      Teuchos::abstractFactoryStd<PreconditionerBase, Ifpack2Impl>(), "Ifpack2");
  Stratimikos::enableMueLu<LO, GO, NodeT>(*linearSolverBuilder, "MueLu");

  Teuchos::RCP<Teko::RequestHandler> requestHandler_;
  Teko::addTekoToStratimikosBuilder(*linearSolverBuilder, requestHandler_);

  RCP<ParameterList> params_ptr = rcp(&params, false);
  linearSolverBuilder->setParameterList(params_ptr);

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lowsFactory
    = linearSolverBuilder->createLinearSolveStrategy("");

  Teuchos::RCP<Teuchos::FancyOStream> out =
    Teuchos::VerboseObjectBase::getDefaultOStream();
  lowsFactory->setOStream(out);
  lowsFactory->setVerbLevel(Teuchos::VERB_HIGH);

  Teuchos::RCP<Thyra::LinearOpWithSolveBase<double> > th_invA =
    Thyra::linearOpWithSolve(
        *lowsFactory,
        Teuchos::rcp_dynamic_cast<const LinearOpT>(A_thyra));

  Thyra::assign(Thyra_x.ptr(), 0.0);
  status = Thyra::solve<double>(
      *th_invA, Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

}

}
