#pragma once

#include <MatrixMarket_Tpetra.hpp>
#include <Sacado_Fad_SLFad.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace calibr8 {

using LO = int;
using GO = long long;

// this is specific to scalar equations in 2D with
// triangles using P2 Lagrange basis functions
static constexpr int nmax_derivs = 6;

using FADT = Sacado::Fad::SLFad<double, nmax_derivs>;
using FAD2T = Sacado::Fad::SLFad<FADT, nmax_derivs>;

using MapT = Tpetra::Map<LO, GO>;
using GraphT = Tpetra::CrsGraph<LO, GO>;
using ImportT = Tpetra::Import<LO, GO>;
using ExportT = Tpetra::Export<LO, GO>;
using VectorT = Tpetra::Vector<double, LO, GO>;
using MultiVectorT = Tpetra::MultiVector<double, LO, GO>;
using MatrixT = Tpetra::CrsMatrix<double, LO, GO>;
using MMWriterT = Tpetra::MatrixMarket::Writer<MatrixT>;

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ParameterList;

using Comm = Teuchos::Comm<int>;

enum {OWNED=0, GHOST=1, NUM_DISTRIB=2};
enum {COARSE=0, FINE=1, NUM_SPACE=2};

}
