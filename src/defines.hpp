#pragma once

//! \file defines.hpp
//! \brief Code-wide definitions

#include <MatrixMarket_Tpetra.hpp>
#include <MiniTensor.h>
#include <Sacado_Fad_SLFad.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include "Eigen/Core"

namespace calibr8 {

//! \brief Local ordinal type
using LO = int;

//! \brief Global oridinal type
using GO = long long;

//! \brief The number of maximum derivatives for the FAD type
static constexpr int nmax_derivs = 16;

//! \brief Forward automatic differention type
using FADT = Sacado::Fad::SLFad<double, nmax_derivs>;

//! \brief The small dense linear algebra vector type
template <typename T>
using Vector = minitensor::Vector<T>;

//! \brief The small dense linear algebra tensor type
template <typename T>
using Tensor = minitensor::Tensor<T>;

//! \brief The small dense linear algebra Eigen vector
using EVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

//! \brief The small dense linear algebra Eigen matrix
using EMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

//! \brief The global (parallel) map type
using MapT = Tpetra::Map<LO, GO>;

//! \brief The global (parallel) graph type
using GraphT = Tpetra::CrsGraph<LO, GO>;

//! \brief The global (parallel) import type
using ImportT = Tpetra::Import<LO, GO>;

//! \brief The global (parallel) export type
using ExportT = Tpetra::Export<LO, GO>;

//! \brief The global (parallel) vector type
using VectorT = Tpetra::Vector<double, LO, GO>;

//! \brief The global (parallel) multi-vector type
using MultiVectorT = Tpetra::MultiVector<double, LO, GO>;

//! \brief The global (parallel) matrix type
using MatrixT = Tpetra::CrsMatrix<double, LO, GO>;

//! \brief The global (parallel) operator type
using OpT = Tpetra::Operator<double, LO, GO>;

//! \brief The global (parallel) matrix market writer type
using MMWriterT = Tpetra::MatrixMarket::Writer<MatrixT>;

//! \brief Reference counted pointer
using Teuchos::rcp;

//! \brief Reference counted pointer
using Teuchos::RCP;

//! \brief Parameter list
using Teuchos::ParameterList;

//! \brief A Teuchos MPI communication object
using Comm = Teuchos::Comm<int>;

}
