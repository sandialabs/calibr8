#include "linear_alg.hpp"

namespace calibr8 {

void LinearAlg::build_data(RCP<Disc> disc) {
  m_disc = disc;
  for (int d = 0; d < NUM_DISTRIB; ++d) {
    RCP<const MapT> map = disc->map(d);
    x[d] = rcp(new VectorT(map));
    b[d] = rcp(new VectorT(map));

    RCP<const GraphT> graph = m_disc->graph(d);
    A[d] = rcp(new MatrixT(graph));
  }
}

void LinearAlg::destroy_data() {
  m_disc = Teuchos::null;
}

void LinearAlg::resume_fill_A() {
  A[OWNED]->resumeFill();
  A[GHOST]->resumeFill();
}

void LinearAlg::complete_fill_A() {
  A[OWNED]->fillComplete();
  A[GHOST]->fillComplete();
}

void LinearAlg::gather_A() {
  RCP<MatrixT> A_owned = A[OWNED];
  RCP<MatrixT> A_ghost = A[GHOST];
  RCP<const ExportT> exporter = m_disc->exporter();
  A_owned->doExport(*A_ghost, *exporter, Tpetra::ADD);
}

void LinearAlg::gather_x() {
  RCP<VectorT> x_owned = x[OWNED];
  RCP<VectorT> x_ghost = x[GHOST];
  RCP<const ExportT> exporter = m_disc->exporter();
  x_owned->doExport(*x_ghost, *exporter, Tpetra::ADD);
}

void LinearAlg::gather_b() {
  RCP<VectorT> b_owned = b[OWNED];
  RCP<VectorT> b_ghost = b[GHOST];
  RCP<const ExportT> exporter = m_disc->exporter();
  b_owned->doExport(*b_ghost, *exporter, Tpetra::ADD);
}

void LinearAlg::assign_b() {
  RCP<VectorT> b_owned = b[OWNED];
  RCP<VectorT> b_ghost = b[GHOST];
  RCP<const ExportT> exporter = m_disc->exporter();
  b_owned->doExport(*b_ghost, *exporter, Tpetra::INSERT);
}

void LinearAlg::zero_all() {
  for (int d = 0; d < NUM_DISTRIB; ++d) {
    x[d]->putScalar(0.);
    b[d]->putScalar(0.);
    A[d]->setAllToScalar(0.);
  }
}

void LinearAlg::scale_b(double val) {
  b[OWNED]->scale(val);
}

double LinearAlg::norm_b() {
  return b[OWNED]->norm2();
}

}
