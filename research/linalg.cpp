#include "linalg.hpp"

namespace calibr8 {

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

}
