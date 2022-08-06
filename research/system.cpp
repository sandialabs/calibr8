#include "system.hpp"

namespace calibr8 {

void System::build_data(RCP<Disc> disc) {
  for (int space = 0; space < NUM_SPACE; ++space) {
    for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
      RCP<const MapT> map = disc->map(space, distrib);
      RCP<const GraphT> graph = disc->graph(space, distrib);
      A[space][distrib] = rcp(new MatrixT(graph));
      x[space][distrib] = rcp(new VectorT(map));
      b[space][distrib] = rcp(new VectorT(map));
    }
  }
}

void System::destroy_data() {
  for (int space = 0; space < NUM_SPACE; ++space) {
    for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
      A[space][distrib] = Teuchos::null;
      x[space][distrib] = Teuchos::null;
      b[space][distrib] = Teuchos::null;
    }
  }
}

void System::resume_fill(int space) {
  A[space][OWNED]->resumeFill();
  A[space][GHOST]->resumeFill();
}

void System::complete_fill(int space) {
  A[space][OWNED]->fillComplete();
  A[space][GHOST]->fillComplete();
}

void System::zero_A(int space) {
  A[space][OWNED]->setAllToScalar(0.);
  A[space][GHOST]->setAllToScalar(0.);
}

void System::zero_x(int space) {
  x[space][OWNED]->putScalar(0.);
  x[space][GHOST]->putScalar(0.);
}

void System::zero_b(int space) {
  b[space][OWNED]->putScalar(0.);
  b[space][GHOST]->putScalar(0.);
}

void System::gather_A(RCP<Disc> disc, int space, Tpetra::CombineMode mode) {
  RCP<const ExportT> exporter = disc->exporter(space);
  A[space][OWNED]->doExport(*(A[space][GHOST]), *exporter, mode);
}

void System::gather_x(RCP<Disc> disc, int space, Tpetra::CombineMode mode) {
  RCP<const ExportT> exporter = disc->exporter(space);
  x[space][OWNED]->doExport(*(x[space][GHOST]), *exporter, mode);
}

void System::gather_b(RCP<Disc> disc, int space, Tpetra::CombineMode mode) {
  RCP<const ExportT> exporter = disc->exporter(space);
  b[space][OWNED]->doExport(*(b[space][GHOST]), *exporter, mode);
}

}
