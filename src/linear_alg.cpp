#include "linear_alg.hpp"

namespace calibr8 {

void LinearAlg::build_data(RCP<Disc> disc) {
  m_disc = disc;
  int const nr = disc->num_residuals();
  for (int d = 0; d < NUM_DISTRIB; ++d) {
    resize(A[d], nr, nr);
    resize(x[d], nr);
    resize(b[d], nr);
    for (int i = 0; i < nr; ++i) {
      RCP<const MapT> map = disc->map(d, i);
      x[d][i] = rcp(new VectorT(map));
      b[d][i] = rcp(new VectorT(map));
      for (int j = 0; j < nr; ++j) {
        RCP<const GraphT> graph = m_disc->graph(d, i, j);
        A[d][i][j] = rcp(new MatrixT(graph));
      }
    }
  }
}

void LinearAlg::destroy_data() {
  m_disc = Teuchos::null;
  for (int d = 0; d < NUM_DISTRIB; ++d) {
    resize(A[d], 0, 0);
    resize(x[d], 0);
    resize(b[d], 0);
  }
}

void LinearAlg::resume_fill_A() {
  int const ngr = A[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    for (int j = 0; j < ngr; ++j) {
      A[OWNED][i][j]->resumeFill();
      A[GHOST][i][j]->resumeFill();
    }
  }
}

void LinearAlg::complete_fill_A() {
  int const ngr = A[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    for (int j = 0; j < ngr; ++j) {
      A[OWNED][i][j]->fillComplete();
      A[GHOST][i][j]->fillComplete();
    }
  }
}

void LinearAlg::gather_A() {
  int const ngr = A[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    for (int j = 0; j < ngr; ++j) {
      RCP<MatrixT> Aij_owned = A[OWNED][i][j];
      RCP<MatrixT> Aij_ghost = A[GHOST][i][j];
      RCP<const ExportT> exporter = m_disc->exporter(i);
      Aij_owned->doExport(*Aij_ghost, *exporter, Tpetra::ADD);
    }
  }
}

void LinearAlg::gather_x(bool sum) {
  Tpetra::CombineMode mode;
  if (sum) mode = Tpetra::ADD;
  else mode = Tpetra::INSERT;
  int const ngr = x[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    RCP<VectorT> x_i_owned = x[OWNED][i];
    RCP<VectorT> x_i_ghost = x[GHOST][i];
    RCP<const ExportT> exporter = m_disc->exporter(i);
    x_i_owned->doExport(*x_i_ghost, *exporter, mode);
  }
}

void LinearAlg::gather_b() {
  int const ngr = b[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    RCP<VectorT> b_i_owned = b[OWNED][i];
    RCP<VectorT> b_i_ghost = b[GHOST][i];
    RCP<const ExportT> exporter = m_disc->exporter(i);
    b_i_owned->doExport(*b_i_ghost, *exporter, Tpetra::ADD);
  }
}

void LinearAlg::assign_b() {
  int const ngr = b[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    RCP<VectorT> b_i_owned = b[OWNED][i];
    RCP<VectorT> b_i_ghost = b[GHOST][i];
    RCP<const ExportT> exporter = m_disc->exporter(i);
    b_i_owned->doExport(*b_i_ghost, *exporter, Tpetra::INSERT);
  }
}

void LinearAlg::zero_A() {
  int const ngr = A[OWNED].size();
  for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
    for (int i = 0; i < ngr; ++i) {
      for (int j = 0; j < ngr; ++j) {
        A[distrib][i][j]->setAllToScalar(0.);
      }
    }
  }
}

void LinearAlg::zero_b() {
  int const ngr = b[OWNED].size();
  for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
    for (int i = 0; i < ngr; ++i) {
      b[distrib][i]->putScalar(0.);
    }
  }
}

void LinearAlg::zero_all() {
  int const ngr = A[OWNED].size();
  for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
    for (int i = 0; i < ngr; ++i) {
      x[distrib][i]->putScalar(0.);
      b[distrib][i]->putScalar(0.);
      for (int j = 0; j < ngr; ++j) {
        A[distrib][i][j]->setAllToScalar(0.);
      }
    }
  }
}

void LinearAlg::scale_b(double val) {
  int const ngr = b[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    b[OWNED][i]->scale(val);
  }
}

double LinearAlg::norm_b() {
  double norm = 0;
  int const ngr = b[OWNED].size();
  for (int i = 0; i < ngr; ++i) {
    double const norm_bi = b[OWNED][i]->norm2();
    norm += norm_bi * norm_bi;
  }
  return std::sqrt(norm);
}

}
