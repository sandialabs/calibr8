#pragma once

#include "disc.hpp"

namespace calibr8 {

class System {
  public:
    RCP<MatrixT> A[NUM_SPACE][NUM_DISTRIB];
    RCP<VectorT> x[NUM_SPACE][NUM_DISTRIB];
    RCP<VectorT> b[NUM_SPACE][NUM_DISTRIB];
  public:
    void build_data(RCP<Disc> disc);
    void destroy_data();
    void resume_fill(int space);
    void complete_fill(int space);
    void zero_A(int space);
    void zero_x(int space);
    void zero_b(int space);
    void gather_A(RCP<Disc> disc, int space, Tpetra::CombineMode mode);
    void gather_x(RCP<Disc> disc, int space, Tpetra::CombineMode mode);
    void gather_b(RCP<Disc> disc, int space, Tpetra::CombineMode mode);
};

}
