#pragma once

#include "disc.hpp"

namespace calibr8 {

class Vector {
  private:
    int m_space;
    RCP<Disc> m_disc;
  public:
    RCP<VectorT> val[NUM_DISTRIB];
  public:
    Vector(int space, RCP<Disc> disc);
    void zero();
    void gather(Tpetra::CombineMode mode);
    void scatter(Tpetra::CombineMode mode);
};

class Matrix {
  private:
    int m_space;
    RCP<Disc> m_disc;
  public:
    RCP<MatrixT> val[NUM_DISTRIB];
  public:
    Matrix(int space, RCP<Disc> disc);
    void begin_fill();
    void zero();
    void gather(Tpetra::CombineMode mode);
    void end_fill();
};


class System {
  public:
    RCP<MatrixT> A;
    RCP<VectorT> x;
    RCP<VectorT> b;
  public:
    System() = default;
    System(int distrib, Matrix& A_in, Vector& x_in, Vector& b_in);
};

}
