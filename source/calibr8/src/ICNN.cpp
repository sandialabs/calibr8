#include "ICNN.hpp"

namespace ML {

template <class T>
ICNN<T>::ICNN()
{
  printf("constructing an input convex neural network\n");
}

template class ICNN<RFAD_SFADT>;

}
