#include "ICNN.hpp"

#include <stdexcept>

namespace ML {

template <class T>
T softplus(T const& x)
{
  return std::log(1. + std::exp(x));
}

static void check_topo(std::vector<int> const& topo)
{
  if (topo.size() < 3) {
    throw std::runtime_error("FICNN: not enough layers");
  }
}

template <class T>
FICNN<T>::FICNN(
    const char* act,
    std::vector<int> const& topology)
{
  std::string const act_type = act;
  if (act_type == "softplus") activation = softplus;
  else throw std::runtime_error("FICNN: unknown " + act_type);
  num_vectors = int(topo.size());
  Wy.resize(num_vectors-1);
  Wz.resize(num_vectors-1);
  b.resize(num_vectors-1);
  x.resize(num_vectors);
  int const end = num_vectors-1;


}

template <class T>
typename FICNN<T>::Vector FICNN<T>::evaluate(Vector const& x)
{
  (void)x;
  Vector f;
  return f;
}

template class FICNN<RFAD_SFADT>;
template class FICNN<RFAD_DFADT>;

}
