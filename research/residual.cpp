#include "residual.hpp"
#include "nlpoisson.hpp"

namespace calibr8 {

template <typename T>
Residual<T>::Residual() {
}

template <typename T>
Residual<T>::~Residual() {
}

template class Residual<double>;
template class Residual<FADT>;

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims) {
  std::string const type = params.get<std::string>("type");
  if (type == "nonlinear poisson") {
    return rcp(new NLPoisson<T>(params, ndims));
  } else {
    return Teuchos::null;
  }
}

template RCP<Residual<double>> create_residual(ParameterList const& params, int ndims);
template RCP<Residual<FADT>> create_residual(ParameterList const& params, int ndims);

}
