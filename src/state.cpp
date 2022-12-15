#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "state.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
RCP<Residuals<T>> create_residuals(ParameterList const& params, int ndims) {
  ParameterList const global_params = params.sublist("global residual");
  ParameterList const local_params = params.sublist("local residual");
  RCP<Residuals<T>> R = rcp(new Residuals<T>);
  R->global = create_global_residual<T>(global_params, ndims);
  int const model_form = 0;
  R->local[model_form] = create_local_residual<T>(local_params, ndims);
  if (params.isSublist("fine local residual")) {
    ParameterList const fine_local_params = params.sublist("fine local residual");
    int const fine_model_form = 1;
    R->local[fine_model_form] = create_local_residual<T>(fine_local_params, ndims);
  }
  return R;
}

template <typename T>
RCP<QoI<T>> create_qois(ParameterList const& params) {
  if (params.isSublist("quantity of interest")) {
    ParameterList const qoi_params = params.sublist("quantity of interest");
    return create_qoi<T>(qoi_params);
  }
  return Teuchos::null;
}

State::State(ParameterList const& params) {
  ParameterList const disc_params = params.sublist("discretization");
  ParameterList const resid_params = params.sublist("residuals");
  disc = rcp(new Disc(disc_params));
  la = rcp(new LinearAlg);
  residuals = create_residuals<double>(resid_params, disc->num_dims());
  d_residuals = create_residuals<FADT>(resid_params, disc->num_dims());
  qoi = create_qois<double>(params);
  d_qoi = create_qois<FADT>(params);
  disc->build_data(residuals->global->num_residuals(), residuals->global->num_eqs());
  la->build_data(disc);
}

}
