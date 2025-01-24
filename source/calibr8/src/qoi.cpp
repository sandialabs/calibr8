#include "avg_disp.hpp"
#include "avg_stress.hpp"
#include "calibration.hpp"
#include "disc.hpp"
#include "disp_comp.hpp"
#include "fad.hpp"
#include "load_mismatch.hpp"
#include "normal_traction.hpp"
#include "point_wise.hpp"
#include "qoi.hpp"
#include "reaction_mismatch.hpp"
#include "surface_mismatch.hpp"

namespace calibr8 {

template <typename T>
QoI<T>::QoI() {
}

template <typename T>
QoI<T>::~QoI() {
}

template <typename T>
void QoI<T>::before_elems(RCP<Disc> disc, int step) {

  // set discretization-based information
  m_mesh = disc->apf_mesh();
  m_num_dims = disc->num_dims();
  m_shape = disc->gv_shape();
  m_step = step;

}

template <typename T>
void QoI<T>::set_elem(apf::MeshElement* mesh_elem) {
  m_mesh_elem = mesh_elem;
}

template <typename T>
void QoI<T>::scatter(double& J) {
  J += val(value_pt);
}

template <typename T>
void QoI<T>::preprocess(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double w,
    double dv) {}

template <typename T>
void QoI<T>::preprocess_finalize(int step) {}

template <typename T>
void QoI<T>::postprocess(double& J) {}

template <typename T>
void QoI<T>::modify_state(RCP<State>) {}

template <>
EVector QoI<double>::eigen_dvector(int) const {
  EVector empty;
  return empty;
}

template <>
EVector QoI<FADT>::eigen_dvector(int nderivs) const {
  EVector dJ(nderivs);
  dJ.setZero();
  for (int i = 0; i < nderivs; ++i) {
    dJ[i] = value_pt.fastAccessDx(i);
  }
  return dJ;
}

template <typename T>
void QoI<T>::unset_elem() {
  m_mesh_elem = nullptr;
}

template <typename T>
void QoI<T>::after_elems() {
  m_num_dims = -1;
  m_step = -1;
  m_mesh = nullptr;
  m_shape = nullptr;
}

template <>
void QoI<double>::initialize_value_pt() {
  value_pt = 0.;
}

template <>
void QoI<FADT>::initialize_value_pt() {
  value_pt = 0.;
  for (int idx = 0.; idx < nmax_derivs; ++idx) {
    value_pt.fastAccessDx(idx) = 0.;
  }
}

template <typename T>
RCP<QoI<T>> create_qoi(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "average displacement") {
    return rcp(new AvgDisp<T>());
  } else if (type == "displacement component") {
    return rcp(new DispComp<T>(params));
  } else if (type == "average stress") {
    return rcp(new AvgStress<T>(params));
  } else if (type == "surface mismatch") {
    return rcp(new SurfaceMismatch<T>(params));
  } else if (type == "load mismatch") {
    return rcp(new LoadMismatch<T>(params));
  } else if (type == "calibration") {
    return rcp(new Calibration<T>(params));
  } else if (type == "normal traction") {
    return rcp(new NormalTraction<T>(params));
  } else if (type == "point displacement") {
    return rcp(new PointWise<T>(params));
  } else if (type == "reaction mismatch") {
    return rcp(new ReactionMismatch<T>(params));
  } else {
    return Teuchos::null;
  }
}

template class QoI<double>;
template class QoI<FADT>;

template RCP<QoI<double>>
create_qoi<double>(ParameterList const&);

template RCP<QoI<FADT>>
create_qoi<FADT>(ParameterList const&);

}
